"""
RoboCasa365 (LeRobot v3.0) -> UniVLA batch dataset.  [agent-B, PLAN S1]

- 直读 parquet (state/action) + AV1 mp4 (imageio FFMPEG 插件).
- 不在 dataloader 里跑 LAM;  返回帧对, 训练 loop 做 vq_encode (照 finetune_realworld.py 契约).
- V1 / V2 只在 `compose_image()` 一处不同, 其余完全一致.
- LAM 输入始终是 left 单视角 (both variants) -> <ACT> 伪标签在 V1/V2 间逐位相同.
"""
import os, re, json, random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import imageio


def split_camel(s: str) -> str:
    s = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', s)
    s = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', s)
    return s.lower()


class _VideoCache:
    """按需打开 mp4 reader 并缓存 (进程内). imageio FFMPEG 插件解 AV1."""
    def __init__(self):
        self._readers = {}

    def frame(self, path: str, idx: int) -> np.ndarray:
        rd = self._readers.get(path)
        if rd is None:
            rd = imageio.get_reader(path, format="FFMPEG")
            self._readers[path] = rd
        return rd.get_data(int(idx))  # (H,W,3) uint8

    def close(self):
        for rd in self._readers.values():
            try: rd.close()
            except Exception: pass
        self._readers.clear()


class RoboCasaLeRobotDataset(Dataset):
    def __init__(
        self,
        data_root: str,                 # .../robocasa_target_human_unified
        task_names,                     # e.g. ["NavigateKitchen", "CloseToasterOvenDoor"]
        n_demos_per_task: int,
        window_size: int,
        vision_variant: str,            # "v1" | "v2"
        image_transform,               # PrismaticImageProcessor.apply_transform  -> (6,224,224)
        proprio_mean, proprio_std,     # np.array(16,)  (来自 unified stats.json)
        v2_each_size: int = 224,       # V2: 每路先 resize 到 (v2_each_size, v2_each_size)
        camelcase_split: bool = True,
        seed: int = 7,
        cache_frames_in_ram: bool = False,   # round-1 可 True 提速; round-2 关掉
    ):
        super().__init__()
        assert vision_variant in ("v1", "v2")
        import pandas as pd
        self.data_root = data_root
        self.window_size = window_size
        self.vision_variant = vision_variant
        self.image_transform = image_transform
        self.v2_each = v2_each_size
        self.camelcase_split = camelcase_split
        self.proprio_mean = np.asarray(proprio_mean, dtype=np.float32)
        self.proprio_std = np.asarray(proprio_std, dtype=np.float32).copy()
        self.proprio_std[self.proprio_std < 1e-6] = 1.0  # idx3,4 std=0 -> 不缩放
        self._rng = np.random.default_rng(seed)
        self._vc = _VideoCache()
        self._ram = cache_frames_in_ram
        self._ram_store = {}

        meta_ep = pd.read_parquet(os.path.join(data_root, "meta", "episodes.parquet"))
        meta_ep["task"] = meta_ep["tasks"].apply(lambda x: x[0])
        tasks_df = pd.read_parquet(os.path.join(data_root, "meta", "tasks.parquet")).reset_index()
        self._name2tidx = {r.task: int(r.task_index) for r in tasks_df.itertuples()}

        # 选 episode
        self.episodes = []   # list of dict
        data_files_needed = set()
        for tn in task_names:
            sub = meta_ep[meta_ep["task"] == tn].sort_values("episode_index").head(n_demos_per_task)
            for _, r in sub.iterrows():
                lv = (int(r["videos/observation.images.robot0_agentview_left/chunk_index"]),
                      int(r["videos/observation.images.robot0_agentview_left/file_index"]))
                wv = (int(r["videos/observation.images.robot0_eye_in_hand/chunk_index"]),
                      int(r["videos/observation.images.robot0_eye_in_hand/file_index"]))
                ep = dict(
                    task=tn,
                    task_index=self._name2tidx[tn],
                    episode_index=int(r["episode_index"]),
                    length=int(r["length"]),
                    data_from=int(r["dataset_from_index"]),
                    data_to=int(r["dataset_to_index"]),
                    data_cf=(int(r["data/chunk_index"]), int(r["data/file_index"])),
                    left_cf=lv,
                    left_f0=round(float(r["videos/observation.images.robot0_agentview_left/from_timestamp"]) * 20),
                    wrist_cf=wv,
                    wrist_f0=round(float(r["videos/observation.images.robot0_eye_in_hand/from_timestamp"]) * 20),
                    instruction=(split_camel(tn) if camelcase_split else tn.lower()),
                )
                self.episodes.append(ep)
                data_files_needed.add(ep["data_cf"])

        # 载入需要的 data parquet, 建 episode_index -> (action[T,12], state[T,16]) 映射
        self._ep_data = {}
        for (ci, fi) in sorted(data_files_needed):
            p = os.path.join(data_root, "data", f"chunk-{ci:03d}", f"file-{fi:03d}.parquet")
            df = pd.read_parquet(p, columns=["action", "observation.state", "frame_index", "episode_index"])
            for ei, g in df.groupby("episode_index"):
                g = g.sort_values("frame_index")
                self._ep_data[int(ei)] = (
                    np.stack(g["action"].values).astype(np.float32),          # (T,12)
                    np.stack(g["observation.state"].values).astype(np.float32),  # (T,16)
                )
        # 只保留有 data 的 episode
        self.episodes = [e for e in self.episodes if e["episode_index"] in self._ep_data]
        print(f"[RoboCasaLeRobotDataset] variant={vision_variant} tasks={task_names} "
              f"-> {len(self.episodes)} episodes, "
              f"total frames={sum(e['length'] for e in self.episodes)}")

    def __len__(self):
        return len(self.episodes)

    # ---- 图像 ----
    def _get_frame(self, cam: str, ep: dict, frame_in_ep: int) -> Image.Image:
        if cam == "left":
            ci, fi = ep["left_cf"]; f = ep["left_f0"] + frame_in_ep
            sub = "robot0_agentview_left"
        else:
            ci, fi = ep["wrist_cf"]; f = ep["wrist_f0"] + frame_in_ep
            sub = "robot0_eye_in_hand"
        path = os.path.join(self.data_root, "videos", f"observation.images.{sub}",
                            f"chunk-{ci:03d}", f"file-{fi:03d}.mp4")
        if self._ram:
            key = (path, f)
            arr = self._ram_store.get(key)
            if arr is None:
                arr = self._vc.frame(path, f); self._ram_store[key] = arr
        else:
            arr = self._vc.frame(path, f)
        return Image.fromarray(arr).convert("RGB")

    def compose_image(self, ep: dict, frame_in_ep: int) -> Image.Image:
        """V1: left 原样(256) ; V2: left+wrist 各 resize -> hconcat -> resize 224. 唯一的 V1/V2 差异点."""
        left = self._get_frame("left", ep, frame_in_ep)
        if self.vision_variant == "v1":
            return left
        wrist = self._get_frame("wrist", ep, frame_in_ep)
        s = self.v2_each
        cat = Image.new("RGB", (2 * s, s))
        cat.paste(left.resize((s, s), Image.BILINEAR), (0, 0))
        cat.paste(wrist.resize((s, s), Image.BILINEAR), (s, 0))
        return cat.resize((224, 224), Image.BILINEAR)

    def _lam_frame(self, ep: dict, frame_in_ep: int) -> torch.Tensor:
        """LAM 输入: 始终 left 单视角, resize 224, ToTensor [0,1]. (V1/V2 相同 -> <ACT> 逐位相同)"""
        img = self._get_frame("left", ep, frame_in_ep).resize((224, 224), Image.BILINEAR)
        return torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)

    def __getitem__(self, idx: int):
        ep = self.episodes[idx]
        act, state = self._ep_data[ep["episode_index"]]
        T = min(len(act), ep["length"])
        ws = self.window_size
        extra = int(self._rng.integers(0, 2))          # 0/1, 照 real_world 的 history 机制
        need = ws + extra
        t0 = int(self._rng.integers(0, max(1, T - need)))   # clip 起点

        # 动作块 (ws,12), 不归一化 (已 Box(-1,1)); 末尾不足则 pad 最后一帧
        end = min(T, t0 + ws)
        chunk = np.zeros((ws, 12), dtype=np.float32)
        chunk[:end - t0] = act[t0:end]
        if end - t0 < ws:
            chunk[end - t0:] = act[end - 1]
        is_pad = np.zeros((ws,), dtype=np.float32); is_pad[:end - t0] = 1.0
        control_mode_tgt = (chunk[:, 4] > 0).astype(np.int64)   # (ws,)

        # proprio: state[t0] z-score
        proprio = (state[t0] - self.proprio_mean) / self.proprio_std

        # VLA 图像 (V1/V2)
        pil_cur = self.compose_image(ep, t0)
        pixel_values = self.image_transform(pil_cur)            # (6,224,224)

        # LAM 帧对 (left 单视角)
        init_pv = self._lam_frame(ep, t0)
        tgt_pv = self._lam_frame(ep, min(T - 1, t0 + ws))
        init_pv_hist = tgt_pv_hist = None
        if extra > 0:
            init_pv_hist = self._lam_frame(ep, min(T - 1, t0 + 1))
            tgt_pv_hist = self._lam_frame(ep, min(T - 1, t0 + 1 + ws))

        return dict(
            pixel_values=pixel_values,
            initial_pixel_values=init_pv,
            target_pixel_values=tgt_pv,
            initial_pixel_values_hist=init_pv_hist,
            target_pixel_values_hist=tgt_pv_hist,
            actions=torch.from_numpy(chunk),                    # (ws,12) float
            control_mode_tgt=torch.from_numpy(control_mode_tgt),  # (ws,) long
            action_is_pad=torch.from_numpy(is_pad),             # (ws,)
            proprio=torch.from_numpy(proprio.astype(np.float32)),  # (16,)
            instruction=ep["instruction"],
            task_index=ep["task_index"],
            dataset_name="robocasa365",
        )


def collate(instances):
    def stk(key):
        return torch.stack([x[key] for x in instances], dim=0)
    has_hist = [x["initial_pixel_values_hist"] is not None for x in instances]
    out = dict(
        pixel_values=stk("pixel_values"),
        initial_pixel_values=stk("initial_pixel_values"),
        target_pixel_values=stk("target_pixel_values"),
        actions=stk("actions"),
        control_mode_tgt=stk("control_mode_tgt"),
        action_is_pad=stk("action_is_pad"),
        proprio=stk("proprio"),
        instructions=[x["instruction"] for x in instances],
        task_index=[x["task_index"] for x in instances],
        dataset_names=[x["dataset_name"] for x in instances],
        with_hist=torch.tensor(has_hist),
    )
    if any(has_hist):
        out["initial_pixel_values_hist"] = torch.stack(
            [x["initial_pixel_values_hist"] for x in instances if x["initial_pixel_values_hist"] is not None])
        out["target_pixel_values_hist"] = torch.stack(
            [x["target_pixel_values_hist"] for x in instances if x["target_pixel_values_hist"] is not None])
    else:
        out["initial_pixel_values_hist"] = []
        out["target_pixel_values_hist"] = []
    return out


def load_proprio_stats(data_root):
    s = json.load(open(os.path.join(data_root, "meta", "stats.json")))
    return (np.asarray(s["observation.state"]["mean"], dtype=np.float32),
            np.asarray(s["observation.state"]["std"], dtype=np.float32))
