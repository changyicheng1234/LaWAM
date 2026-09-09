"""
LeRobot-v3.0 -> LAM frame-pair dataset.  Feeds UniVLA's DINO latent-action model
(ControllableDINOLatentActionModel) a batch of {videos:(B,2,3,224,224), task_instruction:[str]}
straight from `robocasa_target_human_unified` (built by build_unified_v30.py).

The pair is (frame_t0, frame_{t0+gap}) on the agentview_left camera, matching what
rc365/robocasa_dataset.py feeds the LAM at UniVLA-finetune time (gap == window_size).
"""
import os, re, json, random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import imageio.v2 as imageio


def split_camel(s: str) -> str:
    s = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', s)
    s = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', s)
    return s.lower()


class _VideoCache:
    def __init__(self):
        self._readers = {}

    def frame(self, path, idx):
        rd = self._readers.get(path)
        if rd is None:
            rd = imageio.get_reader(path, format="FFMPEG")
            self._readers[path] = rd
        return rd.get_data(int(idx))

    def close(self):
        for rd in self._readers.values():
            try: rd.close()
            except Exception: pass
        self._readers.clear()


class LamFramePairDataset(Dataset):
    def __init__(self, data_root, task_names=None, gap=12, gap_jitter=0,
                 resolution=224, samples_per_epoch=None, seed=0):
        import pandas as pd
        self.data_root = data_root
        self.gap = gap
        self.gap_jitter = gap_jitter
        self.res = resolution
        self._vc = _VideoCache()
        self._rng = np.random.default_rng(seed)

        meta_ep = pd.read_parquet(os.path.join(data_root, "meta", "episodes.parquet"))
        meta_ep["task"] = meta_ep["tasks"].apply(lambda x: x[0])
        if task_names:
            meta_ep = meta_ep[meta_ep["task"].isin(task_names)]

        # data parquet(s): episode_index -> length via frame_index
        self._ep = []
        need = sorted({(int(r["data/chunk_index"]), int(r["data/file_index"])) for _, r in meta_ep.iterrows()})
        ep_len = {}
        for ci, fi in need:
            df = pd.read_parquet(os.path.join(data_root, "data", f"chunk-{ci:03d}", f"file-{fi:03d}.parquet"),
                                 columns=["episode_index", "frame_index"])
            for ei, g in df.groupby("episode_index"):
                ep_len[int(ei)] = int(g["frame_index"].max()) + 1

        for _, r in meta_ep.iterrows():
            ei = int(r["episode_index"])
            L = ep_len.get(ei)
            if L is None or L < gap + 2:
                continue
            self._ep.append(dict(
                task=r["task"],
                instruction=split_camel(r["task"]),
                length=L,
                left_f0=int(round(float(r["videos/observation.images.robot0_agentview_left/from_timestamp"]) * 20)),
                left_mp4=os.path.join(data_root, "videos", "observation.images.robot0_agentview_left",
                                      f"chunk-{int(r['videos/observation.images.robot0_agentview_left/chunk_index']):03d}",
                                      f"file-{int(r['videos/observation.images.robot0_agentview_left/file_index']):03d}.mp4"),
            ))
        self._spe = samples_per_epoch or (len(self._ep) * 20)
        print(f"[LamFramePairDataset] {len(self._ep)} episodes, {self._spe} samples/epoch, gap={gap}±{gap_jitter}")

    def reseed(self, seed):
        """Call from a DataLoader worker_init_fn so each worker draws a distinct
        stream (the dataset is index-agnostic: __getitem__ samples at random)."""
        self._rng = np.random.default_rng(seed)
        self._vc = _VideoCache()

    def __len__(self):
        return self._spe

    def _frame(self, ep, f_in_ep):
        arr = self._vc.frame(ep["left_mp4"], ep["left_f0"] + f_in_ep)
        img = Image.fromarray(arr).convert("RGB").resize((self.res, self.res), Image.BILINEAR)
        return torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)

    def __getitem__(self, idx):
        ep = self._ep[self._rng.integers(0, len(self._ep))]
        g = self.gap
        if self.gap_jitter:
            g = int(self._rng.integers(max(1, self.gap - self.gap_jitter), self.gap + self.gap_jitter + 1))
        t0 = int(self._rng.integers(0, ep["length"] - g - 1))
        f_cur = self._frame(ep, t0)
        f_fut = self._frame(ep, t0 + g)
        return dict(videos=torch.stack([f_cur, f_fut], 0), task_instruction=ep["instruction"])


def collate_lam(batch):
    return dict(
        videos=torch.stack([b["videos"] for b in batch], 0),
        task_instruction=[b["task_instruction"] for b in batch],
        action=torch.zeros(len(batch), 2, 12),          # unused by ControllableDINOLatentActionModel
        dataset_names=["robocasa365"] * len(batch),
    )
