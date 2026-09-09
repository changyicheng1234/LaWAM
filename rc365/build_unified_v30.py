"""
Rebuild `robocasa_target_human_unified` (LeRobot v3.0) from the raw per-task
RoboCasa365 v2.1 datasets.  [recovery — original converter was lost in the 2026-09-07 wipe]

Source  : /mnt/data/l30083605/Robocasa365-tactile/target/{atomic,composite}/<TASK>/<date>/lerobot
          (LeRobot v2.1, per-episode parquet + per-episode h264 mp4, has tactile.* cols)
Output  : <out>/  with exactly the layout rc365/robocasa_dataset.py expects:
          meta/episodes.parquet meta/tasks.parquet meta/stats.json meta/info.json
          data/chunk-000/file-000.parquet         (all episodes, cols: action, observation.state,
                                                    frame_index, episode_index)
          videos/observation.images.robot0_agentview_left/chunk-000/file-000.mp4   (all eps concat, -c copy)
          videos/observation.images.robot0_eye_in_hand/chunk-000/file-000.mp4

Only the two cameras the loader reads (agentview_left + eye_in_hand) are carried;
tactile / agentview_right / reward / done are dropped.

Usage:
  python build_unified_v30.py --out /root/data/robocasa_target_human_unified \
      --tasks CloseToasterOvenDoor,OpenDrawer,TurnOnMicrowave
"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np
import pandas as pd

SRC_ROOTS = [
    "/mnt/data/l30083605/Robocasa365-tactile/target/atomic",
    "/mnt/data/l30083605/Robocasa365-tactile/target/composite",
]
CAMS = ["robot0_agentview_left", "robot0_eye_in_hand"]
FPS = 20


def find_task_dir(task):
    for r in SRC_ROOTS:
        d = os.path.join(r, task)
        if os.path.isdir(d):
            dates = sorted(os.listdir(d))
            if not dates:
                raise SystemExit(f"{task}: no date dir under {d}")
            lr = os.path.join(d, dates[-1], "lerobot")
            if not os.path.isdir(lr):
                raise SystemExit(f"{task}: missing lerobot/ at {lr}")
            return lr
    raise SystemExit(f"{task}: not found under {SRC_ROOTS}")


def ffprobe_nframes(mp4):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", mp4],
        capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def concat_copy(mp4_list, dst):
    """Lossless concat of same-codec mp4s via the concat demuxer."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in mp4_list:
            f.write(f"file '{p}'\n")
        listfile = f.name
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "concat", "-safe", "0", "-i", listfile,
             "-c", "copy", dst],
            check=True)
    finally:
        os.unlink(listfile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", required=True, help="comma-separated task names")
    ap.add_argument("--limit_per_task", type=int, default=0, help="debug: cap episodes/task")
    cfg = ap.parse_args()
    tasks = [t.strip() for t in cfg.tasks.split(",") if t.strip()]
    os.makedirs(os.path.join(cfg.out, "meta"), exist_ok=True)
    os.makedirs(os.path.join(cfg.out, "data", "chunk-000"), exist_ok=True)
    for cam in CAMS:
        os.makedirs(os.path.join(cfg.out, "videos", f"observation.images.{cam}", "chunk-000"), exist_ok=True)

    task_index = {t: i for i, t in enumerate(tasks)}
    ep_rows = []           # meta/episodes.parquet rows
    data_frames = []       # per-episode DataFrames for the master data parquet
    cam_mp4s = {cam: [] for cam in CAMS}
    g_ep = 0               # global episode index
    g_row = 0              # global row (frame) offset into master parquet
    cam_frame_cursor = {cam: 0 for cam in CAMS}   # cumulative frames written per cam

    for t in tasks:
        lr = find_task_dir(t)
        info = json.load(open(os.path.join(lr, "meta", "info.json")))
        n_ep = info["total_episodes"]
        if cfg.limit_per_task:
            n_ep = min(n_ep, cfg.limit_per_task)
        print(f"[{t}] {lr}  ({n_ep} episodes)")
        for local_ei in range(n_ep):
            pq = os.path.join(lr, "data", "chunk-000", f"episode_{local_ei:06d}.parquet")
            df = pd.read_parquet(pq, columns=["action", "observation.state", "frame_index", "timestamp"])
            L = len(df)
            # per-cam frame counts must match the parquet length
            mp4_paths = {}
            for cam in CAMS:
                m = os.path.join(lr, "videos", "chunk-000", f"observation.images.{cam}",
                                 f"episode_{local_ei:06d}.mp4")
                mp4_paths[cam] = m
            # trust parquet length == frame count (verified on samples); ffprobe only on mismatch guard
            out_df = pd.DataFrame({
                "action": list(np.asarray(np.stack(df["action"].values), dtype=np.float32)),
                "observation.state": list(np.asarray(np.stack(df["observation.state"].values), dtype=np.float32)),
                "frame_index": np.arange(L, dtype=np.int64),
                "episode_index": np.full(L, g_ep, dtype=np.int64),
            })
            data_frames.append(out_df)

            row = {
                "episode_index": g_ep,
                "tasks": [t],
                "length": int(L),
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": g_row,
                "dataset_to_index": g_row + L,
            }
            for cam in CAMS:
                key = "robot0_agentview_left" if cam == "robot0_agentview_left" else "robot0_eye_in_hand"
                col = f"videos/observation.images.{key}"
                row[f"{col}/chunk_index"] = 0
                row[f"{col}/file_index"] = 0
                row[f"{col}/from_timestamp"] = cam_frame_cursor[cam] / FPS
                row[f"{col}/to_timestamp"] = (cam_frame_cursor[cam] + L) / FPS
                cam_mp4s[cam].append(mp4_paths[cam])
                cam_frame_cursor[cam] += L
            ep_rows.append(row)
            g_ep += 1
            g_row += L
            if g_ep % 200 == 0:
                print(f"  ... {g_ep} episodes, {g_row} frames")

    # ---- master data parquet ----
    print("writing data/chunk-000/file-000.parquet ...")
    big = pd.concat(data_frames, ignore_index=True)
    big.to_parquet(os.path.join(cfg.out, "data", "chunk-000", "file-000.parquet"), index=False)
    del data_frames, big

    # ---- videos: concat -c copy per camera ----
    for cam in CAMS:
        dst = os.path.join(cfg.out, "videos", f"observation.images.{cam}", "chunk-000", "file-000.mp4")
        print(f"concat {len(cam_mp4s[cam])} mp4s -> {dst}")
        concat_copy(cam_mp4s[cam], dst)
        got = ffprobe_nframes(dst)
        want = cam_frame_cursor[cam]
        print(f"  {cam}: {got} frames (expected {want}) {'OK' if got == want else 'MISMATCH!!'}")
        if got != want:
            raise SystemExit(f"{cam} concat frame count mismatch: {got} != {want}")

    # ---- meta ----
    ep_df = pd.DataFrame(ep_rows)
    ep_df.to_parquet(os.path.join(cfg.out, "meta", "episodes.parquet"), index=False)

    tasks_df = pd.DataFrame({"task": tasks, "task_index": [task_index[t] for t in tasks]}).set_index("task")
    tasks_df.to_parquet(os.path.join(cfg.out, "meta", "tasks.parquet"))

    # proprio stats over all frames
    allpq = pd.read_parquet(os.path.join(cfg.out, "data", "chunk-000", "file-000.parquet"),
                            columns=["observation.state", "action"])
    state = np.stack(allpq["observation.state"].values).astype(np.float64)   # (N,16)
    act = np.stack(allpq["action"].values).astype(np.float64)               # (N,12)
    stats = {
        "observation.state": {"mean": state.mean(0).tolist(), "std": state.std(0).tolist(),
                              "min": state.min(0).tolist(), "max": state.max(0).tolist()},
        "action": {"mean": act.mean(0).tolist(), "std": act.std(0).tolist(),
                   "min": act.min(0).tolist(), "max": act.max(0).tolist()},
    }
    json.dump(stats, open(os.path.join(cfg.out, "meta", "stats.json"), "w"), indent=1)

    info = {
        "codebase_version": "v3.0",
        "robot_type": "robocasa",
        "total_episodes": int(g_ep),
        "total_frames": int(g_row),
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "fps": FPS,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "splits": {"train": f"0:{g_ep}"},
        "features": {
            "observation.state": {"dtype": "float32", "shape": [16]},
            "action": {"dtype": "float32", "shape": [12]},
            "observation.images.robot0_agentview_left": {"dtype": "video", "shape": [256, 256, 3]},
            "observation.images.robot0_eye_in_hand": {"dtype": "video", "shape": [256, 256, 3]},
        },
        "_source": "rebuilt from /mnt/data/l30083605/Robocasa365-tactile/target by build_unified_v30.py",
        "_tasks": tasks,
    }
    json.dump(info, open(os.path.join(cfg.out, "meta", "info.json"), "w"), indent=1)
    print(f"DONE: {g_ep} episodes / {g_row} frames -> {cfg.out}")


if __name__ == "__main__":
    main()
