"""Stage-2 latent-action-model fine-tune on RoboCasa frames, fed straight from
LeRobot v3.0 (no RLDS / no prismatic / no T5 / no Lightning-CLI).

Reproduces UniVLA `latent_action_model/genie/model.py::DINO_LAM.shared_step`
(mse + q + beta*commit, plus the uncontrol VQ terms of the Controllable model)
as a plain torchrun DDP loop.  Init weights come from the released
`lam-stage-2.ckpt`; only the task-centric codebook / encoder / decoder move
(the model freezes `dino_encoder` and the task-irrelevant `vq` internally).

Matches the earlier `lam_robocasa` run (hparams.yaml): stage-2, bs 32/gpu,
lr 5e-5, gap in [h_lo, h_hi], 224px, num_latents 16.

Launch:
  cd /root/dev/LaWAM/rc365
  torchrun --standalone --nproc-per-node 8 lam_finetune_lerobot.py \
      --run_dir /root/runs/lam_robocasa_v30 --max_steps 8000
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/dev/LaWAM/UniVLA")

from lam_data_lerobot import LamFramePairDataset, collate_lam  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

LAM_KW = dict(in_dim=3, model_dim=768, latent_dim=128, num_latents=16,
              patch_size=14, enc_blocks=12, dec_blocks=12, num_heads=12, dropout=0.0)


def is_dist():
    return dist.is_available() and dist.is_initialized()


def rank0():
    return (not is_dist()) or dist.get_rank() == 0


def all_mean(x: torch.Tensor):
    if not is_dist():
        return x
    x = x.clone()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x / dist.get_world_size()


def build_model(init_ckpt, device):
    from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel
    torch.hub.set_dir(os.path.expanduser("~/.cache/torch/hub"))
    model = ControllableDINOLatentActionModel(**LAM_KW)
    if init_ckpt and os.path.exists(init_ckpt):
        sd = torch.load(init_ckpt, map_location="cpu")
        sd = sd["state_dict"] if "state_dict" in sd else sd
        sd = {k.replace("lam.", "", 1): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if rank0():
            print(f"[init] loaded {init_ckpt}  missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print("       missing:", missing[:12])
            if unexpected:
                print("       unexpected:", unexpected[:12])
    else:
        raise FileNotFoundError(f"--init_ckpt not found: {init_ckpt}")
    return model.to(device)


def lam_losses(out, vq_beta):
    """Mirror DINO_LAM.shared_step for the Controllable model."""
    tgt = out["target"]
    mse = ((tgt - out["recon"]) ** 2).mean()
    q = ((out["emb"].detach() - out["z"]) ** 2).mean()
    commit = ((out["emb"] - out["z"].detach()) ** 2).mean()
    loss = mse + q + vq_beta * commit

    q_u = ((out["emb_uncontrol"].detach() - out["z_uncontrol"]) ** 2).mean()
    commit_u = ((out["emb_uncontrol"] - out["z_uncontrol"].detach()) ** 2).mean()
    loss = loss + q_u + vq_beta * commit_u

    with torch.no_grad():
        def usage(idx, n):
            u, c = torch.unique(idx, return_counts=True)
            t = torch.zeros(n, device=idx.device, dtype=torch.long)
            t[u] = c
            return (t != 0).float().mean()
        cu = usage(out["indices"], LAM_KW["num_latents"])
        cu_u = usage(out["indices_uncontrol"], 16)
    stats = dict(mse=mse.detach(), q=q.detach(), commit=commit.detach(),
                 q_u=q_u.detach(), commit_u=commit_u.detach(),
                 code_usage=cu, code_usage_uncontrol=cu_u)
    return loss, stats


def restart_dead_codes(raw_model):
    """`on_train_epoch_end` equivalent.  Upstream only restarts `vq`; we also
    restart the trainable `vq_action` (that is the codebook being learned in
    stage-2).  Rank-0 decides, then codebooks are broadcast so DDP stays in
    sync."""
    for vq in (raw_model.vq, raw_model.vq_action):
        if rank0():
            vq.random_restart()
        vq.reset_usage()
        if is_dist():
            dist.broadcast(vq.codebook.weight.data, src=0)


def save_ckpt(raw_model, cfg, step, mirror=True):
    os.makedirs(cfg.run_dir, exist_ok=True)
    sd = {f"lam.{k}": v for k, v in raw_model.state_dict().items()}
    blob = dict(state_dict=sd, step=step, lam_kwargs=LAM_KW, cfg=vars(cfg))
    path = os.path.join(cfg.run_dir, f"lam_stage2_{step}.ckpt")
    torch.save(blob, path)
    last = os.path.join(cfg.run_dir, "last.ckpt")
    shutil.copyfile(path, last)
    print(f"  saved {path}")
    if mirror and cfg.mirror_dir:
        try:
            os.makedirs(cfg.mirror_dir, exist_ok=True)
            dst = os.path.join(cfg.mirror_dir, os.path.basename(path))
            shutil.copyfile(path, dst)                       # single sequential write, OSS-safe
            shutil.copyfile(path, os.path.join(cfg.mirror_dir, "last.ckpt"))
            print(f"  mirrored -> {dst}")
        except Exception as e:  # never let backup IO kill training
            print(f"  [warn] mirror failed: {e}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/root/data/robocasa_target_human_unified")
    ap.add_argument("--tasks", default="CloseToasterOvenDoor,OpenDrawer,TurnOnMicrowave")
    ap.add_argument("--init_ckpt", default="/root/weights/univla-latent-action-model/lam-stage-2.ckpt")
    ap.add_argument("--run_dir", default="/root/runs/lam_robocasa_v30")
    ap.add_argument("--mirror_dir", default="/mnt/data/changyicheng/rc365/rc365_runs/lam_robocasa_v30")
    ap.add_argument("--batch_size", type=int, default=32, help="per GPU")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--vq_beta", type=float, default=0.25)
    ap.add_argument("--h_lo", type=int, default=8)
    ap.add_argument("--h_hi", type=int, default=16)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--max_steps", type=int, default=8000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--clip_grad", type=float, default=0.1)
    ap.add_argument("--restart_every", type=int, default=1000, help="dead-code restart interval (steps)")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--samples_per_epoch", type=int, default=None)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="resume from <run_dir>/last.ckpt")
    ap.add_argument("--smoke", type=int, default=0, help="if >0: run this many steps, 1 GPU, no mirror")
    cfg = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    rank = dist.get_rank() if is_dist() else 0
    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if cfg.smoke:
        cfg.max_steps = cfg.smoke
        cfg.save_every = max(cfg.save_every, cfg.smoke)
        cfg.mirror_dir = ""
        cfg.num_workers = min(cfg.num_workers, 4)

    gap = (cfg.h_lo + cfg.h_hi) // 2
    jitter = (cfg.h_hi - cfg.h_lo) // 2
    spe = cfg.samples_per_epoch or (cfg.batch_size * max(world, 1) * cfg.log_every * 40)
    ds = LamFramePairDataset(cfg.data_root, task_names=cfg.tasks.split(","),
                             gap=gap, gap_jitter=jitter, resolution=cfg.resolution,
                             samples_per_epoch=spe, seed=cfg.seed + 1000 * rank)

    def worker_init_fn(wid):
        info = torch.utils.data.get_worker_info()
        info.dataset.reseed(cfg.seed + 100003 * (rank + 1) + wid)

    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True,
                    num_workers=cfg.num_workers, collate_fn=collate_lam,
                    worker_init_fn=worker_init_fn,
                    persistent_workers=cfg.num_workers > 0, pin_memory=True)

    model = build_model(cfg.init_ckpt, device)
    raw = model
    if is_dist():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    params = [p for p in raw.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    start_step = 0
    if cfg.resume and os.path.exists(os.path.join(cfg.run_dir, "last.ckpt")):
        blob = torch.load(os.path.join(cfg.run_dir, "last.ckpt"), map_location="cpu")
        raw.load_state_dict({k.replace("lam.", "", 1): v for k, v in blob["state_dict"].items()})
        start_step = int(blob.get("step", 0))
        if rank0():
            print(f"[resume] from step {start_step}")

    def lr_at(s):
        if s < cfg.warmup:
            return cfg.lr * (s + 1) / cfg.warmup
        return cfg.lr

    if rank0():
        os.makedirs(cfg.run_dir, exist_ok=True)
        json.dump(vars(cfg), open(os.path.join(cfg.run_dir, "effective_config.json"), "w"),
                  indent=2, sort_keys=True)
        logf = open(os.path.join(cfg.run_dir, "train.log"), "a")

        def log(m):
            print(m, flush=True)
            logf.write(m + "\n")
            logf.flush()
        log(f"==== lam_finetune_lerobot @ {time.strftime('%F %T')}  world={world} "
            f"global_bs={cfg.batch_size * max(world,1)} ====")
        log(json.dumps(vars(cfg), sort_keys=True))
        log(f"trainable params: {sum(p.numel() for p in params)/1e6:.2f}M")
    else:
        def log(m):
            pass

    model.train()
    raw.dino_encoder.eval()
    step = start_step
    t_log = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    done = False
    while not done:
        for batch in dl:
            videos = batch["videos"].to(device, non_blocking=True)
            mb = dict(videos=videos, task_instruction=batch["task_instruction"])
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(mb)
                loss, st = lam_losses(out, cfg.vq_beta)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad)
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step)
            scaler.step(opt)
            scaler.update()

            if step % cfg.log_every == 0:
                m = {k: all_mean(v.float()).item() for k, v in st.items()}
                lo = all_mean(loss.detach().float()).item()
                dt = (time.time() - t_log) / max(1, cfg.log_every)
                log(f"step {step:6d} | loss {lo:.4f} mse {m['mse']:.4f} "
                    f"q {m['q']:.4f} commit {m['commit']:.4f} q_u {m['q_u']:.4f} "
                    f"commit_u {m['commit_u']:.4f} | code_use {m['code_usage']:.3f} "
                    f"unctrl {m['code_usage_uncontrol']:.3f} | gnorm {float(gnorm):.2f} "
                    f"lr {lr_at(step):.2e} | {dt*1000:.0f} ms/step | "
                    f"peakGPU {torch.cuda.max_memory_allocated(device)/1e9:.1f}GB")
                t_log = time.time()

            step += 1
            if cfg.restart_every and step % cfg.restart_every == 0:
                restart_dead_codes(raw)
            if step % cfg.save_every == 0 or step >= cfg.max_steps:
                if is_dist():
                    dist.barrier()
                if rank0():
                    save_ckpt(raw, cfg, step)
                if is_dist():
                    dist.barrier()
            if step >= cfg.max_steps:
                done = True
                break

    log("DONE")
    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
