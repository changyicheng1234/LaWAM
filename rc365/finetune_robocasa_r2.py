"""RoboCasa365 round-2 fine-tune: round-1 `finetune_robocasa.py` + LoRA on the
VLA + `vla_ce` / `decoder_loss_weight` / `direct` + more steps, fed the newly
RoboCasa-retrained `lam-stage-2.ckpt`.

Reconstructed from the round-2 run configs / logs (the original script was never
committed).  Per-task by default (n_demos 400, one LoRA adapter per task).

Reference profiles (see `_round2_ref/`):
  r2_pertask : --use_lora --lora_rank 32               --decoder_loss_weight 5 --max_steps 6000
  r2_crank   : --use_lora --lora_rank 64 --lora_alpha 128 --vla_ce_weight 10  --max_steps 8000
  r2_direct  : --use_lora --lora_rank 64 --lora_alpha 128 --direct --decoder_loss_weight 1 --max_steps 25000

Single GPU.  Example:
  cd /root/dev/LaWAM/rc365
  python finetune_robocasa_r2.py --vision_variant v1 --tasks CloseToasterOvenDoor \
      --use_lora --lora_rank 64 --lora_alpha 128 --vla_ce_weight 10 \
      --n_demos 400 --max_steps 8000 --run_dir /root/runs/r2_crank/CloseToasterOvenDoor \
      --vla_path /root/weights/univla-7b \
      --lam_path /root/weights/univla-latent-action-model/lam-stage-2.ckpt
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robocasa_dataset import RoboCasaLeRobotDataset, collate, load_proprio_stats  # noqa: E402
from model_robocasa_r2 import WrappedModelRoboCasaR2  # noqa: E402
from vla_prep import build_vla_inputs  # noqa: E402


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vla_path", default="/root/weights/univla-7b")
    ap.add_argument("--lam_path", default="/root/weights/univla-latent-action-model/lam-stage-2.ckpt")
    ap.add_argument("--data_root", default="/root/data/robocasa_target_human_unified")
    ap.add_argument("--tasks", default="CloseToasterOvenDoor")
    ap.add_argument("--n_demos", type=int, default=400)
    ap.add_argument("--vision_variant", choices=["v1", "v2"], required=True)
    ap.add_argument("--v2_each_size", type=int, default=224)
    ap.add_argument("--window_size", type=int, default=12, help="action chunk length")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3.5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--max_steps", type=int, default=8000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--ce_weight", type=float, default=1.0, help="weight of the control_mode CE inside decoder_loss")
    ap.add_argument("--mode_class_weight", default="", help="逗号分隔 2 个数; 空=不加权")
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--cache_frames", action="store_true", help="RoboCasaLeRobotDataset.cache_frames_in_ram")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--run_dir", default="/root/runs/r2_pertask/CloseToasterOvenDoor")
    ap.add_argument("--seed", type=int, default=7)
    # ---- round-2 knobs ----
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=-1, help="-1 => min(lora_rank, 16) (UniVLA default)")
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--decoder_loss_weight", type=float, default=1.0)
    ap.add_argument("--vla_ce_weight", type=float, default=1.0, help="weight on the VLA <ACT_*> CE; ignored if --direct")
    ap.add_argument("--direct", action="store_true", help="no latent-action tokens / no vla_ce; VLA is a LoRA backbone")
    cfg = ap.parse_args()
    if cfg.lora_alpha < 0:
        cfg.lora_alpha = min(cfg.lora_rank, 16)

    set_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda:0"
    run_dir = os.path.join(cfg.run_dir, cfg.vision_variant)
    adapter_dir = os.path.join(run_dir, "lora_adapter")
    os.makedirs(run_dir, exist_ok=True)
    json.dump(vars(cfg), open(os.path.join(run_dir, "effective_config.json"), "w"),
              indent=2, sort_keys=True)
    logf = open(os.path.join(run_dir, "train.log"), "a")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()
    log(f"==== finetune_robocasa_r2 {cfg.vision_variant} @ {time.strftime('%F %T')} ====")
    log(json.dumps(vars(cfg), sort_keys=True))

    from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, AutoModelForVision2Seq
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    log("loading VLA ...")
    proc = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="flash_attention_2").to(dev)

    freeze_vla = not (cfg.use_lora or cfg.direct)
    if cfg.use_lora:
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules="all-linear", init_lora_weights="gaussian", bias="none",
        )
        vla = get_peft_model(vla, lora_cfg)
        log(f"LoRA r={cfg.lora_rank} alpha={cfg.lora_alpha}")
        if hasattr(vla, "print_trainable_parameters"):
            vla.print_trainable_parameters()

    from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel
    lam = ControllableDINOLatentActionModel(in_dim=3, model_dim=768, latent_dim=128, num_latents=16,
        patch_size=14, enc_blocks=12, dec_blocks=12, num_heads=12, dropout=0.)
    ck = torch.load(cfg.lam_path, map_location="cpu")["state_dict"]
    lam.load_state_dict({k.replace("lam.", "", 1): v for k, v in ck.items()}, strict=True)
    lam = lam.to(dev).eval()

    mcw = [float(x) for x in cfg.mode_class_weight.split(",")] if cfg.mode_class_weight else None
    wm = WrappedModelRoboCasaR2(
        vla=vla, freeze_vla=freeze_vla, window_size=cfg.window_size, ce_weight=cfg.ce_weight,
        mode_class_weight=mcw, decoder_loss_weight=cfg.decoder_loss_weight,
        vla_ce_weight=cfg.vla_ce_weight, direct=cfg.direct).to(dev)
    wm.train()
    if freeze_vla:
        wm.vla.eval()
    n_train = sum(p.numel() for p in wm.parameters() if p.requires_grad)
    log(f"trainable params: {n_train:,} ({n_train/1e6:.3f}M)")

    pm, ps = load_proprio_stats(cfg.data_root)
    ds = RoboCasaLeRobotDataset(cfg.data_root, cfg.tasks.split(","), cfg.n_demos, cfg.window_size,
                                cfg.vision_variant, proc.image_processor.apply_transform, pm, ps,
                                v2_each_size=cfg.v2_each_size, seed=cfg.seed,
                                cache_frames_in_ram=cfg.cache_frames)
    from torch.utils.data import DataLoader
    g = torch.Generator(); g.manual_seed(cfg.seed)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    collate_fn=collate, drop_last=True, generator=g, persistent_workers=cfg.num_workers > 0)

    params = [p for p in wm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def lr_at(step):
        if step < cfg.warmup:
            return cfg.lr * (step + 1) / cfg.warmup
        return cfg.lr

    def save(step):
        sp = os.path.join(run_dir, f"action_decoder_{step}.pt")
        torch.save(wm.action_decoder.state_dict(), sp)
        torch.save(wm.action_decoder.state_dict(), os.path.join(run_dir, "action_decoder_last.pt"))
        np.savez(os.path.join(run_dir, "proprio_stats.npz"), mean=pm, std=ps)
        if cfg.use_lora:
            wm.vla.save_pretrained(adapter_dir)
        log(f"  saved {sp}" + (" + lora_adapter" if cfg.use_lora else ""))

    init_l1 = None
    step = 0; opt.zero_grad(); t_log = time.time()
    torch.cuda.reset_peak_memory_stats()
    while step < cfg.max_steps:
        for batch in dl:
            batch = build_vla_inputs(batch, lam, proc, dev, direct=cfg.direct)
            out = wm(batch)
            (out["loss"] / cfg.grad_accum).backward()
            if (step + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad)
                for pg in opt.param_groups:
                    pg["lr"] = lr_at(step)
                opt.step(); opt.zero_grad()
            if init_l1 is None:
                init_l1 = out["l1"].item()
            if step % cfg.log_every == 0:
                dt = (time.time() - t_log) / max(1, cfg.log_every)
                vce = out["vla_ce"]
                vce_s = f"vla_ce {vce.item():.4f} " if vce is not None else ""
                acc_s = ""
                if not cfg.direct and cfg.vla_ce_weight != 0:
                    acc_s = f"act_acc {out['act_acc'].item():.3f} act_distinct {int(out['act_distinct'].item())} "
                log(f"step {step:5d} | loss {out['loss'].item():.4f} l1 {out['l1'].item():.4f} "
                    f"(init {init_l1:.4f}) {vce_s}{acc_s}"
                    f"l1_1step {out['l1_1step'].item():.4f} mode_acc {out['mode_acc'].item():.3f} "
                    f"grip_acc {out['grip_acc'].item():.3f} | {dt*1000:.0f} ms/step | "
                    f"peakGPU {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
                t_log = time.time()
            step += 1
            if step % cfg.save_every == 0 or step >= cfg.max_steps:
                save(step)
            if step >= cfg.max_steps:
                break
    log("DONE")


if __name__ == "__main__":
    main()
