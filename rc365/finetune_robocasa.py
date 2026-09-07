"""RoboCasa365 decoder-only 微调 (freeze_vla, no LoRA).  [agent-B, PLAN S3]
单卡. round-1 用. V1/V2 只差 --vision_variant.
"""
import os, sys, json, time, argparse, random
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from robocasa_dataset import RoboCasaLeRobotDataset, collate, load_proprio_stats
from model_robocasa import WrappedModelRoboCasa
from vla_prep import build_vla_inputs


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vla_path", default="/opt/weights/univla-7b")
    ap.add_argument("--lam_path", default="/opt/weights/univla-latent-action-model/lam-stage-2.ckpt")
    ap.add_argument("--data_root", default="/opt/rc365_data/robocasa_target_human_unified")
    ap.add_argument("--tasks", default="NavigateKitchen,CloseToasterOvenDoor")
    ap.add_argument("--n_demos", type=int, default=40)
    ap.add_argument("--vision_variant", choices=["v1", "v2"], required=True)
    ap.add_argument("--v2_each_size", type=int, default=224)
    ap.add_argument("--window_size", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3.5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--ce_weight", type=float, default=1.0)
    ap.add_argument("--mode_class_weight", default="", help="逗号分隔 2 个数, 如 '1.0,1.5'; 空=不加权")
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--run_dir", default="/opt/rc365_runs/round1")
    ap.add_argument("--seed", type=int, default=7)
    cfg = ap.parse_args()

    set_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda:0"
    run_dir = os.path.join(cfg.run_dir, cfg.vision_variant)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(vars(cfg), open(os.path.join(run_dir, "effective_config.json"), "w"),
              indent=2, sort_keys=True)
    logf = open(os.path.join(run_dir, "train.log"), "a")
    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()
    log(f"==== finetune_robocasa {cfg.vision_variant} @ {time.strftime('%F %T')} ====")
    log(json.dumps(vars(cfg), sort_keys=True))

    from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, AutoModelForVision2Seq
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    log("loading VLA (frozen) ...")
    proc = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="flash_attention_2").to(dev)

    from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel
    lam = ControllableDINOLatentActionModel(in_dim=3, model_dim=768, latent_dim=128, num_latents=16,
        patch_size=14, enc_blocks=12, dec_blocks=12, num_heads=12, dropout=0.)
    ck = torch.load(cfg.lam_path, map_location="cpu")["state_dict"]
    lam.load_state_dict({k.replace("lam.", ""): v for k, v in ck.items()}, strict=True)
    lam = lam.to(dev).eval()

    mcw = [float(x) for x in cfg.mode_class_weight.split(",")] if cfg.mode_class_weight else None
    wm = WrappedModelRoboCasa(vla=vla, freeze_vla=True, window_size=cfg.window_size,
                              ce_weight=cfg.ce_weight, mode_class_weight=mcw).to(dev)
    wm.train(); wm.vla.eval()
    n_train = sum(p.numel() for p in wm.parameters() if p.requires_grad)
    log(f"trainable params: {n_train:,} ({n_train/1e6:.3f}M)")

    pm, ps = load_proprio_stats(cfg.data_root)
    ds = RoboCasaLeRobotDataset(cfg.data_root, cfg.tasks.split(","), cfg.n_demos, cfg.window_size,
                                cfg.vision_variant, proc.image_processor.apply_transform, pm, ps,
                                v2_each_size=cfg.v2_each_size, seed=cfg.seed,
                                cache_frames_in_ram=False)
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
    init_l1 = init_ce = None
    step = 0; opt.zero_grad(); t_log = time.time()
    torch.cuda.reset_peak_memory_stats()
    while step < cfg.max_steps:
        for batch in dl:
            batch = build_vla_inputs(batch, lam, proc, dev)
            out = wm(batch)
            (out["loss"] / cfg.grad_accum).backward()
            if (step + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad)
                for pg in opt.param_groups:
                    pg["lr"] = lr_at(step)
                opt.step(); opt.zero_grad()
            if init_l1 is None:
                init_l1, init_ce = out["l1"].item(), out["ce"].item()
            if step % cfg.log_every == 0:
                dt = (time.time() - t_log) / max(1, cfg.log_every)
                log(f"step {step:5d} | loss {out['loss'].item():.4f} l1 {out['l1'].item():.4f} "
                    f"(init {init_l1:.4f}) ce {out['ce'].item():.4f} (init {init_ce:.4f}) "
                    f"ce_acc {out['ce_acc'].item():.3f} pos {out['pos_rate'].item():.2f}/pred {out['pred_pos_rate'].item():.2f} "
                    f"l1_1step {out['l1_1step'].item():.4f} | {dt*1000:.0f} ms/step | "
                    f"peakGPU {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
                t_log = time.time()
            step += 1
            if step % cfg.save_every == 0 or step >= cfg.max_steps:
                sp = os.path.join(run_dir, f"action_decoder_{step}.pt")
                torch.save(wm.action_decoder.state_dict(), sp)
                torch.save(wm.action_decoder.state_dict(), os.path.join(run_dir, "action_decoder_last.pt"))
                np.savez(os.path.join(run_dir, "proprio_stats.npz"), mean=pm, std=ps)
                log(f"  saved {sp}")
            if step >= cfg.max_steps:
                break
    log("DONE")


if __name__ == "__main__":
    main()
