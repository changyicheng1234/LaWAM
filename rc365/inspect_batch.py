"""S1 验收: dump 一个 batch, 检查字段, 存 pixel_values PNG (反归一化)."""
import os, sys, numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from robocasa_dataset import RoboCasaLeRobotDataset, collate, load_proprio_stats

DATA = "/opt/rc365_data/robocasa_target_human_unified"
TASKS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["NavigateKitchen", "CloseToasterOvenDoor"]
NDEMO = int(sys.argv[2]) if len(sys.argv) > 2 else 8
WS = 12

from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, AutoModelForVision2Seq
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
proc = AutoProcessor.from_pretrained("/opt/weights/univla-7b", trust_remote_code=True)
tfm = proc.image_processor.apply_transform

pm, ps = load_proprio_stats(DATA)
print("proprio mean/std (16):")
print("  mean", np.array2string(pm, precision=3))
print("  std ", np.array2string(ps, precision=3))

# 反归一化参数 (来自 univla-7b/preprocessor_config.json, 两路相同)
MEAN = torch.tensor([0.484375, 0.455078125, 0.40625]).view(3, 1, 1)
STD = torch.tensor([0.228515625, 0.2236328125, 0.224609375]).view(3, 1, 1)

for variant in ["v1", "v2"]:
    print(f"\n{'='*60}\n=== variant {variant} ===")
    ds = RoboCasaLeRobotDataset(DATA, TASKS, NDEMO, WS, variant, tfm, pm, ps,
                                v2_each_size=224, seed=7, cache_frames_in_ram=False)
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate)
    b = next(iter(dl))
    print("--- batch 字段 ---")
    for k, v in b.items():
        if torch.is_tensor(v):
            extra = ""
            if v.dtype.is_floating_point:
                extra = f" range[{v.min():.3f},{v.max():.3f}]"
            print(f"  {k:26s} {tuple(v.shape)} {str(v.dtype):15s}{extra}")
        else:
            print(f"  {k:26s} {v}")
    a = b["actions"]
    print("  actions per-dim min/max:")
    for i in range(12):
        print(f"    dim{i:2d}: [{a[...,i].min():+.3f}, {a[...,i].max():+.3f}]  unique~{len(torch.unique(a[...,i]))}")
    print("  control_mode_tgt unique:", torch.unique(b["control_mode_tgt"]).tolist(),
          " | +1(class1) 占比:", f"{b['control_mode_tgt'].float().mean():.3f}")
    print("  proprio[0]:", np.array2string(b["proprio"][0].numpy(), precision=3))

    # 存 PNG: pixel_values 两路各反归一化
    pv = b["pixel_values"][0]  # (6,224,224)
    for j, tag in enumerate(["path0_dino", "path1_siglip"]):
        img = pv[j*3:(j+1)*3] * STD + MEAN
        img = (img.clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(img).save(f"/opt/rc365_data/_batch_{variant}_{tag}.png")
    # LAM 输入帧
    ip = (b["initial_pixel_values"][0].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    tp = (b["target_pixel_values"][0].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(np.concatenate([ip, tp], axis=1)).save(f"/opt/rc365_data/_batch_{variant}_lam_pair.png")
    print(f"  saved /opt/rc365_data/_batch_{variant}_*.png")

    # 100 batch 压测
    import time
    t = time.time(); n = 0
    for bb in dl:
        n += 1
        if n >= 25: break
    dt = time.time() - t
    print(f"  25 batch(bs=4, nw=0): {dt:.1f}s -> {dt/25*1000:.0f} ms/batch")
    ds._vc.close()
print("\nDONE")
