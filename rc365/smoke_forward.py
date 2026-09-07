"""S2 验收: 一个真 batch 前向+反向, 断言 backbone 无梯度 / decoder 有梯度 / loss 有限."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from robocasa_dataset import RoboCasaLeRobotDataset, collate, load_proprio_stats
from model_robocasa import WrappedModelRoboCasa
from vla_prep import build_vla_inputs

DATA = "/opt/rc365_data/robocasa_target_human_unified"
TASKS = ["NavigateKitchen", "CloseToasterOvenDoor"]
WS = 12
DEV = "cuda:0"
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "v1"

from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, AutoModelForVision2Seq
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

VLA_PATH = "/opt/weights/univla-7b"
LAM_PATH = "/opt/weights/univla-latent-action-model/lam-stage-2.ckpt"

print("[1] load processor + VLA (frozen) + LAM")
proc = AutoProcessor.from_pretrained(VLA_PATH, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    VLA_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    trust_remote_code=True, attn_implementation="flash_attention_2").to(DEV)

from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel
lam = ControllableDINOLatentActionModel(in_dim=3, model_dim=768, latent_dim=128, num_latents=16,
    patch_size=14, enc_blocks=12, dec_blocks=12, num_heads=12, dropout=0.)
ck = torch.load(LAM_PATH, map_location="cpu")["state_dict"]
lam.load_state_dict({k.replace("lam.", ""): v for k, v in ck.items()}, strict=True)
lam = lam.to(DEV).eval()

print("[2] wrapped model (freeze_vla=True)")
wm = WrappedModelRoboCasa(vla=vla, freeze_vla=True, window_size=WS, ce_weight=1.0).to(DEV)
n_train = sum(p.numel() for p in wm.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in wm.parameters())
print(f"    trainable params: {n_train:,} ({n_train/1e6:.3f}M) / total {n_total/1e9:.3f}B")
dec_params = sum(p.numel() for p in wm.action_decoder.parameters())
print(f"    action_decoder params: {dec_params:,} ({dec_params/1e6:.3f}M)")

print("[3] one real batch")
pm, ps = load_proprio_stats(DATA)
ds = RoboCasaLeRobotDataset(DATA, TASKS, 8, WS, VARIANT, proc.image_processor.apply_transform, pm, ps, seed=7)
from torch.utils.data import DataLoader
dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate)
batch = next(iter(dl))
batch = build_vla_inputs(batch, lam, proc, DEV)
print("    input_ids", tuple(batch["input_ids"].shape), "| labels>32000 per-sample:",
      [(batch["labels"][i] > 32000).sum().item() for i in range(batch["labels"].shape[0])])

print("[4] forward + backward")
torch.cuda.reset_peak_memory_stats()
opt = torch.optim.AdamW([p for p in wm.parameters() if p.requires_grad], lr=3.5e-4)
wm.train()
if wm.freeze_vla:
    wm.vla.eval()
t = time.time()
out = wm(batch)
out["loss"].backward()
dt = time.time() - t
print(f"    loss={out['loss'].item():.4f}  l1={out['l1'].item():.4f}  ce={out['ce'].item():.4f}  "
      f"ce_acc={out['ce_acc'].item():.3f}  pos_rate={out['pos_rate'].item():.3f}  pred_pos={out['pred_pos_rate'].item():.3f}")
print(f"    fwd+bwd {dt:.2f}s | peak GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

print("[5] 断言")
bb_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in wm.vla.parameters())
dec_grads = [(n, p.grad) for n, p in wm.action_decoder.named_parameters()]
dec_all_have = all(g is not None for _, g in dec_grads)
dec_any_nonzero = any(g is not None and g.abs().sum() > 0 for _, g in dec_grads)
loss_finite = torch.isfinite(out["loss"]).item()
print(f"    backbone 有非零梯度? {bb_has_grad}   (期望 False)")
print(f"    decoder 全部有梯度? {dec_all_have}   (期望 True)")
print(f"    decoder 有非零梯度? {dec_any_nonzero}   (期望 True)")
print(f"    loss 有限? {loss_finite}   (期望 True)")
ok = (not bb_has_grad) and dec_all_have and dec_any_nonzero and loss_finite
print(f"\n{'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
