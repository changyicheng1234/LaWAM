"""诊断: 训练分布内, decoder 预测 vs GT 动作. 判断 eval SR=0 是"策略弱"还是"管线有bug"."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from robocasa_dataset import RoboCasaLeRobotDataset, load_proprio_stats, split_camel
from model_robocasa import ActionDecoderRoboCasa, CONT_IDX, MODE_IDX

DATA = "/opt/rc365_data/robocasa_target_human_unified"
VAR = sys.argv[1] if len(sys.argv) > 1 else "v1"
RUN = f"/opt/rc365_runs/round1/{VAR}"
N = 32

from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, AutoModelForVision2Seq
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
proc = AutoProcessor.from_pretrained("/opt/weights/univla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained("/opt/weights/univla-7b", torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation="flash_attention_2").to("cuda:0").eval()
dec = ActionDecoderRoboCasa(window_size=12)
dec.load_state_dict(torch.load(f"{RUN}/action_decoder_last.pt", map_location="cpu"))
dec = dec.to("cuda:0").eval()
st = np.load(f"{RUN}/proprio_stats.npz"); pm = st["mean"].astype(np.float32); ps = st["std"].astype(np.float32).copy(); ps[ps < 1e-6] = 1.0

pm2, ps2 = load_proprio_stats(DATA)
ds = RoboCasaLeRobotDataset(DATA, ["NavigateKitchen", "CloseToasterOvenDoor"], 40, 12, VAR,
                            proc.image_processor.apply_transform, pm2, ps2, seed=123)

errs = []; gts = []; prs = []
mode_gt = []; mode_pr = []
rng = np.random.default_rng(0)
for k in range(N):
    ep = ds.episodes[rng.integers(len(ds.episodes))]
    act, state = ds._ep_data[ep["episode_index"]]
    T = min(len(act), ep["length"]); t = int(rng.integers(0, T - 13))
    pil = ds.compose_image(ep, t)
    instr = ep["instruction"]
    prompt = f"In: What action should the robot take to {instr}?\nOut:"
    inputs = proc(prompt, pil).to("cuda:0", dtype=torch.bfloat16)
    with torch.no_grad():
        lat, vemb, gids = vla.predict_latent_action(**inputs, do_sample=False)
        pr = ((state[t] - pm) / ps).astype(np.float32)
        cont, ml = dec(lat.float(), vemb.float(), torch.from_numpy(pr)[None].to("cuda:0"))
    cont0 = cont[0, 0].float().cpu().numpy()
    a_pred = np.zeros(12, np.float32); a_pred[CONT_IDX] = np.clip(cont0, -1, 1)
    a_pred[MODE_IDX] = -1.0 if ml[0, 0].argmax().item() == 0 else 1.0
    a_gt = act[t]
    errs.append(np.abs(a_pred - a_gt)); gts.append(a_gt); prs.append(a_pred)
    mode_gt.append(int(a_gt[4] > 0)); mode_pr.append(int(a_pred[4] > 0))

errs = np.stack(errs); gts = np.stack(gts); prs = np.stack(prs)
print(f"\n=== {VAR}: decoder 预测 vs GT (N={N}, 训练分布内, chunk[0]) ===")
print("dim | mean|err| |  GT mean±std       | PRED mean±std      | corr")
for i in range(12):
    c = np.corrcoef(gts[:, i], prs[:, i])[0, 1] if gts[:, i].std() > 1e-6 else float("nan")
    print(f"{i:3d} | {errs[:,i].mean():8.3f} | {gts[:,i].mean():+.3f}±{gts[:,i].std():.3f} | "
          f"{prs[:,i].mean():+.3f}±{prs[:,i].std():.3f} | {c:+.2f}")
print(f"\n全维 mean|err| = {errs.mean():.3f}  (训练 l1_last ≈ 0.09)")
print(f"control_mode 准确率: {np.mean(np.array(mode_gt)==np.array(mode_pr)):.2f}  (gt +1 占比 {np.mean(mode_gt):.2f})")
print("\n判读: 若 mean|err|≈训练l1 且多数维 corr>0.3 -> 管线正确, SR=0 只是策略弱(40 demo/1500 step);")
print("      若 mean|err|≈1 或 corr 普遍≈0/负 -> eval 侧组装或坐标/尺度有 bug.")
