"""UniVLA policy server (univla env, py3.10).  [agent-B, PLAN S4]
zmq REP. rc365 的 eval_robocasa.py 连上来逐 step 要 action.
round-1: 每 step 预测一个 chunk, 执行 chunk[0], 不做 temporal ensemble (先跑通).
"""
import os, sys, argparse, numpy as np, torch, zmq, msgpack
sys.path.insert(0, os.path.dirname(__file__))
from model_robocasa import ActionDecoderRoboCasa, CONT_IDX, MODE_IDX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vla_path", default="/opt/weights/univla-7b")
    ap.add_argument("--decoder_path", required=True)          # action_decoder_last.pt
    ap.add_argument("--proprio_stats", required=True)         # proprio_stats.npz
    ap.add_argument("--vision_variant", choices=["v1", "v2"], required=True)
    ap.add_argument("--v2_each_size", type=int, default=224)
    ap.add_argument("--window_size", type=int, default=12)
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--device", default="cuda:0")
    cfg = ap.parse_args()
    dev = cfg.device

    from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, AutoModelForVision2Seq
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    from PIL import Image

    print("loading VLA ...", flush=True)
    proc = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="flash_attention_2").to(dev).eval()
    dec = ActionDecoderRoboCasa(window_size=cfg.window_size)
    dec.load_state_dict(torch.load(cfg.decoder_path, map_location="cpu"))
    dec = dec.to(dev).eval()
    st = np.load(cfg.proprio_stats)
    pm = st["mean"].astype(np.float32); ps = st["std"].astype(np.float32).copy(); ps[ps < 1e-6] = 1.0
    tfm = proc.image_processor.apply_transform
    np_patches = vla.vision_backbone.featurizer.patch_embed.num_patches

    def compose(img_left, img_wrist):
        L = Image.fromarray(img_left).convert("RGB")
        if cfg.vision_variant == "v1":
            return L
        W = Image.fromarray(img_wrist).convert("RGB")
        s = cfg.v2_each_size
        cat = Image.new("RGB", (2 * s, s))
        cat.paste(L.resize((s, s), Image.BILINEAR), (0, 0))
        cat.paste(W.resize((s, s), Image.BILINEAR), (s, 0))
        return cat.resize((224, 224), Image.BILINEAR)

    state = {"instr": "", "hist": [""]}

    @torch.no_grad()
    def act(img_left, img_wrist, proprio_raw):
        pil = compose(img_left, img_wrist)
        prompt = f"In: What action should the robot take to {state['instr']}?"
        if len(state["hist"][-1]) > 0:
            prompt = f"In: What action should the robot take to {state['instr']}? History action {state['hist'][-1]}"
        prompt += "\nOut:"
        inputs = proc(prompt, pil).to(dev, dtype=torch.bfloat16)
        lat, vemb, gids = vla.predict_latent_action(**inputs, do_sample=False)  # use_cache=False patched
        det = [f"<ACT_{i}>" for i in range(32)]
        state["hist"].append("".join(det[g.item() - 32001] for g in gids[0]))
        cont, mode_logits = dec(lat.float(), vemb.float(), torch.from_numpy(
            ((proprio_raw - pm) / ps).astype(np.float32))[None].to(dev))
        cont = cont[0, 0].float().cpu().numpy()          # (11,) chunk 第 0 步
        mode = int(mode_logits[0, 0].argmax().item())    # 0/1
        a = np.zeros(12, dtype=np.float32)
        a[CONT_IDX] = np.clip(cont, -1, 1)
        a[MODE_IDX] = -1.0 if mode == 0 else 1.0
        a[11] = 1.0 if a[11] > 0 else -1.0               # gripper 二值化 (同 UniVLA LIBERO)
        return a

    ctx = zmq.Context(); sock = ctx.socket(zmq.REP); sock.bind(f"tcp://127.0.0.1:{cfg.port}")
    print(f"policy server ready on {cfg.port} (variant={cfg.vision_variant})", flush=True)
    while True:
        req = msgpack.unpackb(sock.recv(), raw=False)
        cmd = req["cmd"]
        if cmd == "ping":
            sock.send(msgpack.packb({"ok": True}))
        elif cmd == "reset":
            state["instr"] = req["instruction"].lower(); state["hist"] = [""]
            sock.send(msgpack.packb({"ok": True}))
        elif cmd == "act":
            il = np.frombuffer(req["left"], dtype=np.uint8).reshape(req["shape"])
            iw = np.frombuffer(req["wrist"], dtype=np.uint8).reshape(req["shape"]) if req.get("wrist") else None
            pr = np.array(req["proprio"], dtype=np.float32)
            a = act(il, iw, pr)
            sock.send(msgpack.packb({"action": a.tolist()}))
        elif cmd == "stop":
            sock.send(msgpack.packb({"ok": True})); break
    sock.close(); ctx.term()


if __name__ == "__main__":
    main()
