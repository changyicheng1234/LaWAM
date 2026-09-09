"""UniVLA round-2 policy server (univla env, py3.10).  Step 4.

Round-1 `serve_policy.py` + LoRA adapter + `--exec_horizon` action chunking:
predict a `window_size` chunk, execute `exec_horizon` steps of it open-loop,
then re-query.  The zmq protocol is unchanged (`eval_robocasa.py` needs no
edits) -- the chunk cache lives here.

  python serve_policy_r2.py --vision_variant v1 --exec_horizon 8 \
      --vla_path /root/weights/univla-7b \
      --lora_adapter /root/runs/r2_crank/CloseToasterOvenDoor/v1/lora_adapter \
      --decoder_path /root/runs/r2_crank/CloseToasterOvenDoor/v1/action_decoder_last.pt \
      --proprio_stats /root/runs/r2_crank/CloseToasterOvenDoor/v1/proprio_stats.npz
"""
import argparse
import os
import sys

import msgpack
import numpy as np
import torch
import zmq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_robocasa import ActionDecoderRoboCasa, CONT_IDX, MODE_IDX  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vla_path", default="/root/weights/univla-7b")
    ap.add_argument("--lora_adapter", default="", help="PEFT adapter dir; empty => base VLA (direct/no-lora runs)")
    ap.add_argument("--decoder_path", required=True)
    ap.add_argument("--proprio_stats", required=True)
    ap.add_argument("--vision_variant", choices=["v1", "v2"], required=True)
    ap.add_argument("--v2_each_size", type=int, default=224)
    ap.add_argument("--window_size", type=int, default=12)
    ap.add_argument("--exec_horizon", type=int, default=8, help="steps of each predicted chunk to run open-loop")
    ap.add_argument("--direct", action="store_true", help="match a --direct round-2 run (no <ACT_*> in prompt)")
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--device", default="cuda:0")
    cfg = ap.parse_args()
    dev = cfg.device
    assert 1 <= cfg.exec_horizon <= cfg.window_size

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
        trust_remote_code=True, attn_implementation="flash_attention_2").to(dev)
    if cfg.lora_adapter:
        from peft import PeftModel
        vla = PeftModel.from_pretrained(vla, cfg.lora_adapter)
        vla = vla.merge_and_unload()          # fold LoRA into base -> plain forward speed
        print(f"merged LoRA adapter {cfg.lora_adapter}", flush=True)
    vla = vla.to(dev).eval()

    dec = ActionDecoderRoboCasa(window_size=cfg.window_size)
    dec.load_state_dict(torch.load(cfg.decoder_path, map_location="cpu"))
    dec = dec.to(dev).eval()
    st = np.load(cfg.proprio_stats)
    pm = st["mean"].astype(np.float32)
    ps = st["std"].astype(np.float32).copy(); ps[ps < 1e-6] = 1.0

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

    state = {"instr": "", "hist": "", "chunk": None, "i": 0}

    @torch.no_grad()
    def predict_chunk(img_left, img_wrist, proprio_raw):
        pil = compose(img_left, img_wrist)
        prompt = f"In: What action should the robot take to {state['instr']}?"
        if not cfg.direct and len(state["hist"]) > 0:
            prompt = f"In: What action should the robot take to {state['instr']}? History action {state['hist']}"
        prompt += "\nOut:"
        inputs = proc(prompt, pil).to(dev, dtype=torch.bfloat16)
        # use_cache=False keeps per-step hidden_states aligned (round-1 serve note)
        lat, vemb, gids = vla.predict_latent_action(**inputs, do_sample=False, use_cache=False)
        if not cfg.direct:
            det = [f"<ACT_{i}>" for i in range(32)]
            state["hist"] = "".join(det[g.item() - 32001] for g in gids[0])
        pr = torch.from_numpy(((proprio_raw - pm) / ps).astype(np.float32))[None].to(dev)
        cont, mode_logits = dec(lat.float(), vemb.float(), pr)          # (1,ws,11) (1,ws,2)
        cont = np.clip(cont[0].float().cpu().numpy(), -1, 1)            # (ws,11)
        mode = mode_logits[0].argmax(-1).cpu().numpy()                 # (ws,)
        chunk = np.zeros((cfg.window_size, 12), dtype=np.float32)
        chunk[:, CONT_IDX] = cont
        chunk[:, MODE_IDX] = np.where(mode == 0, -1.0, 1.0)
        chunk[:, 11] = np.where(chunk[:, 11] > 0, 1.0, -1.0)           # binarize gripper (as UniVLA LIBERO)
        return chunk

    def act(img_left, img_wrist, proprio_raw):
        if state["chunk"] is None or state["i"] >= cfg.exec_horizon:
            state["chunk"] = predict_chunk(img_left, img_wrist, proprio_raw)
            state["i"] = 0
        a = state["chunk"][state["i"]].copy()
        state["i"] += 1
        return a

    ctx = zmq.Context(); sock = ctx.socket(zmq.REP); sock.bind(f"tcp://127.0.0.1:{cfg.port}")
    print(f"policy server ready on {cfg.port} (variant={cfg.vision_variant} "
          f"exec_horizon={cfg.exec_horizon} lora={'yes' if cfg.lora_adapter else 'no'})", flush=True)
    while True:
        req = msgpack.unpackb(sock.recv(), raw=False)
        cmd = req["cmd"]
        if cmd == "ping":
            sock.send(msgpack.packb({"ok": True}))
        elif cmd == "reset":
            state["instr"] = req["instruction"].lower()
            state["hist"] = ""; state["chunk"] = None; state["i"] = 0
            sock.send(msgpack.packb({"ok": True}))
        elif cmd == "act":
            il = np.frombuffer(req["left"], dtype=np.uint8).reshape(req["shape"])
            iw = np.frombuffer(req["wrist"], dtype=np.uint8).reshape(req["shape"]) if req.get("wrist") else None
            pr = np.array(req["proprio"], dtype=np.float32)
            sock.send(msgpack.packb({"action": act(il, iw, pr).tolist()}))
        elif cmd == "stop":
            sock.send(msgpack.packb({"ok": True})); break
    sock.close(); ctx.term()


if __name__ == "__main__":
    main()
