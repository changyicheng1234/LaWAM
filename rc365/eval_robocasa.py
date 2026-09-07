"""RoboCasa365 rollout client (rc365 env, py3.12).  [agent-B, PLAN S4]
连 serve_policy.py (univla env) 逐 step 要 action. round-1 跑通用.
"""
import os, sys, json, time, argparse, numpy as np, zmq, msgpack

def get_proprio(obs):
    return np.concatenate([
        np.asarray(obs["robot0_base_pos"], np.float32),          # 3  -> state[0:3]
        np.asarray(obs["robot0_base_quat"], np.float32),         # 4  -> state[3:7] (idx3,4 恒 0)
        np.asarray(obs["robot0_base_to_eef_pos"], np.float32),   # 3  -> state[7:10]
        np.asarray(obs["robot0_base_to_eef_quat"], np.float32),  # 4  -> state[10:14]
        np.asarray(obs["robot0_gripper_qpos"], np.float32),      # 2  -> state[14:16]
    ]).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="NavigateKitchen,CloseToasterOvenDoor")
    ap.add_argument("--n_episodes", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=350)
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--save_video_n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--egl_device", type=int, default=5)
    cfg = ap.parse_args()
    os.environ["MUJOCO_GL"] = "egl"; os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(cfg.egl_device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.egl_device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    import robosuite
    from robosuite.controllers import load_composite_controller_config
    import robocasa  # noqa
    import imageio

    ctx = zmq.Context(); sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 120000); sock.connect(f"tcp://127.0.0.1:{cfg.port}")
    def rpc(d): sock.send(msgpack.packb(d)); return msgpack.unpackb(sock.recv(), raw=False)
    print("ping server:", rpc({"cmd": "ping"}), flush=True)

    ctrl = load_composite_controller_config(robot="PandaOmron")
    results = {}
    for task in cfg.tasks.split(","):
        env = robosuite.make(env_name=task, robots="PandaOmron", controller_configs=ctrl,
            has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
            camera_names=["robot0_agentview_left", "robot0_eye_in_hand"],
            camera_heights=256, camera_widths=256, control_freq=20, obj_registries=("lightwheel",))
        instr = " ".join([w for w in _camel(task)])
        succ = []; eps_s = []
        for ep in range(cfg.n_episodes):
            try:
                if callable(getattr(env, "seed", None)): env.seed(cfg.seed + ep)
            except Exception: pass
            obs = env.reset()
            rpc({"cmd": "reset", "instruction": instr})
            frames = []; done_succ = False; t0 = time.time()
            for t in range(cfg.max_steps):
                il = np.flipud(obs["robot0_agentview_left_image"]).copy()
                iw = np.flipud(obs["robot0_eye_in_hand_image"]).copy()
                if ep < cfg.save_video_n:
                    frames.append(il)
                rep = rpc({"cmd": "act", "left": il.tobytes(), "wrist": iw.tobytes(),
                           "shape": list(il.shape), "proprio": get_proprio(obs).tolist()})
                a = np.array(rep["action"], dtype=np.float32)
                obs, r, done, info = env.step(a)
                if env._check_success():
                    done_succ = True; break
            dt = time.time() - t0
            succ.append(int(done_succ)); eps_s.append(dt)
            print(f"[{task}] ep{ep}: success={done_succ} steps={t+1} {dt:.1f}s", flush=True)
            if ep < cfg.save_video_n and frames:
                base = os.path.join(cfg.out_dir, f"{task}_ep{ep}_{'S' if done_succ else 'F'}")
                try:
                    imageio.mimwrite(base + ".mp4", frames, fps=20, codec="libx264", macro_block_size=1)
                except Exception as e:
                    np.save(base + ".npy", np.stack(frames)); print("  video->npy (", e, ")", flush=True)
        env.close()
        results[task] = {"sr": float(np.mean(succ)), "n": len(succ), "succ": succ,
                         "mean_ep_s": float(np.mean(eps_s))}
        print(f"[{task}] SR = {np.mean(succ):.2f} ({sum(succ)}/{len(succ)})", flush=True)

    alls = [s for v in results.values() for s in v["succ"]]
    results["_overall"] = {"sr": float(np.mean(alls)), "n": len(alls)}
    json.dump(results, open(os.path.join(cfg.out_dir, "results.json"), "w"), indent=2)
    print("OVERALL SR:", results["_overall"], flush=True)
    rpc({"cmd": "stop"})

def _camel(s):
    import re
    s = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', s)
    s = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', s)
    return s.lower().split()

if __name__ == "__main__":
    main()
