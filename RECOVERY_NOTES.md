# Recovery notes — changyicheng1234 fork

This fork of `RLinf/LaWAM` also carries the working code for two experiment lines
that were lost when `/mnt/workspace/` was wiped on a server reboot (2026-09-07).

## Layout added on top of upstream

| Path | What it is | Upstream |
|---|---|---|
| `UniVLA/` | Vendored copy of `OpenDriveLab/UniVLA` @ `0ab9e9d` (source only, its `.git` removed) | https://github.com/OpenDriveLab/UniVLA |
| `rc365/` | My RoboCasa365 finetune/eval scripts for UniVLA (was `/mnt/workspace/changyicheng/work/univla/rc365`) | — |

Everything under the original LaWAM directories (`latent_action_model/`,
`starVLA/`, `examples/`, `deployment/`, …) is upstream `RLinf/LaWAM` `main` and was
re-cloned fresh — any local edits I had before the reboot are gone.

## Large artifacts NOT in git (they live on the persistent disk `/mnt/data/changyicheng/`)

| Path | Contents |
|---|---|
| `univla/weights/univla-7b/` | UniVLA-7B base checkpoint (15 GB, complete) |
| `univla/weights/univla-7b-224-sft-libero/univla-libero-10/` | LIBERO SFT checkpoint (15 GB) |
| `univla/weights/univla-latent-action-model/lam-stage-2.ckpt` | LAM stage-2 (2.6 GB) |
| `univla/univla-env.tar.gz` | UniVLA conda env pack (4.6 GB) |
| `envs/rc365-py312-cu126-*.tar.zst` + `restore_rc365.sh` | rc365 conda env restore kit (torch 2.7.1+cu126, mujoco 3.3.1, robocasa 1.0.1 `be22d65`, robosuite 1.5.2 `85abee2`, lerobot 0.6.1) |
| `rc365/rc365_runs/round1/{v1,v2}` | Trained `action_decoder_*.pt`, `effective_config.json`, `proprio_stats.npz`, `train.log` |
| `rc365/rc365_eval/{v1,v2}_results.json` | round-1 eval: SR 0.0 on NavigateKitchen + CloseToasterOvenDoor (training converged, rollout failed) |
| `datasets/robocasa_CloseFridge/` | LeRobot v3.0, 513 eps — intact |
| `datasets/robocasa_target_human_unified/` | ⚠️ only `meta/` survived; `data/` + `videos/` empty — main finetune training set, needs re-download |

## To restore the rc365 environment

```bash
bash /mnt/data/changyicheng/envs/restore_rc365.sh
# then re-clone + `pip install -e` robocasa (be22d65) and robosuite (85abee2)
```

## Remotes

- `origin`   → https://github.com/changyicheng1234/LaWAM.git  (this fork, push here)
- `upstream` → https://github.com/RLinf/LaWAM.git             (pull upstream updates)
