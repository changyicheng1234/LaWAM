# CLAUDE.md — changyicheng 的工作环境说明（给后续 agent）

> 这台 Aliyun DSW 实例**重启会清空非持久盘**。2026-09-07 一次重启把 `/mnt/workspace` 下的所有代码/环境清掉了，才有了这份恢复工作。看这份文件之前请先读 `~/.claude/projects/-mnt-data-changyicheng-dev/memory/` 里的 memory。

---

## 1. 磁盘规则（最重要）

**这台机器没有"又快又持久"的盘。** 必须区分对待：

| 挂载点 | 类型 | 重启后保留 | 速度 | 用途 |
|---|---|---|---|---|
| `/`、`/root` | overlay（容器 rootfs，~4.7T） | ❌ **清空** | ⚡ ~3 GB/s | 代码、训练时的数据、正在写的 checkpoint |
| `/mnt/workspace` | ext4（`emptydir/host-volume`，~30G） | ❌ **清空** | ⚡ 本地快 | 只放小临时文件，别依赖它 |
| `/dev/shm` | tmpfs（内存，1.6T） | ❌ 易失 | ⚡⚡ 内存 | 热的小数据集 |
| `/mnt/data`、`/mnt/public` | ossfs2（OSS 对象存储，512T） | ✅ **持久** | 🐌 顺序写 ~550MB/s，**随机读/大量小文件极慢**（每次 open = 一次 HTTP） | 备份、golden copy、归档 —— **不要在这上面跑训练** |

### 每次训练 / eval 前的固定动作
1. **代码**：在 `/root/dev/` 下工作，随时 `git push`（见 §3）。
2. **权重**：从 `/mnt/data/changyicheng/...` `cp` 到 `/root/weights/`。（只加载一次也许能忍，但 resume/eval 会反复读，拷过去。）
3. **数据集**：从 `/mnt/data` `cp` 到 `/root/data/` 或 `/dev/shm/`。
   **绝对不要**把 LeRobot / torch dataloader 直接指向 `/mnt/data` —— 随机读几千个 parquet/mp4 会慢到没法训练。
4. 训练产物写到 `/root/...`，**关机前**把最终 checkpoint 拷回 `/mnt/data/changyicheng/` 或传 ModelScope，否则丢。

---

## 2. 两条工作线 & 文件位置

### A) LaWAM on RoboCasa
- 代码：`/root/dev/LaWAM/`（`RLinf/LaWAM` 的 fork，见 §3）
- 备份（ModelScope 拉回）：`/mnt/data/changyicheng/restore/lawam-robocasa-backup/`
  - `sft/{CloseToasterOvenDoor,OpenDrawer,TurnOnMicrowave}/final_model/pytorch_model.pt` — 3 任务 SFT final model（各 ~7GB）
  - `env/lawam-py310-cu124-20260907.tar.zst` — LaWAM 专用 conda 环境包（py3.10 / cu124），恢复方式类似 §4
  - `pool/data.tar.gz`、`eval/` rollout npy
- LaWAM 用的是**它自己的环境**，不是 rc365。

### B) UniVLA on RoboCasa365（rc365）
- 代码：
  - `/root/dev/LaWAM/UniVLA/` — vendored `OpenDriveLab/UniVLA` 源码（@ 0ab9e9d）
  - `/root/dev/LaWAM/rc365/` — 微调/eval 脚本（`finetune_robocasa.py` `eval_robocasa.py` `serve_policy.py` …）
  - 脚本里的老路径 `/mnt/workspace/changyicheng/work/univla/...` 已失效，改用 `/root/dev/`。
- round-1 产物（持久盘一直在）：`/mnt/data/changyicheng/rc365/rc365_runs/`、`rc365_eval/`
- round-2 产物（ModelScope 拉回）：`/mnt/data/changyicheng/restore/univla-rc365-backup/`
  - `rc365_runs/lam_robocasa/` — robocasa 上训的 LAM ckpt
  - `rc365_runs/round2_lora/` `r2_pertask/` `r2_crank/` `r2_direct*/` — round-2 LoRA adapters
- 基座权重（持久盘，~32GB，完整）：`/mnt/data/changyicheng/univla/weights/`
  - `univla-7b/`、`univla-7b-224-sft-libero/univla-libero-10/`、`univla-latent-action-model/lam-stage-2.ckpt`

### 数据集（`/mnt/data/changyicheng/datasets/`）
- `robocasa_CloseFridge/` — 完整（LeRobot v3.0，513 eps）
- `robocasa_target_human_unified/` — ⚠️ **只剩 `meta/`，`data/` 和 `videos/` 是空的**。这是主微调训练集（50 任务 / 25307 eps）。**没备份到 ModelScope**，需要从原始公开来源重新下载。

---

## 3. GitHub

- 账号：`changyicheng1234`（ModelScope 同名，但 ModelScope 在**国内站 modelscope.cn**，见 §5）
- 仓库：**`changyicheng1234/LaWAM`**（`RLinf/LaWAM` 的 fork）—— LaWAM 和 UniVLA 都放这一个仓库
  - `origin` → `https://github.com/changyicheng1234/LaWAM.git`（push 到这）
  - `upstream` → `https://github.com/RLinf/LaWAM.git`
- git 身份已配：`changyicheng1234 / yic19250@gmail.com`
- **没有** `gh` CLI，也没有持久化的 git 凭证。push 时需要用户提供 GitHub PAT，用临时 `GIT_ASKPASS` 传（别写进 `.git/config`）。
- 大文件（权重/ckpt/数据集/env 包）**不进 git**，留在 `/mnt/data` 或传 ModelScope。

---

## 4. 恢复 rc365（RoboCasa365）环境

已恢复到 `/root/conda/envs/rc365`（重启后需重做）。自检通过：`torch 2.7.1+cu126 / 8 GPU / robocasa 1.0.1 / robosuite 1.5.2 / mujoco 3.3.1 / lerobot 0.6.1`。

重启后重新恢复：
```bash
# 1. zstd 不可用(apt 装不了) -> 用 pip 的 zstandard 解包；miniforge 用清华镜像(快, ~37MB/s)
bash /tmp/.../restore_rc365_v2.sh    # 见 scratchpad，或按下面手动
```
手动关键步骤：
1. Miniforge → `/root/conda`：`https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh`（**别用 mirror.nju，慢到 55KB/s**）
2. 解包 `/mnt/data/changyicheng/envs/rc365-py312-cu126-20260901.tar.zst` → `/root/conda/envs/rc365`
   无 `zstd` 二进制：`pip install zstandard` 后用 `zstandard.ZstdDecompressor().stream_reader` + `tarfile.open(mode="r|")`
3. `/root/conda/envs/rc365/bin/conda-unpack`
4. editable 装：`conda run -n rc365 pip install --no-deps -e /root/dev/robocasa` 和 `-e /root/dev/robosuite`
5. robocasa 厨房资产(~10GB)：`yes y | conda run -n rc365 python -m robocasa.scripts.download_kitchen_assets`
   （装进 `/root/dev/robocasa/robocasa/models/assets/`，在 `/` 上 → 重启也要重下。另一用户 `/mnt/data/l30083605/robocasa`(同 commit `be22d65`) 已有资产，可拷。）

repo commit：robocasa `be22d65`(v1.0.1)，robosuite `85abee2`(v1.5.2)。
`robosuite` / `robocasa` 首次跑会提示 `setup_macros.py`，按提示跑一次即可消警告。

---

## 5. ModelScope

- 用户账号 `changyicheng1234` 在**国内站 `https://www.modelscope.cn`**（不是国际站 modelscope.ai，两站账号/token/仓库互不相通）。
- 国内站 token 形如 `ms-<uuid>`。SDK 用法：`export MODELSCOPE_ENDPOINT=https://www.modelscope.cn` 再 `HubApi().login(token)`。
- 可用的 SDK venv：`/root/dev/.ms-venv`（modelscope 1.39.1 + zstandard）。
- 两个私有备份仓库（均 model 类型）：
  - `changyicheng1234/univla-rc365-backup`（~19GB）
  - `changyicheng1234/lawam-robocasa-backup`（~27GB）
  - **没有 dataset 仓库** → `robocasa_target_human_unified` 数据未备份。
- 下载用 SDK `snapshot_download(repo_id, repo_type="model", local_dir=..., max_workers=8)`，比 git 快。

---

## 6. 其它
- 代理：clash 在 `127.0.0.1:7890`（`http_proxy`/`https_proxy`/`all_proxy` 已设）。国际流量走它；国内镜像(清华/ModelScope.cn)直连更快，必要时 `unset *_proxy`。
- GPU：8 卡。
- 这份文件的持久副本在 `/mnt/data/changyicheng/dev/CLAUDE.md`，另一份在 `/root/dev/CLAUDE.md`。
