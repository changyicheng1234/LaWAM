"""RoboCasa365 的 ActionDecoder + WrappedModel.  [agent-B, PLAN S2]

- 连续动作头: 11 维 L1 (action 全 12 维去掉 idx4 control_mode; 含恒 0 的 idx3)
- 离散头:     control_mode(idx4) 的 2 类 CE
- proprio:    16 维 -> Proprio Projection (2 层 MLP hidden 512), 与 latent-pool 输出 cat
- decoder-only: freeze_vla=True 时 backbone requires_grad_(False)+eval(), 前向 no_grad
"""
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
from prismatic.models.policy.transformer_utils import MAPBlock

# 连续回归的动作维 (去掉 idx4 = control_mode)
CONT_IDX = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11]      # 11 维
MODE_IDX = 4
N_CONT = len(CONT_IDX)


class ActionDecoderRoboCasa(nn.Module):
    def __init__(self, window_size=12, hidden_dim=512):
        super().__init__()
        self.window_size = window_size
        self.latent_action_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64)
        self.visual_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64)
        self.proprio_proj = nn.Sequential(
            nn.Linear(16, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.proj_cont = nn.Sequential(nn.Linear(hidden_dim * 2, N_CONT * window_size), nn.Tanh())
        self.head_mode = nn.Linear(hidden_dim * 2, 2 * window_size)

    def forward(self, latent_action_tokens, visual_embed, proprio):
        v = self.visual_pool(visual_embed)                                  # (B,512)
        lat = self.latent_action_pool(latent_action_tokens[:, -4:], init_embed=v)  # (B,512)
        p = self.proprio_proj(proprio)                                      # (B,512)
        feat = torch.cat([lat, p], dim=-1)                                  # (B,1024)
        cont = self.proj_cont(feat).reshape(-1, self.window_size, N_CONT)   # (B,ws,11)
        mode_logits = self.head_mode(feat).reshape(-1, self.window_size, 2) # (B,ws,2)
        return cont, mode_logits


class WrappedModelRoboCasa(nn.Module):
    def __init__(self, vla, freeze_vla=True, window_size=12, ce_weight=1.0, mode_class_weight=None):
        super().__init__()
        self.vla = vla
        self.freeze_vla = freeze_vla
        self.window_size = window_size
        self.ce_weight = ce_weight
        self.register_buffer(
            "mode_cw",
            torch.tensor(mode_class_weight, dtype=torch.float32) if mode_class_weight is not None else None,
            persistent=False,
        )
        self.action_decoder = ActionDecoderRoboCasa(window_size=window_size)
        if freeze_vla:
            self.vla.requires_grad_(False)
            self.vla.eval()

    def _num_patches(self):
        m = self.vla
        return m.vision_backbone.featurizer.patch_embed.num_patches

    def forward(self, batch):
        fwd_ctx = torch.no_grad() if self.freeze_vla else nullcontext()
        with fwd_ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            vla_output = self.vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=None if self.freeze_vla else batch["labels"],
                output_hidden_states=True,
            )
        np_ = self._num_patches()
        hidden = vla_output.hidden_states[-1]
        visual_embed = hidden[:, :np_].float()                     # (B,256,4096)
        latent_tokens = hidden[:, np_:]                            # (B,L,4096)
        mask = batch["labels"].to(latent_tokens.device) > 32000    # (B,L)

        lat_list = []
        for i, toks in enumerate(latent_tokens):
            sel = toks[mask[i], :]
            if sel.shape[0] < 4:                                   # 保险: 不足 4 个则右侧补最后一个
                pad = sel[-1:].repeat(4 - sel.shape[0], 1) if sel.shape[0] > 0 else toks[-4:]
                sel = torch.cat([sel, pad], dim=0)
            lat_list.append(sel[-4:])
        latent_action_tokens = torch.stack(lat_list).float()       # (B,4,4096)

        cont, mode_logits = self.action_decoder(latent_action_tokens, visual_embed, batch["proprio"].float())

        act = batch["actions"].float()                             # (B,ws,12)
        cont_tgt = act[:, :, CONT_IDX]                             # (B,ws,11)
        mode_tgt = batch["control_mode_tgt"].to(cont.device).long()  # (B,ws)
        is_pad = batch.get("action_is_pad")
        if is_pad is not None:
            w = is_pad.to(cont.device).float().unsqueeze(-1)       # (B,ws,1)
            l1 = (F.l1_loss(cont, cont_tgt, reduction="none") * w).sum() / w.sum().clamp_min(1) / N_CONT
            ce_all = F.cross_entropy(mode_logits.reshape(-1, 2), mode_tgt.reshape(-1),
                                     weight=self.mode_cw.to(cont.device) if self.mode_cw is not None else None,
                                     reduction="none").reshape(mode_tgt.shape)
            ce = (ce_all * w.squeeze(-1)).sum() / w.squeeze(-1).sum().clamp_min(1)
        else:
            l1 = F.l1_loss(cont, cont_tgt)
            ce = F.cross_entropy(mode_logits.reshape(-1, 2), mode_tgt.reshape(-1),
                                 weight=self.mode_cw.to(cont.device) if self.mode_cw is not None else None)

        loss = l1 + self.ce_weight * ce
        with torch.no_grad():
            pred_mode = mode_logits.argmax(-1)
            ce_acc = (pred_mode == mode_tgt).float().mean()
            l1_1step = F.l1_loss(cont[:, 0], cont_tgt[:, 0])
            pos_rate = mode_tgt.float().mean()
            pred_pos_rate = pred_mode.float().mean()
        return dict(loss=loss, l1=l1.detach(), ce=ce.detach(), ce_acc=ce_acc,
                    l1_1step=l1_1step, pos_rate=pos_rate, pred_pos_rate=pred_pos_rate,
                    vla_ce=(vla_output.loss.detach() if vla_output.loss is not None else None))
