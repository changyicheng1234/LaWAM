"""Round-2 WrappedModel: round-1 `model_robocasa.WrappedModelRoboCasa` + LoRA /
`vla_ce` / `direct` knobs, and the extra diagnostics that showed up in the
`r2_crank` logs (`act_acc`, `act_distinct`, `grip_acc`).

loss = decoder_loss_weight * (l1 + ce_weight * mode_ce)
       + (0 if direct else vla_ce_weight * vla_ce)

- LoRA / full-FT: `freeze_vla=False`, VLA forward runs with grad and `labels`
  are supplied so `vla_output.loss` (the CE over the `<ACT_*>` targets) is live.
- `direct=True`: the prompt carries no `<ACT_*>` tokens (see `vla_prep`), so the
  action decoder falls back to the last 4 LLM hidden states and there is no
  `vla_ce` term -- the VLA is just a LoRA-adapted backbone.

Chunk = `window_size` steps, exactly as round-1 (`proj_cont -> (B, ws, 11)`);
eval executes `exec_horizon` of them.
"""
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_robocasa import ActionDecoderRoboCasa, CONT_IDX, MODE_IDX, N_CONT

# gripper is continuous action idx 11 (bimodal +/-1); its position inside the
# 11-d CONT vector:
GRIP_CONT_POS = CONT_IDX.index(11)
ACT_TOKEN_MIN = 32000  # `<ACT_*>` ids live above this in the LLaMA tokenizer


class WrappedModelRoboCasaR2(nn.Module):
    def __init__(self, vla, *, freeze_vla=False, window_size=12, ce_weight=1.0,
                 mode_class_weight=None, decoder_loss_weight=1.0, vla_ce_weight=1.0,
                 direct=False):
        super().__init__()
        self.vla = vla
        self.freeze_vla = freeze_vla
        self.window_size = window_size
        self.ce_weight = ce_weight
        self.decoder_loss_weight = decoder_loss_weight
        self.vla_ce_weight = vla_ce_weight
        self.direct = direct
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
        # unwrap PEFT (PeftModel.base_model -> LoraModel.model -> OpenVLA...)
        for _ in range(3):
            if hasattr(m, "vision_backbone"):
                break
            m = getattr(m, "base_model", None) or getattr(m, "model", None) or m
        return m.vision_backbone.featurizer.patch_embed.num_patches

    def forward(self, batch):
        want_vla_ce = (not self.direct) and self.vla_ce_weight != 0.0
        fwd_ctx = torch.no_grad() if self.freeze_vla else nullcontext()
        with fwd_ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            vla_output = self.vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"] if want_vla_ce else None,
                output_hidden_states=True,
            )
        np_ = self._num_patches()
        hidden = vla_output.hidden_states[-1]
        visual_embed = hidden[:, :np_].float()                      # (B,256,4096)
        latent_tokens = hidden[:, np_:]                             # (B,L,4096)
        labels = batch["labels"].to(latent_tokens.device)
        mask = labels > ACT_TOKEN_MIN                               # (B,L) all-False in `direct`

        lat_list = []
        for i, toks in enumerate(latent_tokens):
            sel = toks[mask[i], :]
            if sel.shape[0] < 4:                                    # `direct`, or short: take prompt tail
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

        decoder_loss = l1 + self.ce_weight * ce
        loss = self.decoder_loss_weight * decoder_loss
        vla_ce = vla_output.loss if want_vla_ce else None
        if vla_ce is not None:
            loss = loss + self.vla_ce_weight * vla_ce

        with torch.no_grad():
            pred_mode = mode_logits.argmax(-1)
            mode_acc = (pred_mode == mode_tgt).float().mean()
            l1_1step = F.l1_loss(cont[:, 0], cont_tgt[:, 0])
            grip_acc = ((cont[..., GRIP_CONT_POS] > 0) == (cont_tgt[..., GRIP_CONT_POS] > 0)).float().mean()
            act_acc = torch.tensor(0.0, device=cont.device)
            act_distinct = torch.tensor(0.0, device=cont.device)
            if want_vla_ce and getattr(vla_output, "logits", None) is not None:
                lg = vla_output.logits                              # (B, n_patches+L, V)
                L = labels.shape[1]
                txt_lg = lg[:, -L:]                                 # (B, L, V) text region
                shift_lg = txt_lg[:, :-1]
                shift_lbl = labels[:, 1:]
                m = shift_lbl > ACT_TOKEN_MIN
                if m.any():
                    pred_tok = shift_lg.argmax(-1)[m]
                    tgt_tok = shift_lbl[m]
                    act_acc = (pred_tok == tgt_tok).float().mean()
                    # collapse detector: how many distinct ACT ids the VLA emits
                    act_distinct = torch.tensor(float(pred_tok.unique().numel()), device=cont.device)

        return dict(loss=loss, l1=l1.detach(), ce=ce.detach(), decoder_loss=decoder_loss.detach(),
                    mode_acc=mode_acc, ce_acc=mode_acc, grip_acc=grip_acc,
                    l1_1step=l1_1step, act_acc=act_acc, act_distinct=act_distinct,
                    pos_rate=mode_tgt.float().mean(), pred_pos_rate=pred_mode.float().mean(),
                    vla_ce=(vla_ce.detach() if vla_ce is not None else None))
