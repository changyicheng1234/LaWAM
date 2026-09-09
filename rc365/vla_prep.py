"""训练/冒烟共用: batch -> LAM <ACT> -> prompt -> input_ids/labels/attention_mask.
照 finetune_realworld.py 的 loop 逻辑抽出来. [agent-B, PLAN S2]"""
import torch
from torch.nn.utils.rnn import pad_sequence
from prismatic.models.backbones.llm.prompting import PurePromptBuilder

IGNORE = -100


def _unwrap(m):
    return m.module if hasattr(m, "module") else m


@torch.no_grad()
def build_vla_inputs(batch, latent_action_model, processor, device, direct=False):
    """就地给 batch 加 input_ids / attention_mask / labels. 返回 batch.

    direct=True (round-2 `--direct`): prompt 里不放 <ACT_*> token, labels 全 IGNORE
    -> 没有 vla_ce 监督, action decoder 退回读 prompt 尾部 hidden states."""
    lam = _unwrap(latent_action_model)
    ini = batch["initial_pixel_values"].to(device)
    tgt = batch["target_pixel_values"].to(device)
    video = torch.stack([ini, tgt], dim=1)                       # (B,2,3,224,224)
    lai = lam.vq_encode(video)["indices"]                        # (B,4)
    if lai.dim() == 1:
        lai = lai.unsqueeze(0)

    has_hist = bool(len(batch.get("initial_pixel_values_hist", [])) > 1) if not torch.is_tensor(
        batch.get("initial_pixel_values_hist", [])) else (batch["initial_pixel_values_hist"].shape[0] > 0)
    lai_hist = None
    if has_hist:
        ih = batch["initial_pixel_values_hist"].to(device)
        th = batch["target_pixel_values_hist"].to(device)
        lai_hist = lam.vq_encode(torch.stack([ih, th], dim=1))["indices"]
        if lai_hist.dim() == 1:
            lai_hist = lai_hist.unsqueeze(0)

    tok = processor.tokenizer
    ids_list, lbl_list = [], []
    hist_ptr = 0
    with_hist = batch.get("with_hist")
    for i in range(lai.shape[0]):
        act_vocab = [f"<ACT_{j.item()}>" for j in lai[i]]
        act_tokens = "".join(act_vocab)
        instr = batch["instructions"][i].lower()
        if with_hist is not None and bool(with_hist[i]) and lai_hist is not None:
            hv = "".join(f"<ACT_{j.item()}>" for j in lai_hist[hist_ptr]); hist_ptr += 1
            prompt = f"What action should the robot take to {instr}? History action " + hv
        else:
            prompt = f"What action should the robot take to {instr}?"
        pb = PurePromptBuilder("openvla")
        pb.add_turn("human", prompt)
        pb.add_turn("gpt", "" if direct else act_tokens)
        ids = tok(pb.get_prompt(), add_special_tokens=True).input_ids
        lbl = list(ids)
        ids, lbl = torch.tensor(ids), torch.tensor(lbl)
        if direct:
            lbl[:] = IGNORE
        else:
            lbl[: -(len(act_vocab) + 1)] = IGNORE
        ids_list.append(ids); lbl_list.append(lbl)

    input_ids = pad_sequence(ids_list, batch_first=True, padding_value=tok.pad_token_id)
    labels = pad_sequence(lbl_list, batch_first=True, padding_value=IGNORE)
    input_ids = input_ids[:, : tok.model_max_length]
    labels = labels[:, : tok.model_max_length]
    attn = input_ids.ne(tok.pad_token_id)

    batch["input_ids"] = input_ids.to(device)
    batch["attention_mask"] = attn.to(device)
    batch["labels"] = labels.to(device)
    batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16).to(device)
    batch["actions"] = batch["actions"].to(device)
    batch["proprio"] = batch["proprio"].to(device)
    if "control_mode_tgt" in batch:
        batch["control_mode_tgt"] = batch["control_mode_tgt"].to(device)
    if "action_is_pad" in batch:
        batch["action_is_pad"] = batch["action_is_pad"].to(device)
    return batch
