"""汇总 round-1: 训练曲线要点 + V1/V2 eval 结果 + config diff -> 打印 + 写 md 片段."""
import json, os, re

def last_metrics(logpath):
    if not os.path.exists(logpath): return None
    steps = []
    for ln in open(logpath):
        m = re.match(r"step\s+(\d+) \| loss ([\d.]+) l1 ([\d.]+) \(init ([\d.]+)\) ce ([\d.]+) \(init ([\d.]+)\) ce_acc ([\d.]+)", ln)
        if m: steps.append(tuple(float(x) for x in m.groups()))
    if not steps: return None
    s0, sl = steps[0], steps[-1]
    return dict(n=int(sl[0]), l1_init=s0[3], l1_last=sl[2], ce_init=s0[5], ce_last=sl[4], ce_acc_last=sl[6])

def find_log(variant):
    d = "/mnt/workspace/changyicheng/work/univla/logs"
    cand = sorted([f for f in os.listdir(d) if f.startswith(f"train_{variant}_")], reverse=True)
    return os.path.join(d, cand[0]) if cand else None

lines = ["## round-1 结果汇总\n"]
lines.append("| variant | steps | l1 init→last | ce init→last | ce_acc last | eval SR (overall) | 逐任务 SR |")
lines.append("|---|---|---|---|---|---|---|")
for v in ["v1", "v2"]:
    tm = last_metrics(find_log(v)) or {}
    ev = {}
    ep = f"/opt/rc365_eval/round1_{v}/results.json"
    if os.path.exists(ep):
        ev = json.load(open(ep))
    ov = ev.get("_overall", {}).get("sr")
    per = " ; ".join(f"{k}={vv['sr']:.2f}({sum(vv['succ'])}/{vv['n']})"
                     for k, vv in ev.items() if not k.startswith("_"))
    lines.append(f"| {v} | {tm.get('n','?')} | "
                 f"{tm.get('l1_init',0):.3f}→{tm.get('l1_last',0):.3f} | "
                 f"{tm.get('ce_init',0):.3f}→{tm.get('ce_last',0):.3f} | "
                 f"{tm.get('ce_acc_last',0):.3f} | "
                 f"{'%.2f'%ov if ov is not None else 'N/A'} | {per or 'N/A'} |")
out = "\n".join(lines)
print(out)
open("/mnt/workspace/changyicheng/work/univla/rc365/round1_summary.md", "w").write(out + "\n")
