"""S5: 证明 V1/V2 的生效配置除 vision_variant 外逐位相同."""
import json, sys
a = json.load(open("/opt/rc365_runs/round1/v1/effective_config.json"))
b = json.load(open("/opt/rc365_runs/round1/v2/effective_config.json"))
allow = {"vision_variant"}
diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
print("config diff:", json.dumps(diff, indent=2))
extra = set(diff) - allow
if extra:
    print(f"\nFAIL: 除 {allow} 外还有差异: {extra}")
    sys.exit(1)
print(f"\nPASS: 仅 {sorted(diff)} 不同 (允许)")
