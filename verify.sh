#!/usr/bin/env bash
# ============================================================
# verify.sh —— 重构工程的手动校验脚本（按阶段依次执行）
# 用法:
#   bash verify.sh          # 依次跑阶段 1-4（quick 冒烟为止）
#   bash verify.sh full     # 额外跑阶段 5-6（完整流程 + 回归测试，约 1-3 分钟）
# 任一阶段失败即停止，退出码非 0。
# ============================================================
set -e
cd "$(dirname "$0")"

# 若存在本地虚拟环境则优先使用
if [ -x "vvsecom/bin/python" ]; then
    PY="vvsecom/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi
echo "使用解释器: $($PY --version 2>&1) ($PY)"

echo ""
echo "===== 阶段 1/6: 语法编译检查 ====="
$PY -m py_compile main.py src/*.py tests/*.py
echo "OK: 所有 .py 编译通过"

echo ""
echo "===== 阶段 2/6: 依赖检查 ====="
$PY - <<'EOF'
import importlib.util, sys
required = ["numpy", "pandas", "sklearn", "scipy", "yaml", "shap", "matplotlib"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    print(f"缺少依赖: {missing}")
    print("请先创建虚拟环境并安装:")
    print("  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt")
    sys.exit(1)
import numpy, pandas, sklearn
print(f"OK: numpy={numpy.__version__} pandas={pandas.__version__} sklearn={sklearn.__version__}")
EOF

echo ""
echo "===== 阶段 3/6: 配置与数据文件检查 ====="
$PY - <<'EOF'
from pathlib import Path
import yaml
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
root = Path(".").resolve()
for key in ("features_path", "labels_path"):
    p = (root / cfg["data"][key]).resolve()
    assert p.exists(), f"数据文件不存在: {p}（从 UCI 下载后放到 Secom/ 根目录）"
    print(f"OK: {key} -> {p}")
assert Path("baseline_metrics.json").exists(), "缺少 baseline_metrics.json"
print("OK: baseline_metrics.json 存在")
EOF

echo ""
echo "===== 阶段 4/6: quick 冒烟运行（小样本、跳过 RFE，秒级）====="
$PY main.py --quick
echo "OK: quick 冒烟通过"

if [ "$1" != "full" ]; then
    echo ""
    echo "冒烟校验全部通过。完整校验请运行: bash verify.sh full"
    exit 0
fi

echo ""
echo "===== 阶段 5/6: 完整流程 + 基线容差比对（约 1-3 分钟）====="
# 先清理固定命名的 SHAP 生成物，防止旧产物令阶段 5.5 假通过
# （rm 范围严格限于 shap_ 前缀生成文件，不触碰其他内容）
rm -f outputs/shap_summary_bar_*.png outputs/shap_values_*.json \
      outputs/shap_explanation_wafer_*.txt outputs/shap_contribution_wafer_*.png \
      outputs/shap_manifest.json
$PY main.py
echo "OK: 完整流程通过且指标落在基线容差内"

echo ""
echo "===== 阶段 5.5/6: SHAP 产物条件验收（基于 manifest）====="
# 与 explain.required 抛错构成双保险；required=false 的降级失败也在此现形。
# 规则：generated 必须两文件齐全且自洽；skipped 必须有原因；failed/无 manifest 即失败。
$PY - <<'EOF'
import json, sys
from pathlib import Path

out = Path("outputs")
mf_path = out / "shap_manifest.json"
if not mf_path.exists():
    sys.exit("缺少 outputs/shap_manifest.json（SHAP 阶段未运行或未落盘）")
mf = json.loads(mf_path.read_text(encoding="utf-8"))

if not mf.get("enabled", True):
    print(f"有条件通过：explain 已禁用（{mf.get('reason', '未记录原因')}）")
    sys.exit(0)
if mf.get("status") != "ok":
    sys.exit(f"SHAP 阶段状态为 {mf.get('status')!r}（应为 'ok'）——验证失败")

problems = []
g = mf.get("global") or {}
for key in ("png", "json"):
    f = g.get(key)
    if not f or not (out / f).exists():
        problems.append(f"全局产物缺失: {key} -> {f}")

for case, info in (mf.get("cases") or {}).items():
    status = info.get("status")
    if status == "generated":
        for key in ("report", "plot"):
            f = info.get(key)
            if not f or not (out / f).exists():
                problems.append(f"{case} 记录为 generated 但文件缺失: {f}")
        dev = info.get("deviation")
        dev_txt = f"{dev:.2e}" if isinstance(dev, (int, float)) else str(dev)
        if not info.get("consistency_ok"):
            problems.append(f"{case} 自洽校验未通过 (deviation={dev_txt})")
        else:
            print(f"{case}: 已生成（wafer {info.get('wafer_id')}，自洽偏差 {dev_txt}）")
    elif status == "skipped":
        if info.get("reason"):
            print(f"{case}: 跳过（{info['reason']}）——有条件通过")
        else:
            problems.append(f"{case} 跳过但未记录原因")
    else:
        problems.append(f"{case} 状态异常: {status!r}")

if problems:
    sys.exit("；".join(problems))
print("OK: SHAP 产物条件验收通过")
EOF

echo ""
echo "===== 阶段 6/6: pytest 回归测试 ====="
$PY -m pytest tests/ -v
echo ""
echo "全部校验通过 ✓"
