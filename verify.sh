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
required = ["numpy", "pandas", "sklearn", "scipy", "yaml"]
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
$PY main.py
echo "OK: 完整流程通过且指标落在基线容差内"

echo ""
echo "===== 阶段 6/6: pytest 回归测试 ====="
$PY -m pytest tests/ -v
echo ""
echo "全部校验通过 ✓"
