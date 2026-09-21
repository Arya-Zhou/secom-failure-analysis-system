#!/usr/bin/env bash
# 默认跑阶段 1–4（quick 冒烟）；full 追加完整流程、产物验收和回归测试。
# 任一阶段失败即停止；full 耗时取决于环境和配置。
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
required = ["numpy", "pandas", "sklearn", "scipy", "yaml", "shap", "matplotlib", "imblearn"]
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
assert Path("baseline_metrics_strict.json").exists(), "缺少 baseline_metrics_strict.json"
print("OK: 严格链路的基线指标文件存在")
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
echo "===== 阶段 5/6: 完整流程 + 基线容差比对（约 44 分钟）====="
# 清理本流程固定前缀的旧生成物，防止其令阶段 5.5/5.6 假通过（不触碰其他内容）
rm -f outputs/shap_summary_bar_*.png outputs/shap_values_*.json \
      outputs/shap_explanation_wafer_*.txt outputs/shap_contribution_wafer_*.png \
      outputs/shap_manifest.json \
      outputs/imbalance_comparison.json outputs/imbalance_comparison.md \
      outputs/imbalance_pr_curve_*.png outputs/imbalance_confusion_*.png \
      outputs/imbalance_operating_point_*.pkl \
      outputs/ablation_comparison.json outputs/ablation_comparison.md \
      outputs/ablation_grid.png outputs/temporal_metrics.json \
      outputs/cost_sensitivity.json outputs/cost_sensitivity.md \
      outputs/drift_report.json outputs/drift_report.md outputs/drift_overview.png
$PY main.py
echo "OK: 完整流程通过且指标落在基线容差内"

echo ""
echo "===== 阶段 5.4/6: 主流程选择口径验收 ====="
# 按落盘产物把"谁是 reference 模型"重算一遍：只信 CV 折外代价，不信产物里的结论字段。
$PY - <<'EOF'
import json, re, sys
from pathlib import Path

import yaml

sys.path.insert(0, ".")
from src.config import canonical_costs
from src.pipeline import SELECTION_BASIS, SELECTION_KEY

out = Path("outputs")
mfiles = sorted(out.glob("metrics_full_*.json"))
if not mfiles:
    sys.exit("未找到 metrics_full_*.json")
run_tag = re.search(r"full_\d{8}_\d{6}", mfiles[-1].name).group(0)
data = json.loads(mfiles[-1].read_text(encoding="utf-8"))
sel = data.get(SELECTION_KEY)
if not sel:
    sys.exit(f"metrics 产物缺少 {SELECTION_KEY} 段：选择依据不可审计")

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
fn_cost, fp_cost = canonical_costs(cfg)
models = {k: v for k, v in data.items() if not k.startswith("_")}
problems = []

if sel.get("selection_basis") != SELECTION_BASIS:
    problems.append(f"selection_basis={sel.get('selection_basis')!r}，应为 {SELECTION_BASIS!r}")
if sel.get("costs") != {"fn": fn_cost, "fp": fp_cost}:
    problems.append(f"落盘成本 {sel.get('costs')} 与 config 顶层 costs 不一致")

# 重算：reference 必须是训练侧 CV 代价最小者；代价必须与自己的折外混淆自洽
recomputed = sorted(models, key=lambda k: (models[k]["CV折外期望代价"], k))
if sel.get("reference_model") != recomputed[0]:
    problems.append(
        f"reference_model={sel.get('reference_model')!r}，但按 CV 折外代价重算应为 {recomputed[0]!r}")
if sel.get("ranking_by_cv_cost") != recomputed:
    problems.append(f"ranking_by_cv_cost 与重算结果不符：{sel.get('ranking_by_cv_cost')} vs {recomputed}")
for name, m in models.items():
    cm = m.get("CV折外混淆") or {}
    want = cm.get("fn", -1) * fn_cost + cm.get("fp", -1) * fp_cost
    if abs(m.get("CV折外期望代价", -1) - want) > 1e-9:
        problems.append(f"{name}: CV 折外代价与自己的折外混淆不自洽")

ref = sel.get("reference_model")
if not list(out.glob(f"reference_model_{ref}_{run_tag}.pkl")):
    problems.append(f"未找到 reference_model_{ref}_{run_tag}.pkl：落盘产物与选择结果不一致")
if list(out.glob(f"best_model_*_{run_tag}.pkl")):
    problems.append("本批仍产出 best_model_*：该命名已废弃（名字本身在断言'它赢过谁'）")

mf = out / "shap_manifest.json"
if mf.exists():
    shap_model = json.loads(mf.read_text(encoding="utf-8")).get("model")
    if shap_model and shap_model != ref:
        problems.append(f"SHAP 解释对象 {shap_model!r} 不是 reference 模型 {ref!r}")

if problems:
    sys.exit("；".join(problems))
agree = "一致" if sel.get("criteria_agree") else "不一致（按期望代价为准，已落盘）"
print(f"OK: reference={ref}（{SELECTION_BASIS}，CV {sel.get('cv_folds')} 折），"
      f"两判据排序{agree}")
EOF

echo ""
echo "===== 阶段 5.5/6: SHAP 产物条件验收（基于 manifest）====="
# 规则：generated 须两文件齐全且自洽、skipped 须有原因、failed 或无 manifest 即失败；
# 与 explain.required 抛错构成双保险，required=false 的降级失败也在此现形。
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
    sys.exit(f"SHAP 阶段状态为 {mf.get('status')!r}（应为 'ok'）：验证失败")

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
            print(f"{case}: 跳过（{info['reason']}）：有条件通过")
        else:
            problems.append(f"{case} 跳过但未记录原因")
    else:
        problems.append(f"{case} 状态异常: {status!r}")

if problems:
    sys.exit("；".join(problems))
print("OK: SHAP 产物条件验收通过")
EOF

echo ""
echo "===== 阶段 5.6/6: 不平衡策略对比产物条件验收 ====="
# 规则：启用时 json 须 status=ok、策略与图文产物齐全；禁用时须留 disabled 记录；
# 缺 json 或 status=failed 即失败。
$PY - <<'EOF'
import json, sys
from pathlib import Path
import yaml

sys.path.insert(0, ".")
from src.evaluation import SCORE_SPACE_CALIBRATED
from src.imbalance import CALIBRATED_CONTRAST, CALIBRATED_STRATEGIES

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cmp_cfg = (cfg.get("imbalance") or {}).get("comparison") or {}
out = Path("outputs")
jp = out / "imbalance_comparison.json"
if not jp.exists():
    sys.exit("缺少 outputs/imbalance_comparison.json（对比阶段未运行或未落盘）")
res = json.loads(jp.read_text(encoding="utf-8"))

if not cmp_cfg.get("enabled", True):
    if res.get("status") == "disabled":
        print("有条件通过：不平衡对比已禁用（config imbalance.comparison.enabled=false）")
        sys.exit(0)
    sys.exit(f"config 已禁用但 json 状态为 {res.get('status')!r}（应为 'disabled'）：产物过期")
if res.get("status") != "ok":
    sys.exit(f"不平衡对比状态为 {res.get('status')!r}（应为 'ok'）：验证失败")

problems = []
expected_strategies = list(cmp_cfg.get("strategies", []))
if res.get("selection_basis") != "train_cv_expected_cost":
    problems.append("selection_basis 应为 train_cv_expected_cost（策略选择必须只用训练侧信息）")
if not res.get("models"):
    problems.append("json 中无任何模型结果")
# 运行环境留痕：只作审计与排障，不参与放行判断；须记到包版本之下一层（libc、线程池）
env = res.get("runtime_env") or {}
missing_env = [k for k in ("python", "numpy", "sklearn", "imblearn", "platform",
                           "libc", "threadpools")
               if not env.get(k)]
if missing_env:
    problems.append(f"结果 json 未记录运行环境项: {missing_env}")
for display, m in (res.get("models") or {}).items():
    got = list((m.get("strategies") or {}))
    missing = [s for s in expected_strategies if s not in got]
    if missing:
        problems.append(f"{display} 缺少策略结果: {missing}")
    best = m.get("best_by_cv_cost")
    if best not in got:
        problems.append(f"{display} best_by_cv_cost={best!r} 不在策略结果中")
    # 选择池按产物**重算**一遍：不采信 selection_pool 字段，也不采信 best
    pool = m.get("selection_pool")
    recomputed_pool = [s for s, r in (m.get("strategies") or {}).items()
                       if r.get("selection_eligible")]
    if pool != recomputed_pool:
        problems.append(f"{display} selection_pool={pool} 与按 selection_eligible "
                        f"重算的 {recomputed_pool} 不符")
    if CALIBRATED_CONTRAST in (pool or []):
        problems.append(f"{display} 同表对照进了选择池：对照行只报数字，不产交付物")
    if recomputed_pool:
        want = min(recomputed_pool,
                   key=lambda s: (m["strategies"][s]["cv"]["CV期望代价"], s))
        if best != want:
            problems.append(
                f"{display} best_by_cv_cost={best!r}，但按池内 CV 代价重算应为 {want!r}")
    else:
        problems.append(f"{display} 选择池为空")
    # 落盘的"不可选原因"必须与诊断结论一致，不许写一个说法、按另一个说法选
    for strat, r in (m.get("strategies") or {}).items():
        diag = r.get("calibration_diagnostics")
        if (strat in CALIBRATED_STRATEGIES) != (diag is not None):
            problems.append(f"{display}/{strat} 校准诊断的有无与策略类型不符")
        if diag is None:
            continue
        if diag.get("passed") != (not (diag.get("structural_problems")
                                       or diag.get("quality_problems"))):
            problems.append(f"{display}/{strat} 诊断 passed 与问题列表自相矛盾")
        if strat != CALIBRATED_CONTRAST and r.get("selection_eligible") != diag["passed"]:
            problems.append(
                f"{display}/{strat} selection_eligible={r.get('selection_eligible')} "
                f"与诊断 passed={diag['passed']} 不符")
    dep = m.get("deployment")
    if not isinstance(dep, dict) or "supported_in_main" not in dep:
        problems.append(f"{display} 缺少 deployment 部署块")
    elif not dep.get("supported_in_main") and not dep.get("note"):
        problems.append(f"{display} deployment 标记不可直接部署但未记原因")
    for key in ("pr_curve", "confusion", "operating_point"):
        f = (m.get("artifacts") or {}).get(key)
        if not f or not (out / f).exists():
            problems.append(f"{display} 产物缺失: {key} -> {f}")
    if not (m.get("artifacts") or {}).get("operating_point_sha256"):
        problems.append(f"{display} 未在清单登记工作点产物的文件 SHA-256")
    for strat, r in (m.get("strategies") or {}).items():
        if "CV期望代价" not in (r.get("cv") or {}):
            problems.append(f"{display}/{strat} 缺少训练侧 CV期望代价（选择依据）")
        probe = r.get("behavior_probe") or {}
        if not all(probe.get(k) for k in ("scores_sha256", "predictions_sha256",
                                          "input_sha256", "n_rows")):
            problems.append(
                f"{display}/{strat} 行为指纹字段不全"
                "（探针输入摘要 + 分数摘要 + 最终预测摘要须齐备）")
        if strat == "threshold_moving" and r.get("threshold") is None:
            problems.append(f"{display}/{strat} 未记录所选阈值")
        # 校准行：分数空间必须与未校准概率分开，且推理图里确实有校准器那一环
        if strat in CALIBRATED_STRATEGIES:
            if r.get("score_space") != SCORE_SPACE_CALIBRATED:
                problems.append(
                    f"{display}/{strat} 分数空间为 {r.get('score_space')!r}，"
                    f"应为 {SCORE_SPACE_CALIBRATED!r}")
            if "calibrator" not in (r.get("decision_graph") or []):
                problems.append(f"{display}/{strat} 推理图不含校准器: {r.get('decision_graph')}")
            if r.get("threshold") is None:
                problems.append(f"{display}/{strat} 未记录所选阈值")
if not (out / "imbalance_comparison.md").exists():
    problems.append("缺少 outputs/imbalance_comparison.md（对比表）")

if problems:
    sys.exit("；".join(problems))
n_models = len(res["models"])
print(f"OK: 不平衡对比产物验收通过（{n_models} 个模型 × {len(expected_strategies)} 种策略）")
EOF

echo ""
echo "===== 阶段 5.7/6: 工作点产物可部署性验收 ====="
# 正例：加载各模型工作点 pkl（含清单哈希核对与产物自检），在真实测试集上复现混淆矩阵。
# 负例：在副本上注入多类篡改，校验须逐一拒绝；只验正例证明不了校验有效。
$PY - <<'EOF'
import json, pickle, shutil, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")
from src.calibration import base_pipeline, is_calibrated, positive_class
from src.data_io import load_secom
from src.evaluation import SCORE_SPACE_PROBABILITY
from src.preprocessing import drop_all_nan_columns
from src.validation import artifact_split, resolve_artifact_split
from src.imbalance import (
    load_operating_point, apply_operating_point, expected_cost, payload_digest,
    _mark_loaded)

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
out = Path("outputs")
res = json.loads((out / "imbalance_comparison.json").read_text(encoding="utf-8"))
if res.get("status") != "ok":
    print("上一阶段已判定失败，跳过")
    sys.exit(1)

root = Path(".").resolve()
X, y, _ = load_secom(
    str((root / cfg["data"]["features_path"]).resolve()),
    str((root / cfg["data"]["labels_path"]).resolve()),
    cfg["data"]["timestamp_format"])
X, _ = drop_all_nan_columns(X)
# 与主流程同一个划分口径：reference 划分 = 重采样总体的第 0 个划分
spec = resolve_artifact_split(cfg)
_, test_idx = artifact_split(y, spec["test_size"], spec["seed"])
X_te, y_te = X.loc[test_idx], y.loc[test_idx]

problems = []
loaded = {}
for display, m in res["models"].items():
    best = m["best_by_cv_cost"]
    path = out / m["artifacts"]["operating_point"]
    try:
        op = load_operating_point(path)   # 清单哈希 + 产物自检，不符即抛错
    except Exception as e:
        problems.append(f"{display} 工作点产物不可用: {e}")
        continue
    loaded[display] = (path, op)
    y_pred = apply_operating_point(op, X_te)
    _, cm = expected_cost(y_te, y_pred, res["costs"]["fn"], res["costs"]["fp"])
    recorded = m["strategies"][best]["confusion"]
    if cm != recorded:
        problems.append(f"{display} 产物复现不一致: 复现={cm} 落盘={recorded}")
    else:
        print(f"{display} / {best}: 校验通过，混淆矩阵逐位复现 {cm}")


class ShiftedCalibrator:
    """把校准器输出整体压低：模拟"校准器被换掉、基础分类器原样保留"。
    必须定义在模块顶层——局部类 pickle 不了，写副本时会当场抛 AttributeError。"""

    def __init__(self, base):
        self.base = base

    def predict(self, T):
        return np.clip(np.asarray(self.base.predict(T), dtype=float) * 0.5, 0.0, 1.0)


SELF_CHECK_REASONS = ("推理字段摘要不符", "行为指纹不符", "推理语义不符", "结构版本")


def must_reject(label, path, mutate, reasons=SELF_CHECK_REASONS, recompute_digest=False):
    """在副本上施加改动，load_operating_point 必须拒绝，且必须因为对的原因。"""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shutil.copy(out / "imbalance_comparison.json", td / "imbalance_comparison.json")
        dst = td / path.name
        with open(path, "rb") as f:
            op = pickle.load(f)
        mutate(op)
        if recompute_digest:
            op["payload_digest"] = payload_digest(op)   # 攻击者可重算字段摘要
        with open(dst, "wb") as f:
            pickle.dump(op, f)
        # 清单沿用原文件哈希，故文件哈希层先拒；再单独验产物自检层
        for manifest, layer, reasons in (
                (td / "imbalance_comparison.json", "清单文件哈希",
                 ("SHA-256 与清单不符",)),
                (None, "产物自检", reasons)):
            try:
                load_operating_point(dst, manifest=manifest)
            except ValueError as e:
                if not any(r in str(e) for r in reasons):
                    problems.append(
                        f"负例报错原因不对[{layer}]: {label}，期望命中 {reasons}，实际 {e}")
                continue
            problems.append(f"负例未被拦截[{layer}]: {label}")


if loaded:
    # ① 模型内部状态改变（属性清单式摘要抓不到的那类：树叶类别计数反转）
    rf = next((v for k, v in loaded.items() if "森林" in k or "forest" in k.lower()), None)
    if rf is not None:
        def flip_trees(op):
            for est in base_pipeline(op["pipeline"]).steps[-1][1].estimators_:
                v = est.tree_.value
                v[:, :, :] = v[:, :, ::-1]
        must_reject("随机森林树叶类别计数反转", rf[0], flip_trees)
    else:
        print("提示: 未找到随机森林工作点，跳过树状态负例")

    # ② 推理字段改变（阈值），链路本身一字未动
    thr = next((v for v in loaded.values() if v[1]["threshold"] is not None), None)
    if thr is not None:
        must_reject("阈值被改写", thr[0],
                    lambda op: op.__setitem__("threshold", float(op["threshold"]) + 0.3))
    else:
        print("提示: 无含阈值的工作点，跳过阈值负例")

    # ③ 探针输入被替换：行为指纹同时摘要探针输入与其输出，二者绑定
    any_op = next(iter(loaded.values()))
    must_reject("探针输入被替换", any_op[0],
                lambda op: op.__setitem__("probe_input", op["probe_input"] * 0.0 + 1.0))

    # ④ 分数之后的决策层被改：反转末端分类器的类别映射，分数一字不变而最终预测全翻，
    #    只摘要分数的自检对它无感，须由部署路径输出摘要拦下。仅无阈值工作点走 predict()。
    def flip_class_mapping(op):
        m = base_pipeline(op["pipeline"]).steps[-1][1]
        lb = getattr(m, "_label_binarizer", None)
        if lb is not None:                      # 岭分类器的类别映射在此
            lb.classes_ = np.asarray(lb.classes_)[::-1].copy()
        else:
            m.classes_ = np.asarray(m.classes_)[::-1].copy()

    noth = next((v for v in loaded.values() if v[1]["threshold"] is None), None)
    if noth is not None:
        path_n, op_n = noth
        before = apply_operating_point(op_n, X_te)
        probe_before = op_n["probe"]["scores_sha256"]
        tampered = pickle.loads(path_n.read_bytes())
        flip_class_mapping(tampered)
        # 给未经加载校验的副本补来源标记：此处要验的是改坏后真实预测确实变了，不是来源闸
        after = apply_operating_point(_mark_loaded(tampered), X_te)
        from src.imbalance import behavior_fingerprint
        fresh = behavior_fingerprint(
            tampered["pipeline"], tampered["probe_input"],
            threshold=tampered["threshold"], score_space=tampered["score_space"],
            seed=tampered["probe"]["seed"])
        if fresh["scores_sha256"] != probe_before:
            problems.append("前提不成立：类别映射反转不应改变任何分数")
        n_flip = int((after != before).sum())
        if n_flip == 0:
            problems.append("前提不成立：类别映射反转应改变真实测试集预测")
        else:
            print(f"  类别映射反转：分数摘要不变，真实测试集预测改变 {n_flip}/{len(before)} 片")
        must_reject("类别映射反转（分数不变、最终预测全翻）", path_n, flip_class_mapping)
    else:
        print("提示: 无无阈值工作点，跳过类别映射负例")

    # ⑤ 整份产物替换成另一个模型的自洽产物：只有外部清单能识别
    if len(loaded) >= 2:
        (p_a, _), (p_b, op_b) = list(loaded.values())[:2]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            shutil.copy(out / "imbalance_comparison.json", td / "imbalance_comparison.json")
            dst = td / p_a.name              # 用 A 的文件名装 B 的内容
            shutil.copy(p_b, dst)
            try:
                load_operating_point(dst, manifest=td / "imbalance_comparison.json")
                problems.append("负例未被拦截[清单文件哈希]: 整份产物被替换为另一自洽产物")
            except ValueError as e:
                if "SHA-256 与清单不符" not in str(e):
                    problems.append(f"负例报错原因不对[清单文件哈希]: 整份产物被替换，实际报 {e}")

    # ⑥ 只把链路与探针的列名一致改掉（值/列序/统计量未动）：链路按列名校验输入，真实
    #    调用会崩，而只摘要值的探针摘要会全部放行。先断言前提，再要求拒绝。
    any_path, any_op = next(iter(loaded.values()))
    orig_names = list(base_pipeline(any_op["pipeline"]).steps[0][1].feature_names_in_)

    def rename_columns(op):
        renamed = [f"renamed_{i}" for i in range(len(orig_names))]
        base_pipeline(op["pipeline"]).steps[0][1].feature_names_in_ = np.asarray(
            renamed, dtype=object)
        op["probe_input"] = pd.DataFrame(op["probe_input"].values, columns=renamed)
        op["payload_digest"] = payload_digest(op)   # 攻击者可重算字段摘要

    probe_tmp = pickle.loads(any_path.read_bytes())
    rename_columns(probe_tmp)
    _mark_loaded(probe_tmp)     # 同上：本段测的是链路的列名校验，不是来源闸
    try:
        apply_operating_point(probe_tmp, X_te[orig_names])
        problems.append("前提不成立：列名被改后真实调用本应失败")
    except ValueError as e:
        if "feature names" not in str(e):
            problems.append(f"前提不成立：列名被改后本应报列名不符，实际报的是 {e}")
        else:
            print("  列名重命名：真实调用已不可用（链路按列名校验输入）")
    must_reject("探针与链路列名被一致改写", any_path, rename_columns)

    # ⑦ 显式指定的清单存在、却没登记这份产物，不得降级告警
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        empty = td / "empty_manifest.json"
        empty.write_text(json.dumps({"models": {}}), encoding="utf-8")
        try:
            load_operating_point(any_path, manifest=empty)
            problems.append("负例未被拦截[清单严格性]: 显式清单未登记该产物却放行")
        except ValueError as e:
            if "未登记产物" not in str(e):
                problems.append(f"负例报错原因不对[清单严格性]: 显式清单未登记，实际报 {e}")
        try:
            load_operating_point(any_path, manifest=None, require_manifest=True)
            problems.append("负例未被拦截[清单严格性]: require_manifest 与 manifest=None 冲突未报错")
        except ValueError as e:
            if "冲突" not in str(e):
                problems.append(f"负例报错原因不对[清单严格性]: require/None 冲突，实际报 {e}")

    # ⑧ 自行反序列化真实产物后直接出预测：三层校验一层没跑，受控入口须在运行时按来源拒绝。
    #    该判据认对象来源，与代码写法无关；不调受控入口的取裸链路写法不在此列，属已知未覆盖。
    self_loaded = pickle.loads(any_path.read_bytes())
    try:
        apply_operating_point(self_loaded, X_te)
        problems.append("负例未被拦截[加载来源]: 自行反序列化的产物照样出了预测")
    except ValueError as e:
        if "未经 load_operating_point" not in str(e):
            problems.append(f"负例报错原因不对[加载来源]: {e}")
        else:
            print("  自行反序列化：受控入口按来源拒绝出预测")

    # ⑨ 校准层专项负例：只在有校准工作点时施加（无则如实打印跳过，不算通过）
    calibrated = [(p, o) for p, o in loaded.values() if is_calibrated(o["pipeline"])]
    if not calibrated:
        print("提示: 本批无校准类工作点，跳过校准层负例（校准行未被选为工作点）")
    else:
        cal_path, cal_op = calibrated[0]

        # 9a 替换校准器、保留基础分类器：校准方法字段仍是对的，只有行为指纹抓得到
        def swap_calibrator(op):
            cc = op["pipeline"].calibrated_classifiers_[0]
            cc.calibrators[0] = ShiftedCalibrator(cc.calibrators[0])

        must_reject("校准器被替换（基础分类器原样保留）", cal_path, swap_calibrator,
                    reasons=("行为指纹不符",))

        # 9b 改 calibration.method 记录：按对象重算即对不上
        must_reject("calibration.method 被改写", cal_path,
                    lambda op: op.__setitem__(
                        "calibration", dict(op["calibration"], method="isotonic")),
                    reasons=("推理语义不符",), recompute_digest=True)

        # 9c 调换类别顺序 / 改正类索引
        def flip_positive_index(op):
            est = op["pipeline"]
            est.classes_ = np.asarray(est.classes_)[::-1].copy()
            assert positive_class(est) == 0, "前提不成立：正类索引未被改掉"

        must_reject("正类索引被调换", cal_path, flip_positive_index,
                    reasons=("positive_class",), recompute_digest=True)

        # 9d 让决策路径绕过校准器直接读基础分数
        must_reject("决策路径绕过校准器（换成内层裸链路）", cal_path,
                    lambda op: op.__setitem__("pipeline", base_pipeline(op["pipeline"])),
                    reasons=("推理语义不符",), recompute_digest=True)

        # 9e 改 score_space：语义静默错配是加校准层后新引入的失效模式
        must_reject("score_space 被改成未校准语义", cal_path,
                    lambda op: op.__setitem__("score_space", SCORE_SPACE_PROBABILITY),
                    reasons=("推理字段摘要不符",))

        # 9f 结构版本缺失 / 不符：必填、无默认值
        must_reject("artifact_schema_version 缺失", cal_path,
                    lambda op: op.pop("artifact_schema_version"),
                    reasons=("artifact_schema_version",))
        must_reject("artifact_schema_version 不符", cal_path,
                    lambda op: op.__setitem__("artifact_schema_version", "旧版"),
                    reasons=("结构版本",), recompute_digest=True)

        # 9g 改成本档位并重算摘要：成本不进决策路径，两个内部层确实无感 —— 如实验证，
        #     再证明外部清单层能识别。这一条是三层分工的现场演示，不是冗余。
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            shutil.copy(out / "imbalance_comparison.json", td / "imbalance_comparison.json")
            dst = td / cal_path.name
            op = pickle.loads(cal_path.read_bytes())
            op["costs"] = {"fn": 99.0, "fp": 1.0}
            op["payload_digest"] = payload_digest(op)
            dst.write_bytes(pickle.dumps(op))
            try:
                load_operating_point(dst, manifest=None)
                print("  成本档位改写：内部两层如实放行（成本不进决策路径）")
            except ValueError as e:
                problems.append(f"前提不成立：改成本档位不该被内部自检拦下，实际报 {e}")
            try:
                load_operating_point(dst, manifest=td / "imbalance_comparison.json")
                problems.append("负例未被拦截[清单文件哈希]: 成本档位被改写")
            except ValueError as e:
                if "SHA-256 与清单不符" not in str(e):
                    problems.append(f"负例报错原因不对[清单文件哈希]: 成本改写，实际报 {e}")

        # 9h 探针必须跨阈值，否则阈值篡改检测是哑的（按产物重算，不采信记录字段）
        from src.calibration import probe_coverage
        from src.evaluation import score_samples
        scores, _ = score_samples(cal_op["pipeline"], cal_op["probe_input"])
        cov = probe_coverage(scores, cal_op["threshold"])
        if cov != cal_op["probe_coverage"]:
            problems.append(f"探针覆盖记录与重算不符: {cal_op['probe_coverage']} vs {cov}")
        elif cov["straddles_threshold"] is not True:
            problems.append(f"校准工作点探针未跨阈值: {cov}")
        else:
            print(f"  探针跨阈值：下方 {cov['below_threshold']} / "
                  f"上方 {cov['at_or_above_threshold']}，"
                  f"校准曲线覆盖 {cov['distinct_scores']} 个不同取值")

    if not problems:
        print("负例全部被拒绝：树状态改变 / 阈值改写 / 探针输入替换 / "
              "类别映射反转 / 整份产物替换 / 列名改写 / 清单未登记 / "
              "绕过加载自行反序列化 / 校准器替换 / 校准方法改写 / 正类索引调换 / "
              "绕过校准器 / score_space 改写 / 结构版本缺失与不符 / 成本档位改写")

if problems:
    sys.exit("；".join(problems))
print("OK: 工作点产物可部署性验收通过")
EOF

echo ""
echo "===== 阶段 5.8/6: 消融实验产物条件验收 ====="
# 规则：启用时 json 须 status=ok、四格齐全、effects 与格内指标自洽、图文产物齐备；
# 严格模式下还要求消融的对照格与主流程指标逐位一致（证明两者确实是同一条链路）。
$PY - <<'EOF'
import json, sys
from pathlib import Path
import yaml

sys.path.insert(0, ".")
from src.ablation import SELECTORS, WEIGHTINGS, cell_key
from src.modeling import registry_names

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
ab_cfg = cfg.get("ablation") or {}
out = Path("outputs")
jp = out / "ablation_comparison.json"
if not jp.exists():
    sys.exit("缺少 outputs/ablation_comparison.json（消融阶段未运行或未落盘）")
res = json.loads(jp.read_text(encoding="utf-8"))

if not ab_cfg.get("enabled", True):
    if res.get("status") == "disabled":
        print("有条件通过：消融实验已禁用（config ablation.enabled=false）")
        sys.exit(0)
    sys.exit(f"config 已禁用但 json 状态为 {res.get('status')!r}（应为 'disabled'）：产物过期")
if res.get("status") != "ok":
    sys.exit(f"消融实验状态为 {res.get('status')!r}（应为 'ok'）：验证失败")

problems = []
expected_models = (registry_names() if ab_cfg.get("models", "all") == "all"
                   else list(ab_cfg["models"]))
expected_cells = {cell_key(s, w) for s in SELECTORS for w in WEIGHTINGS}
missing_models = [m for m in expected_models if m not in (res.get("models") or {})]
if missing_models:
    problems.append(f"缺少模型结果: {missing_models}")

for name, m in (res.get("models") or {}).items():
    cells = m.get("cells") or {}
    if set(cells) != expected_cells:
        problems.append(f"{name} 消融格不全: 期望 {sorted(expected_cells)} 实际 {sorted(cells)}")
        continue
    for ck, c in cells.items():
        if c.get("n_features") != res["n_features"]:
            problems.append(f"{name}/{ck} 特征数 {c.get('n_features')} 与配置不符")
        cm = c.get("confusion") or {}
        if set(cm) != {"tn", "fp", "fn", "tp"}:
            problems.append(f"{name}/{ck} 混淆矩阵字段不全")
            continue
        want = cm["fn"] * res["costs"]["fn"] + cm["fp"] * res["costs"]["fp"]
        if abs(c.get("expected_cost", -1) - want) > 1e-9:
            problems.append(f"{name}/{ck} 期望代价与混淆矩阵不符")
    # effects 必须由对应两格算出，不能是另行统计的数字
    for key, (a, b) in {
        "weighting_on_vote": (("vote", "none"), ("vote", "class_weight")),
        "selector_on_none": (("vote", "none"), ("shap", "none")),
        "selector_on_weighted": (("vote", "class_weight"), ("shap", "class_weight")),
    }.items():
        eff = (m.get("effects") or {}).get(key)
        if not eff:
            problems.append(f"{name} 缺少 effects.{key}")
            continue
        ca, cb = cells[cell_key(*a)], cells[cell_key(*b)]
        if abs(eff["BER变化"] - (cb["测试集BER"] - ca["测试集BER"])) > 1e-9:
            problems.append(f"{name}/{key} BER变化 与两格差值不符")
        if eff["漏检变化"] != cb["confusion"]["fn"] - ca["confusion"]["fn"]:
            problems.append(f"{name}/{key} 漏检变化 与两格差值不符")
    # SHAP 引导那两格的探针信息：全维训练 + 记录 class_weight，缺了就无从判读排名来源
    probes = m.get("shap_probe") or {}
    for w in WEIGHTINGS:
        p = probes.get(w)
        if not p:
            problems.append(f"{name} 缺少 {w} 档的 SHAP 探针记录")
            continue
        if not p.get("shap_method"):
            problems.append(f"{name}/{w} 探针未记录 SHAP 方法")
        if p.get("probe_class_weight") != (None if w == "none" else "balanced"):
            problems.append(f"{name}/{w} 探针 class_weight 与本格不一致")
    if not isinstance(m.get("feature_overlap"), dict):
        problems.append(f"{name} 未记录两种特征选择的重合数")

for key, fname in (("grid_png", None), ("markdown", "ablation_comparison.md")):
    f = (res.get("artifacts") or {}).get(key)
    if not f or not (out / f).exists():
        problems.append(f"消融产物缺失: {key} -> {f}")
    elif fname and f != fname:
        problems.append(f"消融产物文件名意外: {key} -> {f}")

strategy = cfg["imbalance"]["strategy"]
main_cell = cell_key("vote", "class_weight" if strategy == "class_weight" else "none")

# 同链路交叉核对：消融的对照格应与主流程指标逐位一致
ck = main_cell
mfiles = sorted(out.glob("metrics_full_*.json"))
if not mfiles:
    problems.append("未找到 metrics_full_*.json，无法交叉核对主流程与消融链路")
else:
    main_metrics = json.loads(mfiles[-1].read_text(encoding="utf-8"))
    for name, m in (res.get("models") or {}).items():
        display = m.get("display_name")
        if display not in main_metrics:
            problems.append(f"主流程指标中无 {display}，无法交叉核对")
            continue
        cell = m["cells"][ck]
        for metric in ("测试集BER", "召回率", "精确率", "F1分数", "准确率"):
            d = abs(cell[metric] - main_metrics[display][metric])
            if d > 1e-9:
                problems.append(
                    f"{display} 主流程与消融 {ck} 的 {metric} 不一致（差 {d:.3e}）："
                    "两者本应是同一条严格链路")
    if not problems:
        print(f"  交叉核对通过：主流程指标与消融 {ck} 格逐位一致")

if problems:
    sys.exit("；".join(problems))
n_models = len(res["models"])
print(f"OK: 消融产物验收通过（{n_models} 个模型 × {len(expected_cells)} 个格）")
EOF

echo ""
echo "===== 阶段 5.9/6: 重采样区间产物条件验收 ====="
# 重采样默认关闭（R=20 是小时级开销），故本阶段是条件验收：产物在就严格校，
# 不在就显式报"未跑"而不是静默放行——落盘的区间数字是 README 的对外口径。
$PY - <<'EOF'
import json, sys
from pathlib import Path

sys.path.insert(0, ".")
from src.pipeline import HEADLINE_METRICS, INTERVAL_METRICS, RESAMPLE_FILE
from src.validation import INTERVAL_LOW_PCT, INTERVAL_HIGH_PCT

path = Path("outputs") / RESAMPLE_FILE
if not path.exists():
    print(f"SKIP: 无 outputs/{RESAMPLE_FILE}（validation.resample.enabled=false）。"
          "README 若引用区间数字，须先跑一次开启重采样的完整流程")
    sys.exit(0)

res = json.loads(path.read_text(encoding="utf-8"))
problems = []
if res.get("status") != "ok":
    problems.append(f"status={res.get('status')!r}")
if (res.get("protocol") or {}).get("quick"):
    problems.append("区间产物出自 quick 子采样，不能作为对外口径")

ss = res.get("same_source_assertion") or {}
if not ss.get("ok") or ss.get("max_abs_diff") != 0:
    problems.append(f"同源断言未通过: {ss.get('mismatches') or ss}")

n = len(res.get("metrics_by_split") or [])
if n != (res.get("protocol") or {}).get("n_splits"):
    problems.append(f"折级记录 {n} 条，与协议声明的 n_splits 不符")

# reference 百分位。硬闸门只加在 reference 模型的头条指标上：reference 自己是分布的成员，
# 逐格设闸门等于要求它在 27 个 cell 上都不是最好也不是最差，必然误报。其余格只告警。
ref = res.get("reference_model")
for model, pcts in (res.get("reference_percentile") or {}).items():
    for metric, pct in pcts.items():
        if pct is None:
            continue
        if INTERVAL_LOW_PCT <= pct <= INTERVAL_HIGH_PCT:
            continue
        msg = (f"{model}.{metric} 的 reference_percentile={pct:.1f} 落在 "
               f"[{INTERVAL_LOW_PCT}, {INTERVAL_HIGH_PCT}] 之外")
        if model == ref and metric in HEADLINE_METRICS:
            problems.append(
                msg + "：这是 README 引用的头条数字，要么 reference 划分不典型（须在 README "
                "声明它不具代表性），要么两条路径 recipe 不同（bug）")
        else:
            print(f"  警告：{msg}（非头条格，记录备查）")

for model, iv in (res.get("intervals") or {}).items():
    for metric in INTERVAL_METRICS:
        if metric not in iv:
            problems.append(f"{model} 缺少 {metric} 的区间")
            continue
        got = iv[metric]
        if got is None:      # 如岭分类器无 Brier：允许为空，但必须是"整列都算不出"
            continue
        if not got["low"] <= got["point"] <= got["high"]:
            problems.append(f"{model}.{metric} 点估计落在区间之外: {got}")
        if got["n"] != n:
            problems.append(f"{model}.{metric} 区间只用了 {got['n']}/{n} 个划分")

if not res.get("paired_diff"):
    problems.append("缺少配对差值区间（validation.resample.paired_diff）")

# 陈旧性：重采样产物不随阶段 5 重跑（小时级开销），故须对着本轮 metrics 核一遍，
# 否则上一版代码留下的区间会一路绿灯混进 README。
mfiles = sorted(Path("outputs").glob("metrics_full_*.json"))
if not mfiles:
    problems.append("未找到 metrics_full_*.json，无法核对重采样产物是否陈旧")
else:
    cur = json.loads(mfiles[-1].read_text(encoding="utf-8"))
    for model, point in (res.get("reference_point") or {}).items():
        if model not in cur:
            problems.append(f"重采样产物含模型 {model}，本轮主流程没有：产物已陈旧")
            continue
        for metric, val in point.items():
            now = cur[model].get(metric)
            if val is None or now is None:
                if val is not now:
                    problems.append(f"{model}.{metric}: 重采样={val!r} 本轮={now!r}")
                continue
            # 容差 1e-9 而非 0：这是**跨运行**比对，两次 main.py 是两个进程，BLAS 归约次序
            # 可让末位差 ~1e-13（见 ARCHITECTURE D11）。同源断言是进程内比对，仍是逐位 0。
            if abs(float(val) - float(now)) > 1e-9:
                problems.append(
                    f"{model}.{metric}: 重采样产物记的 reference 点估计 {val!r} "
                    f"与本轮主流程 {now!r} 不同——产物出自旧代码或旧配置，须重跑")

if problems:
    sys.exit("；".join(problems))
best = res["intervals"][ref]["测试集BER"]
print(f"OK: 重采样 {n} 个划分，同源断言逐位通过；reference={ref} 测试集BER "
      f"{best['point']:.3f} [{best['low']:.3f}, {best['high']:.3f}]，"
      f"配对差值 {len(res['paired_diff'])} 组")
EOF

echo ""
echo "===== 阶段 5.10/6: 时间序协议产物验收 ====="
# 时间序默认开启，故这里是无条件验收：产物必须存在、必须出自本轮、必须与随机协议并列可比。
$PY - <<'EOF'
import json, sys
from pathlib import Path

sys.path.insert(0, ".")
from src.pipeline import (
    INTERVAL_METRICS, RESAMPLE_FILE, SELECTION_KEY, TEMPORAL_FILE,
    temporal_escalation,
)

path = Path("outputs") / TEMPORAL_FILE
if not path.exists():
    sys.exit(f"缺少 outputs/{TEMPORAL_FILE}：时间序协议默认开启，产物不该缺失"
             "（若确系有意关闭，须同时把 README 的时间序数字撤掉）")

res = json.loads(path.read_text(encoding="utf-8"))
proto = res.get("protocol") or {}
problems = []
if res.get("status") != "ok":
    problems.append(f"status={res.get('status')!r}")
if proto.get("quick"):
    problems.append("时间序产物出自 quick 子采样，不能作为对外口径")
if proto.get("n_test_positive", 0) < 1 or proto.get("n_train_positive", 0) < 1:
    problems.append("训练段或测试段没有失效样本：本协议此时本应显式失败")
if proto.get("train_span", ["", ""])[1] > proto.get("test_span", ["", ""])[0]:
    problems.append("训练段末尾晚于测试段开头：时间顺序没生效")

drift = res.get("prior_drift") or {}
if sum(b["failures"] for b in drift.get("bins") or []) != drift.get("failures"):
    problems.append("先验漂移表的分箱失败数之和与总数不符")
if sum(b["n"] for b in drift.get("bins") or []) != drift.get("n"):
    problems.append("先验漂移表的分箱样本数之和与总数不符")

# 时间序指标与主流程 metrics 出自同一轮：这里只核"是不是同一批模型"，
# 指标本身按定义不同（不同划分），不能拿去逐位比。
mfiles = sorted(Path("outputs").glob("metrics_full_*.json"))
if not mfiles:
    problems.append("未找到 metrics_full_*.json，无法核对时间序产物是否出自本轮")
else:
    cur = json.loads(mfiles[-1].read_text(encoding="utf-8"))
    models_now = {k for k in cur if not k.startswith("_")}
    if set(res.get("models") or {}) != models_now:
        problems.append(f"时间序产物的模型集合 {set(res.get('models') or {})} "
                        f"与本轮主流程 {models_now} 不同：产物已陈旧")
for name, m in (res.get("models") or {}).items():
    missing = [k for k in INTERVAL_METRICS if k not in m]
    if missing:
        problems.append(f"{name} 缺指标 {missing}")

cmp_ = res.get("random_protocol_comparison")
if cmp_ is None:
    print(f"  未与随机协议对照：{res.get('comparison_unavailable_reason')}"
          f"（跑一次 validation.resample.enabled=true 即可产出 {RESAMPLE_FILE}）")
else:
    if cmp_.get("interval_source") not in ("in_run", "artifact"):
        problems.append(f"对照区间来源不明: {cmp_.get('interval_source')!r}")
    # 用 metrics._selection 中的主流程 reference 重算升级规则，不采信产物自带结论。
    # 时间序协议可能另选 reference；误用它会让检查在两者恰好均触发时空转。
    recorded = res.get("escalation") or {}
    main_ref = None
    if mfiles:
        main_ref = (json.loads(mfiles[-1].read_text(encoding="utf-8"))
                    .get(SELECTION_KEY) or {}).get("reference_model")
    if main_ref is None:
        problems.append("取不到主流程 reference 模型，升级规则无法独立重算")
    elif recorded.get("evaluated_on") != main_ref:
        problems.append(
            f"escalation.evaluated_on={recorded.get('evaluated_on')!r} 与主流程 reference "
            f"{main_ref!r} 不符：判据判的不是 README 头条那个模型")
    recomputed = temporal_escalation(cmp_.get("models") or {}, main_ref)
    if bool(recorded.get("required")) != recomputed["required"]:
        problems.append(f"escalation.required={recorded.get('required')!r} 与按产物重算的 "
                        f"{recomputed['required']!r} 不符")
    if recomputed["required"]:
        print("  升级规则触发（README 须把时间序结果作为主要风险结论）：")
        for r in recomputed["reasons"]:
            print(f"    - {r}")

if problems:
    sys.exit("；".join(problems))
ref = res.get("reference_model")
m = res["models"][ref]
print(f"OK: 时间序协议验收通过（测试段 {proto['n_test']} 片 / 失效 "
      f"{proto['n_test_positive']} 片，先验最大最小比 "
      f"{drift.get('max_over_min_ratio'):.1f} 倍）；reference={ref} "
      f"BER={m['测试集BER']:.3f} 召回={m['召回率']:.3f}")
EOF

echo ""
echo "===== 阶段 5.11/6: 折外校准与成本敏感性产物验收 ====="
# 规则：校准两行齐全、正确性诊断按产物重算、主表与敏感性表的 canonical 档逐位对齐；
# 校准行进了选择池就必须诊断全过。诊断结论一律重算，不采信产物里的 passed 字段。
$PY - <<'EOF'
import json, sys
from math import isfinite
from pathlib import Path
import yaml

sys.path.insert(0, ".")
from src.calibration import AUC_TOLERANCE, cost_grid, quality_problems, structural_problems
from src.config import canonical_costs
from src.evaluation import SCORE_SPACE_CALIBRATED
from src.imbalance import (
    CALIBRATED_CONTRAST, CALIBRATED_PRIMARY, CALIBRATED_STRATEGIES,
    COST_SENSITIVITY_JSON, COST_SENSITIVITY_MD,
)

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cmp_cfg = (cfg.get("imbalance") or {}).get("comparison") or {}
wanted = [s for s in (cmp_cfg.get("strategies") or []) if s in CALIBRATED_STRATEGIES]
out = Path("outputs")
res = json.loads((out / "imbalance_comparison.json").read_text(encoding="utf-8"))
problems = []

if not wanted:
    print("有条件通过：config 未启用任何校准策略（imbalance.comparison.strategies）")
    sys.exit(0)
if res.get("status") != "ok":
    sys.exit(f"不平衡对比状态为 {res.get('status')!r}，校准验收无从谈起")

fn_cost, fp_cost = canonical_costs(cfg)
cal_cfg = (cfg.get("imbalance") or {}).get("calibration") or {}
for display, m in (res.get("models") or {}).items():
    strategies = m.get("strategies") or {}
    for strat in wanted:
        r = strategies.get(strat)
        if r is None:
            problems.append(f"{display} 缺少校准策略结果: {strat}")
            continue
        diag = r.get("calibration_diagnostics") or {}
        if not diag:
            problems.append(f"{display}/{strat} 缺少校准正确性诊断")
            continue
        # 按落盘的诊断量**重算**结论，不采信 passed / *_problems 三个字段
        recomputed = structural_problems(diag) + quality_problems(diag)
        recorded = list(diag.get("structural_problems") or []) + \
            list(diag.get("quality_problems") or [])
        if bool(recomputed) != bool(recorded):
            problems.append(
                f"{display}/{strat} 诊断结论与按诊断量重算不符: 重算 {recomputed} vs 落盘 {recorded}")
        if diag.get("passed") != (not recomputed):
            problems.append(f"{display}/{strat} passed 字段与重算结论不符")
        # 单调性与值域是接线正确性，任何情况下都必须成立
        for key in ("monotonic_non_decreasing", "in_unit_interval", "finite", "shape_ok"):
            if not diag.get(key):
                problems.append(f"{display}/{strat} 结构性诊断 {key} 未通过")
        if not diag.get("ties_introduced") and abs(diag.get("auc_delta") or 0) > AUC_TOLERANCE:
            problems.append(f"{display}/{strat} 未并出并列却改变了 AUC")
        if r.get("threshold_selected_on") != "train_oof":
            problems.append(f"{display}/{strat} 阈值来源记为 {r.get('threshold_selected_on')!r}，"
                            "应为 train_oof（完全折外的校准概率）")
        if r.get("score_space") != SCORE_SPACE_CALIBRATED:
            problems.append(f"{display}/{strat} 分数空间不是 {SCORE_SPACE_CALIBRATED}")
        method = (r.get("calibration") or {}).get("method")
        want_method = (cal_cfg.get("method") if strat == CALIBRATED_PRIMARY
                       else cal_cfg.get("contrast_method"))
        if method != want_method:
            problems.append(f"{display}/{strat} 校准方法 {method!r} 与 config {want_method!r} 不符")
    # 进了选择池的校准行必须诊断全过（"全过才可标为可部署"）
    for strat in m.get("selection_pool") or []:
        diag = (strategies.get(strat) or {}).get("calibration_diagnostics")
        if diag is not None and not diag.get("passed"):
            problems.append(f"{display}/{strat} 诊断未过却进了选择池")
    # 保留主口径、额外加入对照会不会改选择：结论须与候选池的 CV 代价自洽
    c = m.get("calibration_contrast")
    if CALIBRATED_PRIMARY in wanted and CALIBRATED_CONTRAST in wanted:
        if not c:
            problems.append(f"{display} 缺少 calibration_contrast 结论段")
        else:
            pool = list(m.get("selection_pool") or [])
            want = min(pool + [CALIBRATED_CONTRAST],
                       key=lambda s: (strategies[s]["cv"]["CV期望代价"], s))
            if c.get("selected_with_contrast_in_pool") != want:
                problems.append(f"{display} 对照入池后的选择结论与重算不符")
            if c.get("would_change_selection") != (want != m.get("best_by_cv_cost")):
                problems.append(f"{display} would_change_selection 与重算不符")

# 成本敏感性：主表与敏感性表的 canonical 档必须逐位一致
def check_cost_fields(actual, expected, label, reference="重算结果"):
    if not isinstance(actual, dict):
        problems.append(f"{label} 必须为对象")
        return
    for field, value in expected.items():
        if field not in actual or actual[field] != value:
            problems.append(f"{label} {field} 与{reference}不符")


csp = out / COST_SENSITIVITY_JSON
cost_changes = {}
if CALIBRATED_PRIMARY not in wanted:
    print(f"提示: 未启用 {CALIBRATED_PRIMARY}，跳过成本敏感性验收")
elif not csp.exists():
    problems.append(f"缺少 outputs/{COST_SENSITIVITY_JSON}")
elif not (out / COST_SENSITIVITY_MD).exists():
    problems.append(f"缺少 outputs/{COST_SENSITIVITY_MD}")
else:
    cs = json.loads(csp.read_text(encoding="utf-8"))
    want_ratios = [float(x) for x in ((cfg.get("imbalance") or {}).get("cost_sensitivity") or [])]
    want_grid = cost_grid(want_ratios, fp_cost, fn_cost)
    if want_ratios != sorted(set(want_ratios)):
        problems.append("config 成本档位必须升序且唯一")
    if sum(entry["is_canonical"] for entry in want_grid) != 1:
        problems.append("config 成本档位中 canonical 档不唯一")
    check_cost_fields(cs, {"status": "ok", "canonical": {"fn": fn_cost, "fp": fp_cost},
                           "grid": want_grid}, "成本敏感性", "config 及预期状态")
    main_models = res.get("models") or {}
    cost_models = cs.get("models")
    if not isinstance(cost_models, dict):
        cost_models = {}
    if not cost_models or set(cost_models) != set(main_models):
        problems.append("成本敏感性模型集合与主表不符")
    n_test, n_positive = res["test_set"]["n"], res["test_set"]["n_fail"]
    n_negative = n_test - n_positive
    for display, main_model in main_models.items():
        if display not in cost_models:
            continue
        model = cost_models[display]
        if not isinstance(model, dict):
            problems.append(f"{display} 成本敏感性模型记录必须为对象")
            continue
        main = (main_model.get("strategies") or {}).get(CALIBRATED_PRIMARY) or {}
        check_cost_fields(model, {
            "strategy": CALIBRATED_PRIMARY,
            "calibration_method": (main.get("calibration") or {}).get("method"),
            "score_space": main.get("score_space"),
            "n_test": n_test, "n_test_positive": n_positive,
        }, display, "主表")
        rows = model.get("rows")
        if (not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
                or [row.get("ratio") for row in rows] != want_ratios):
            problems.append(f"{display} 成本档位必须完整、升序且唯一，并与 config 一致")
            continue
        if sum(row.get("is_canonical") is True for row in rows) != 1:
            problems.append(f"{display} 成本敏感性缺少唯一的 canonical 行")
        test_confusions = []
        for row, entry in zip(rows, want_grid):
            label = f"{display} 成本档位 {entry['ratio']:g}"
            check_cost_fields(row, {"fn_cost": entry["fn"], "fp_cost": entry["fp"]},
                              label, "config")
            if row.get("is_canonical") is not entry["is_canonical"]:
                problems.append(f"{label} canonical 标记与 config 不符")
            test_counts, train_counts = row.get("test"), row.get("train_oof")
            valid_counts = True
            for counts, keys, name in ((test_counts, ("tn", "fp", "fn", "tp"), "test"),
                                       (train_counts, ("fn", "fp"), "train_oof")):
                if (not isinstance(counts, dict)
                        or any(type(counts.get(key)) is not int or counts[key] < 0
                               for key in keys)):
                    problems.append(f"{label} {name} 计数必须为非负整数")
                    valid_counts = False
            if not valid_counts:
                continue
            if (test_counts["fn"] + test_counts["tp"] != n_positive
                    or test_counts["tn"] + test_counts["fp"] != n_negative):
                problems.append(f"{label} 测试混淆计数与主表测试集规模不符")
            test_cost = test_counts["fn"] * entry["fn"] + test_counts["fp"] * entry["fp"]
            trivial_cost = n_positive * entry["fn"]
            recall = test_counts["tp"] / n_positive if n_positive else 0.0
            ber = ((test_counts["fn"] / n_positive if n_positive else 0.0)
                   + (test_counts["fp"] / n_negative if n_negative else 0.0)) / 2
            check_cost_fields(test_counts, {"召回率": recall, "BER": ber}, label + " test")
            check_cost_fields(train_counts, {
                "cost": train_counts["fn"] * entry["fn"] + train_counts["fp"] * entry["fp"],
            }, label + " train_oof")
            check_cost_fields(row, {
                "test_cost": test_cost,
                "cost_per_wafer": test_cost / n_test if n_test else None,
                "trivial_all_pass_cost": trivial_cost,
                "improvement_vs_all_pass": ((trivial_cost - test_cost) / trivial_cost
                                            if trivial_cost else None),
            }, label)
            test_confusions.append((test_counts["fn"], test_counts["fp"]))
            if entry["is_canonical"]:
                selection = main.get("threshold_selection") or {}
                check_cost_fields(row, {
                    "threshold": main.get("threshold"),
                    "train_oof": {"fn": selection.get("train_fn"),
                                  "fp": selection.get("train_fp"),
                                  "cost": selection.get("train_cost")},
                    "test_cost": main.get("expected_cost"),
                }, f"{display} canonical", "主表")
                check_cost_fields(test_counts, {
                    **(main.get("confusion") or {}),
                    "召回率": (main.get("test") or {}).get("召回率"),
                    "BER": (main.get("test") or {}).get("BER"),
                }, f"{display} canonical test", "主表")
        if len(test_confusions) == len(rows):
            check_cost_fields(model, {
                "changes_test_confusion": len(set(test_confusions)) > 1,
            }, display)
        thresholds = [row.get("threshold") for row in rows]
        if any(type(threshold) not in (int, float) or not isfinite(threshold)
               for threshold in thresholds):
            problems.append(f"{display} 成本档位阈值必须为有限数值")
            continue
        if thresholds != sorted(thresholds, reverse=True):
            problems.append(
                f"{display} 阈值未随 FN:FP 上升而下降（{thresholds}）：漏检更贵时应更早报警")
        n_distinct = len({round(threshold, 12) for threshold in thresholds})
        cost_changes[display] = n_distinct > 1
        check_cost_fields(model, {"distinct_thresholds": n_distinct,
                                  "changes_operating_point": cost_changes[display]}, display)
    check_cost_fields(res.get("cost_sensitivity"), {
        "artifact": COST_SENSITIVITY_JSON, "markdown": COST_SENSITIVITY_MD,
        "ratios": want_ratios, "changes_operating_point": cost_changes,
    }, "主表成本敏感性摘要", "成本敏感性表")

if problems:
    sys.exit("；".join(problems))
lines = []
for display, m in (res.get("models") or {}).items():
    r = (m.get("strategies") or {}).get(CALIBRATED_PRIMARY) or {}
    d = r.get("calibration_diagnostics") or {}
    lines.append(f"  {display}: 校准诊断{'通过' if d.get('passed') else '未过'}"
                 f"（折外 Brier 改善 {d.get('oof_brier_improvement', float('nan')):+.6f}）"
                 f"｜工作点={m.get('best_by_cv_cost')}"
                 f"｜换成本比改工作点="
                 f"{cost_changes.get(display)}")
print("OK: 折外校准与成本敏感性产物验收通过")
print("\n".join(lines))
EOF

echo ""
echo "===== 阶段 5.12/6: 训练侧稳定性与参数候选产物验收 ====="
$PY - <<'EOF'
from pathlib import Path
import sys

from src.config import load_config
from src.data_io import load_secom
from src.feature_selection import load_feature_override
from src.stability_analysis import STABILITY_DETAIL_FILE, validate_stability_artifacts
from src.validation import artifact_split, resolve_artifact_split

cfg = load_config("config.yaml")
out = Path(cfg["output"]["results_dir"])
if not (out / STABILITY_DETAIL_FILE).exists():
    if cfg["stability"]["enabled"]:
        sys.exit("稳定性已启用但缺少完整分析产物")
    print("SKIP: stability.enabled=false 且尚无完整分析产物；不计为稳定性验收通过")
    sys.exit(0)
features, labels, _timestamps = load_secom(
    cfg["data"]["features_path"], cfg["data"]["labels_path"], cfg["data"]["timestamp_format"])
split = resolve_artifact_split(cfg)
reference_train, reference_test = artifact_split(labels, split["test_size"], split["seed"])
reference_files = sorted(out.glob("selected_features_full_*.txt"))
if not reference_files:
    sys.exit("缺少主流程 reference 特征清单，无法核验稳定性产物")
result = validate_stability_artifacts(
    cfg, features, labels, reference_train, reference_test, out,
    load_feature_override(reference_files[-1]))
print("OK: 独立抽样协议、训练来源及逐轮证据重算通过")
print("OK: 三档清单、全部敏感性档位、相关组、Spearman / Jaccard 与累计趋势一致")
print(result["summary"]["counts_by_tier"])
EOF

echo ""
echo "===== 阶段 5.13/6: 模型角色、训练侧配对选择与唯一交付物验收 ====="
$PY - <<'EOF'
from pathlib import Path
from src.config import load_config
from src.data_io import load_secom
from src.model_comparison_analysis import (
    COMPARISON_DETAIL_FILE, COMPARISON_FILE, COMPARISON_PLAN_FILE, COMPARISON_SELECTION_FILE,
    COMPARISON_MODEL_FILE, COMPARISON_MARKDOWN, validate_model_comparison_artifacts,
)
from src.validation import artifact_split, resolve_artifact_split

cfg = load_config("config.yaml")
out = Path(cfg["output"]["results_dir"])
artifacts = (COMPARISON_DETAIL_FILE, COMPARISON_FILE, COMPARISON_PLAN_FILE, COMPARISON_SELECTION_FILE,
             COMPARISON_MODEL_FILE, COMPARISON_MARKDOWN)
if not any((out / filename).exists() for filename in artifacts):
    if cfg["model_comparison"]["enabled"]:
        raise SystemExit("模型比较已启用但缺少完整产物")
    print("SKIP: 尚无模型比较产物，不计为模型比较验收通过")
else:
    features, labels, _timestamps = load_secom(
        cfg["data"]["features_path"], cfg["data"]["labels_path"], cfg["data"]["timestamp_format"])
    split = resolve_artifact_split(cfg)
    reference_train, reference_test = artifact_split(labels, split["test_size"], split["seed"])
    result = validate_model_comparison_artifacts(cfg, features, labels, reference_train, reference_test, out)
    print("OK: 角色隔离、训练侧来源、逐轮预测、全部配对差值与选择锁重算通过")
    print("OK: 外层指标、外部对标限定、唯一部署工作点与 SHAP 产物一致")
    print(result["summary"]["selection"])
EOF

echo ""
echo "===== 阶段 5.14/6: 时间漂移分析与严格早训晚测产物验收 ====="
$PY - <<'EOF'
from pathlib import Path
from src.config import load_config
from src.data_io import load_secom
from src.drift import resolve_drift
from src.drift_reporting import validate_drift_artifacts

config = load_config("config.yaml")
if not resolve_drift(config)["enabled"]:
    print("SKIP: drift.enabled=false，不计为时间漂移分析验收通过")
else:
    features, labels, timestamps = load_secom(
        config["data"]["features_path"], config["data"]["labels_path"], config["data"]["timestamp_format"])
    result = validate_drift_artifacts(config, features, labels, timestamps, Path(config["output"]["results_dir"]))
    print("OK: 先验/月度/滑窗、全部原始特征 PSI/KS/BH 及来源重算通过")
    print("OK: 早期训练链路、边界时间戳隔离、逐行未来预测及图文重算通过")
    print(result["performance"]["overall"]["metrics"])
EOF

echo ""
echo "===== 阶段 6/6: pytest 回归测试 ====="
$PY -m pytest tests/ -v
echo ""
echo "全部校验通过 ✓"
