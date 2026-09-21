#!/usr/bin/env python3
"""清理 outputs/ 里过期的时间戳批次产物。默认干跑，加 --apply 才真删。

保留 full/quick 各最新 N 批与 docs/demo 清单引用的那批（三者并集）；无时间戳产物不在管辖内。

用法：
    python tools/prune_outputs.py              # 干跑，打印待删清单
    python tools/prune_outputs.py --apply      # 执行删除
    python tools/prune_outputs.py --keep 3     # 每种模式各保留最新 3 批
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DEMO_MANIFEST = ROOT / "docs" / "demo" / "shap_manifest.json"

# 批次标签形如 full_<8位日期>_<6位时刻>，嵌在文件名中间或末尾
TAG_RE = re.compile(r"(full|quick)_(\d{8}_\d{6})")


def scan_batches(outputs: Path) -> dict[str, list[Path]]:
    """按批次标签归拢 outputs/ 下的带时间戳文件；无时间戳文件不返回。"""
    batches: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(outputs.iterdir()):
        if not p.is_file():
            continue
        m = TAG_RE.search(p.name)
        if m:
            batches[m.group(0)].append(p)
    return batches


def demo_pinned_tags(manifest: Path) -> set[str]:
    """docs/demo 清单引用的批次必须保住；SHAP 只在 full 模式产出，故 run_tag 对应 full 批次。"""
    if not manifest.exists():
        return set()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    run_tag = data.get("run_tag")
    return {f"full_{run_tag}"} if run_tag else set()


def plan(keep: int) -> tuple[dict[str, list[Path]], set[str], set[str]]:
    batches = scan_batches(OUTPUTS)
    pinned = demo_pinned_tags(DEMO_MANIFEST)

    keep_tags: set[str] = set(t for t in pinned if t in batches)
    for mode in ("full", "quick"):
        # 标签内嵌的是零填充时间戳，字典序即时间序
        tags = sorted(t for t in batches if t.startswith(f"{mode}_"))
        keep_tags.update(tags[-keep:] if keep > 0 else [])
    return batches, keep_tags, pinned


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="真正删除（缺省为干跑）")
    ap.add_argument("--keep", type=int, default=2, help="每种模式各保留最新几批（默认 2）")
    args = ap.parse_args()

    if not OUTPUTS.is_dir():
        print(f"outputs 目录不存在: {OUTPUTS}")
        return 1

    batches, keep_tags, pinned = plan(args.keep)
    if not batches:
        print("outputs/ 下没有带时间戳的批次产物，无事可做。")
        return 0

    drop_tags = sorted(set(batches) - keep_tags)
    drop_files = [p for t in drop_tags for p in batches[t]]
    n_untagged = sum(1 for p in OUTPUTS.iterdir()
                     if p.is_file() and not TAG_RE.search(p.name))

    print(f"outputs/           {OUTPUTS}")
    print(f"批次总数           {len(batches)}（无时间戳产物 {n_untagged} 个，不在管辖内）")
    print(f"保留规则           full/quick 各最新 {args.keep} 批 + demo 清单引用批")
    print(f"demo 清单引用      {', '.join(sorted(pinned)) or '（无）'}")
    print(f"\n保留 {len(keep_tags)} 批：")
    for t in sorted(keep_tags):
        why = []
        if t in pinned:
            why.append("demo 引用")
        why.append(f"{t.split('_')[0]} 最新 {args.keep} 批之一")
        print(f"  + {t:26s} {len(batches[t])} 文件   （{' / '.join(why)}）")
    print(f"\n删除 {len(drop_tags)} 批 / {len(drop_files)} 文件：")
    for t in drop_tags:
        print(f"  - {t:26s} {len(batches[t])} 文件")

    if not args.apply:
        print("\n[干跑] 未删除任何文件。确认无误后加 --apply 执行。")
        return 0

    for p in drop_files:
        p.unlink()
    print(f"\n已删除 {len(drop_files)} 个文件。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
