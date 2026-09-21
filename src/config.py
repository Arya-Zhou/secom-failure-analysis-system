"""配置加载：模块统一用 load_config() 读 config.yaml，密钥用 load_secrets() 读 .env。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # 允许无 dotenv 时降级
    load_dotenv = None


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """读取 YAML 配置为 dict。"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def canonical_costs(cfg: dict[str, Any]) -> tuple[float, float]:
    """项目唯一的成本假设 (fn, fp)：三处选择共用；必填无默认值，缺失或旧位置残留即报错。"""
    legacy = ((cfg.get("imbalance") or {}).get("comparison") or {}).get("costs")
    if legacy is not None:
        raise ValueError(
            "imbalance.comparison.costs 已上移为顶层 costs（主流程选模型与策略对比选工作点"
            "必须用同一组成本假设）。请删除旧位置的 costs 后重试")
    costs = cfg.get("costs")
    if not isinstance(costs, dict) or costs.get("fn") is None or costs.get("fp") is None:
        raise ValueError("config 缺少顶层 costs.fn / costs.fp：成本假设必填、无默认值")
    return float(costs["fn"]), float(costs["fp"])


def load_secrets() -> dict[str, str | None]:
    """从 .env 读取密钥"""
    if load_dotenv is not None:
        load_dotenv()
    return {
        "llm_api_key": os.environ.get("LLM_API_KEY") or None,
        "llm_base_url": os.environ.get("LLM_BASE_URL") or None,
        "service_token": os.environ.get("SERVICE_TOKEN") or None,
    }
