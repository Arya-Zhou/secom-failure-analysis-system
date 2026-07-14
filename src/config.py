"""配置加载：让 config.yaml 成为唯一事实来源。

其他模块一律通过 load_config() 取值，不在代码里散落魔法数字。
密钥从 .env 读取（不进 config，不进 Git）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # 基础版允许无 dotenv 时降级
    load_dotenv = None


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """读取 YAML 配置为 dict。"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_secrets() -> dict[str, str | None]:
    """从 .env 读取密钥（基础版可能全为 None，属正常）。"""
    if load_dotenv is not None:
        load_dotenv()
    return {
        "llm_api_key": os.environ.get("LLM_API_KEY") or None,
        "llm_base_url": os.environ.get("LLM_BASE_URL") or None,
        "service_token": os.environ.get("SERVICE_TOKEN") or None,
    }
