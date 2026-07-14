"""可解释性 / 失效根因 —— 扩展点（基础版占位）。

面试反馈后再按需实现，届时只动本模块：
    - SHAP 全局重要性 + 单晶圆失效归因报告
    - 相关性 + 模型重要性 + SHAP 三维交叉的"根因候选链"
基础版不写任何用不到的抽象，保持模块空但边界清晰。
"""
from __future__ import annotations


def explain_global(model, X):
    """全局特征影响（扩展点）。"""
    raise NotImplementedError("扩展方向：SHAP summary/bar plot")


def explain_wafer(model, x_row):
    """单晶圆失效归因报告（扩展点）。"""
    raise NotImplementedError("扩展方向：单样本 SHAP 贡献 + 影响方向")
