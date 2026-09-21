"""Gradio 演示入口：报告模式可只读快照，本机预测模式再加载工作点。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .demo_data import (
    BANNER, FROZEN_SAMPLES, GLOBAL_CHOICE, PROJECT_ROOT, QUADRANT_GLOSSARY,
)
from .demo_predict import (
    format_prediction, frozen_display_result, predict_frozen_wafer, predict_inputs_ready,
    predict_mode_enabled, predict_status_markdown, startup_predict_bundle,
)
from .demo_views import (
    candidates_markdown, candidates_view, global_shap_markdown, image_for_sample,
    overview_markdown, overview_view, report_status, sample_image_note, shap_case_view,
    temporal_markdown, temporal_panel_choices, temporal_panel_image,
    temporal_panel_markdown, temporal_view,
)

INSTALL_HINT = "未安装 Gradio。请在仓库根目录执行：pip install -r requirements-demo.txt"
STACK_HINT = (
    "当前 FastAPI/Starlette 与 Gradio 4.44.1 不兼容。"
    "请在仓库根目录执行：pip install -r requirements-demo.txt"
)


def parse_major_minor(version: str) -> tuple[int, int]:
    """解析主次版本号，供兼容性检查使用。"""
    numbers = []
    for part in version.split("."):
        if not part.isdigit():
            break
        numbers.append(int(part))
        if len(numbers) == 2:
            break
    while len(numbers) < 2:
        numbers.append(0)
    return numbers[0], numbers[1]


def starlette_incompatible(version: str) -> bool:
    """Starlette 0.45+ 调换了 TemplateResponse 参数顺序。"""
    major, minor = parse_major_minor(version)
    return major >= 1 or (major == 0 and minor >= 45)


def check_gradio_stack() -> None:
    """拒绝会触发 unhashable type: dict 的过新 Starlette。"""
    try:
        import starlette
    except ImportError as exc:
        raise SystemExit(STACK_HINT) from exc
    if starlette_incompatible(starlette.__version__):
        raise SystemExit(STACK_HINT)


def ensure_localhost_not_proxied() -> None:
    """本机健康检查必须直连 127.0.0.1，不能走 HTTP_PROXY。"""
    extras = ("127.0.0.1", "localhost", "::1")
    for key in ("NO_PROXY", "no_proxy"):
        parts = [item.strip() for item in os.environ.get(key, "").split(",") if item.strip()]
        for extra in extras:
            if extra not in parts:
                parts.append(extra)
        os.environ[key] = ",".join(parts)


def disable_gradio_analytics() -> None:
    """本机演示不向 Gradio 上报遥测，避免代理握手超时刷屏。"""
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"


def import_gradio():
    """延迟导入 Gradio，缺依赖或不兼容时给出安装提示。"""
    disable_gradio_analytics()
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit(INSTALL_HINT) from exc
    check_gradio_stack()
    return gr


def sample_choices() -> list[str]:
    """冻结样本与全局 SHAP 下拉项。"""
    return [f"{row['case']} {row['wafer_id']}" for row in FROZEN_SAMPLES] + [GLOBAL_CHOICE]


def parse_choice(label: str) -> int:
    """从下拉标签解析行号。"""
    return int(label.rsplit(" ", 1)[1])


def build_interface(predict_bundle=None, demo_dir: Path | None = None):
    """构建四页签界面；不在导入时启动服务。"""
    gr = import_gradio()
    report = report_status(demo_dir) if demo_dir is not None else report_status()
    if not report["ready"]:
        missing = "、".join(report["missing"])
        raise SystemExit(f"缺少演示快照：{missing}。报告模式需要 docs/demo/ 中的已跟踪文件。")

    overview = overview_view(demo_dir) if demo_dir is not None else overview_view()
    candidates = candidates_view(demo_dir) if demo_dir is not None else candidates_view()
    temporal = temporal_view(demo_dir) if demo_dir is not None else temporal_view()
    shap_case = shap_case_view(demo_dir) if demo_dir is not None else shap_case_view()
    status = predict_inputs_ready()
    enabled = predict_bundle is not None and predict_mode_enabled(status)
    case_choices = sample_choices()
    panel_choices = temporal_panel_choices()

    def show_sample(choice: str):
        image = image_for_sample(choice, shap_case)
        note = sample_image_note(choice)
        if choice == GLOBAL_CHOICE:
            return global_shap_markdown(), note, image
        wafer_id = parse_choice(choice)
        if enabled:
            result = predict_frozen_wafer(
                predict_bundle["op"], predict_bundle["X"], predict_bundle["y"], wafer_id,
            )
        else:
            result = frozen_display_result(wafer_id)
        return format_prediction(result), note, image

    def show_temporal_panel(choice: str):
        return temporal_panel_markdown(choice), temporal_panel_image(temporal, choice)

    with gr.Blocks(title="SECOM 离线失效分析演示", analytics_enabled=False) as demo:
        gr.Markdown(f"# SECOM 离线失效分析演示\n{BANNER}")
        with gr.Tabs():
            with gr.Tab("总览"):
                gr.Markdown(overview_markdown(overview))
            with gr.Tab("候选清单"):
                gr.Markdown(candidates_markdown(candidates))
            with gr.Tab("成本策略工作点"):
                if not enabled:
                    gr.Markdown(predict_status_markdown(status))
                gr.Markdown(QUADRANT_GLOSSARY)
                first_text, first_note, first_image = show_sample(case_choices[0])
                choice = gr.Dropdown(case_choices, label="样本", value=case_choices[0])
                output = gr.Markdown(first_text)
                image_note = gr.Markdown(first_note)
                image = gr.Image(value=first_image, label="对应图片")
                choice.change(
                    show_sample, inputs=choice, outputs=[output, image_note, image],
                    api_name=False,
                )
            with gr.Tab("时间风险"):
                gr.Markdown(temporal_markdown(temporal))
                first_panel_text, first_panel_image = show_temporal_panel(panel_choices[0])
                panel = gr.Dropdown(panel_choices, label="漂移图", value=panel_choices[0])
                panel_text = gr.Markdown(first_panel_text)
                panel_image = gr.Image(value=first_panel_image, label="对应图片")
                panel.change(
                    show_temporal_panel, inputs=panel, outputs=[panel_text, panel_image],
                    api_name=False,
                )
    return demo


def main(argv: list[str] | None = None) -> int:
    """命令行入口：先判断报告模式，再按需加载工作点。"""
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        print("本演示不接受额外参数。用法：python -m src.demo_app", file=sys.stderr)
        return 2
    report = report_status()
    if not report["ready"]:
        missing = "、".join(report["missing"])
        print(f"缺少演示快照：{missing}。期望路径：{PROJECT_ROOT / 'docs' / 'demo'}", file=sys.stderr)
        return 1
    bundle = startup_predict_bundle()
    demo = build_interface(bundle)
    ensure_localhost_not_proxied()
    demo.launch(share=False, server_name="127.0.0.1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
