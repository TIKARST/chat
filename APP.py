# -*- coding: utf-8 -*-
"""
app.py: 程式總入口，負責初始化、串接 UI 與 Backend 邏輯並啟動服務。
"""

from __future__ import annotations

import sys
from typing import List, Optional

import gradio as gr
from backend import generate_response, preflight, stage_chart_for_query
from config import (
    CLOUD_MODEL,
    MODEL_NAME,
    NUM_CTX,
    OLLAMA_URL,
    SERVER_NAME,
    SERVER_PORT,
    logger,
)
from skill_manager import create_skill_router, skill_registry
from ui import UIComponents, build_ui
from utils import (
    append_user_message,
    port_available,
    prepare_autofix_prompt,
    ui_apply_patch,
    ui_clear_file,
    ui_load_file,
    ui_restore_backup,
)


def unified_submit_pipeline(
    message: str,
    history: Optional[List[dict]],
    raw_path: str,
    start_line: float,
    end_line: float,
    mode: str,
    think: bool,
    use_web: bool,
    tech_only: bool,
    backend: str,
    cloud_url: str,
    cloud_key: str,
    cloud_model: str,
    staged_chart: str,
):
    """
    統一單步流式入口：彻底消除兩步 .then() 導致的清屏閃動與丟字。
    輸出順序：(msg_box, chatbot, staged_chart_state, submit_analysis_btn, *gen_outputs)
    """
    clean_msg = (message or "").strip()
    current_history = list(history or [])

    if not clean_msg:
        yield (
            "",
            current_history,
            staged_chart,
            gr.Button(interactive=bool((staged_chart or "").strip())),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )
        return

    # 1. 檢查是否命中排盤腳本（第一批暫存）
    try:
        staged_text, tip = stage_chart_for_query(clean_msg)
    except Exception as exc:
        logger.exception("排盤預執行異常：%s", exc)
        staged_text, tip = None, ""

    if staged_text:
        current_history.append({"role": "user", "content": clean_msg})
        current_history.append(
            {"role": "assistant", "content": f"{staged_text}\n\n> {tip}"}
        )
        yield (
            "",
            current_history,
            staged_text,
            gr.Button(interactive=True),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )
        return

    # 2. 普通對話或無腳本任務：先立即上屏用戶訊息並清空輸入框
    current_history.append({"role": "user", "content": clean_msg})
    yield (
        "",
        current_history,
        "",
        gr.Button(interactive=False),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
    )

    # 3. 緊接在同一生成器內流式輸出模型回覆
    for gen_out in generate_response(
        history=current_history,
        raw_path=raw_path,
        start_line=start_line,
        end_line=end_line,
        mode=mode,
        think=think,
        use_web=use_web,
        tech_only=tech_only,
        backend=backend,
        cloud_url=cloud_url,
        cloud_key=cloud_key,
        cloud_model=cloud_model,
        staged_chart="",
    ):
        yield (
            gr.skip(),
            gen_out[0],
            "",
            gr.skip(),
            *gen_out[1:],
        )


def unified_analysis_pipeline(
    history: Optional[List[dict]],
    raw_path: str,
    start_line: float,
    end_line: float,
    mode: str,
    think: bool,
    use_web: bool,
    tech_only: bool,
    backend: str,
    cloud_url: str,
    cloud_key: str,
    cloud_model: str,
    staged_chart: str,
):
    """點擊【提交分析】的單步流式入口。"""
    current_history = list(history or [])
    if not (staged_chart or "").strip():
        yield (
            current_history,
            "",
            gr.Button(interactive=False),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )
        return

    current_history.append({"role": "user", "content": "已核对暂存盘面，请提交分析。"})
    yield (
        current_history,
        staged_chart,
        gr.Button(interactive=False),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
    )

    for gen_out in generate_response(
        history=current_history,
        raw_path=raw_path,
        start_line=start_line,
        end_line=end_line,
        mode=mode,
        think=think,
        use_web=use_web,
        tech_only=tech_only,
        backend=backend,
        cloud_url=cloud_url,
        cloud_key=cloud_key,
        cloud_model=cloud_model,
        staged_chart=staged_chart,
    ):
        yield (
            gen_out[0],
            staged_chart,
            gr.skip(),
            *gen_out[1:],
        )


def bind_events(components: UIComponents) -> None:
    """串接 UI 元件事件至後端邏輯與工具函式。"""
    components.tech_web_cb.change(
        fn=lambda checked: gr.Checkbox(value=True) if checked else gr.skip(),
        inputs=[components.tech_web_cb],
        outputs=[components.web_cb],
        queue=False,
    )

    components.load_btn.click(
        fn=ui_load_file,
        inputs=[
            components.file_path_box,
            components.start_line_box,
            components.end_line_box,
            components.backend_radio,
        ],
        outputs=[components.code_preview, components.file_status_md],
    )
    components.file_path_box.submit(
        fn=ui_load_file,
        inputs=[
            components.file_path_box,
            components.start_line_box,
            components.end_line_box,
            components.backend_radio,
        ],
        outputs=[components.code_preview, components.file_status_md],
    )
    components.clear_file_btn.click(
        fn=ui_clear_file,
        inputs=[],
        outputs=[
            components.file_path_box,
            components.start_line_box,
            components.end_line_box,
            components.code_preview,
            components.file_status_md,
            components.diff_preview,
            components.apply_status_md,
            components.apply_btn,
            components.autofix_btn,
            components.candidate_code_state,
            components.syntax_err_state,
        ],
    )

    components.apply_btn.click(
        fn=ui_apply_patch,
        inputs=[
            components.file_path_box,
            components.start_line_box,
            components.end_line_box,
            components.candidate_code_state,
            components.backend_radio,
        ],
        outputs=[
            components.code_preview,
            components.file_status_md,
            components.apply_status_md,
            components.apply_btn,
        ],
    )
    components.restore_btn.click(
        fn=ui_restore_backup,
        inputs=[
            components.file_path_box,
            components.start_line_box,
            components.end_line_box,
            components.candidate_code_state,
            components.backend_radio,
        ],
        outputs=[
            components.code_preview,
            components.file_status_md,
            components.apply_status_md,
            components.apply_btn,
        ],
    )

    gen_inputs = components.get_gen_inputs()
    gen_outputs = components.get_gen_outputs()

    # 單步流式綁定：msg_box 作為首個輸入，生成開始即自動清空，無多段 .then() 斷層
    send_ev = gr.on(
        triggers=[components.msg_box.submit, components.send_btn.click],
        fn=unified_submit_pipeline,
        inputs=[components.msg_box, *gen_inputs],
        outputs=[
            components.msg_box,
            components.chatbot,
            components.staged_chart_state,
            components.submit_analysis_btn,
            *gen_outputs[1:],
        ],
        concurrency_limit=1,
    )

    submit_analysis_ev = components.submit_analysis_btn.click(
        fn=unified_analysis_pipeline,
        inputs=gen_inputs,
        outputs=[
            components.chatbot,
            components.staged_chart_state,
            components.submit_analysis_btn,
            *gen_outputs[1:],
        ],
        concurrency_limit=1,
    )

    autofix_ev = components.autofix_btn.click(
        fn=prepare_autofix_prompt,
        inputs=[components.syntax_err_state],
        outputs=[components.msg_box],
    ).then(
        fn=append_user_message,
        inputs=[components.msg_box, components.chatbot],
        outputs=[components.msg_box, components.chatbot],
    ).then(
        fn=generate_response,
        inputs=gen_inputs,
        outputs=gen_outputs,
        concurrency_limit=1,
    )

    components.stop_btn.click(
        fn=None,
        inputs=[],
        outputs=[],
        cancels=[send_ev, submit_analysis_ev, autofix_ev],
    )
    components.clear_chat_btn.click(
        fn=lambda: (
            [],
            "",
            gr.Button(interactive=False),
            "",
            "ℹ️ 当前未挂载本地文件，无需生成写入补丁。",
            gr.Button(interactive=False),
            gr.Button(interactive=False),
            "",
            "",
        ),
        inputs=[],
        outputs=[
            components.chatbot,
            components.staged_chart_state,
            components.submit_analysis_btn,
            components.diff_preview,
            components.apply_status_md,
            components.apply_btn,
            components.autofix_btn,
            components.candidate_code_state,
            components.syntax_err_state,
        ],
    )


def main() -> None:
    """工作台啟動主流程。"""
    logger.info(
        "配置：local_model=%s endpoint=%s num_ctx=%s cloud_model=%s",
        MODEL_NAME,
        OLLAMA_URL,
        NUM_CTX,
        CLOUD_MODEL,
    )

    preflight()

    if not port_available(SERVER_NAME, SERVER_PORT):
        logger.error(
            "端口 %s:%s 已被占用，可通过 GRADIO_SERVER_PORT 指定其他端口",
            SERVER_NAME,
            SERVER_PORT,
        )
        sys.exit(1)

    loaded_skills = skill_registry.scan_skills()
    logger.info("已自動載入 %d 個 Skills: %s", loaded_skills, list(skill_registry._skills.keys()))

    demo, components = build_ui()
    with demo:
        bind_events(components)

    demo.app.include_router(create_skill_router(skill_registry))

    demo.launch(server_name=SERVER_NAME, server_port=SERVER_PORT, show_error=True)


if __name__ == "__main__":
    main()