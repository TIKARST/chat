# -*- coding: utf-8 -*-
"""
app.py: 程式總入口，負責初始化、串接 UI 與 Backend 邏輯並啟動服務。
"""

from __future__ import annotations

import sys
import gradio as gr
from config import (
    CLOUD_MODEL,
    MODEL_NAME,
    NUM_CTX,
    OLLAMA_URL,
    SERVER_NAME,
    SERVER_PORT,
    logger,
)
from utils import (
    append_user_message,
    port_available,
    prepare_autofix_prompt,
    ui_apply_patch,
    ui_clear_file,
    ui_load_file,
    ui_restore_backup,
)
from backend import generate_response, preflight
from ui import UIComponents, build_ui
from skill_manager import create_skill_router, skill_registry


def bind_events(components: UIComponents) -> None:
    """串接 UI 元件事件至後端邏輯與工具函式。"""
    # 勾選「僅搜技術文檔」時自動同步勾選「開啟 Exa 聯網搜尋」
    components.tech_web_cb.change(
        fn=lambda checked: gr.Checkbox(value=True) if checked else gr.skip(),
        inputs=[components.tech_web_cb],
        outputs=[components.web_cb],
        queue=False,
    )

    # 檔案載入與重設
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

    # 安全寫入與備份還原
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

    # 推理生成事件
    gen_inputs = components.get_gen_inputs()
    gen_outputs = components.get_gen_outputs()

    submit_ev = components.msg_box.submit(
        fn=append_user_message,
        inputs=[components.msg_box, components.chatbot],
        outputs=[components.msg_box, components.chatbot],
        queue=False,
    ).then(
        fn=generate_response,
        inputs=gen_inputs,
        outputs=gen_outputs,
        concurrency_limit=1,
    )

    click_ev = components.send_btn.click(
        fn=append_user_message,
        inputs=[components.msg_box, components.chatbot],
        outputs=[components.msg_box, components.chatbot],
        queue=False,
    ).then(
        fn=generate_response,
        inputs=gen_inputs,
        outputs=gen_outputs,
        concurrency_limit=1,
    )

    autofix_ev = components.autofix_btn.click(
        fn=prepare_autofix_prompt,
        inputs=[components.syntax_err_state],
        outputs=[components.msg_box],
        queue=False,
    ).then(
        fn=append_user_message,
        inputs=[components.msg_box, components.chatbot],
        outputs=[components.msg_box, components.chatbot],
        queue=False,
    ).then(
        fn=generate_response,
        inputs=gen_inputs,
        outputs=gen_outputs,
        concurrency_limit=1,
    )

    # 停止生成與清空對話
    components.stop_btn.click(
        fn=None,
        inputs=[],
        outputs=[],
        cancels=[submit_ev, click_ev, autofix_ev],
    )
    components.clear_chat_btn.click(
        fn=lambda: [],
        inputs=[],
        outputs=[components.chatbot],
        queue=False,
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

    # 本地模型環境預檢
    preflight()

    # 端口佔用預檢
    if not port_available(SERVER_NAME, SERVER_PORT):
        logger.error(
            "端口 %s:%s 已被占用，可通过 GRADIO_SERVER_PORT 指定其他端口",
            SERVER_NAME,
            SERVER_PORT,
        )
        sys.exit(1)

    # 自動掃描並載入 Skills
    loaded_skills = skill_registry.scan_skills()
    logger.info("已自動載入 %d 個 Skills: %s", loaded_skills, list(skill_registry._skills.keys()))

    # 建立 UI、綁定事件並啟動服務
    demo, components = build_ui()
    with demo:
        bind_events(components)

    # 掛載外部 HTTP RESTful API 路由（支援外部微服務或客戶端調用 Skills）
    demo.app.include_router(create_skill_router(skill_registry))

    demo.launch(server_name=SERVER_NAME, server_port=SERVER_PORT, show_error=True)


if __name__ == "__main__":
    main()
