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
from history_store import (
    append_message,
    delete_session,
    list_session_choices,
    load_messages,
    new_session_id,
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


def _persist_turn(session_id: str, user_text: str, latest_history: Optional[List[dict]]) -> str:
    """将本轮用户提问与最新一条助手回复追加写入本地 .jsonl 文件"""
    active_sid = session_id or new_session_id()
    if user_text:
        append_message(active_sid, "user", user_text)
    if latest_history:
        last_msg = latest_history[-1]
        if isinstance(last_msg, dict) and last_msg.get("role") == "assistant":
            assistant_text = str(last_msg.get("content") or "")
            if assistant_text:
                append_message(active_sid, "assistant", assistant_text)
    return active_sid


def _dropdown_update(active_sid: Optional[str] = None):
    """刷新历史会话下拉框选项"""
    choices = list_session_choices()
    valid_ids = {sid for _, sid in choices}
    selected = active_sid if (active_sid and active_sid in valid_ids) else None
    return gr.update(choices=choices, value=selected)


def on_new_session():
    """点击“新建会话”：生成全新 ID、清空对话框与暂存状态、刷新下拉列表"""
    fresh_id = new_session_id()
    return (
        fresh_id,
        [],
        "",
        gr.Button(interactive=False),
        _dropdown_update(None),
    )


def on_refresh_sessions(current_sid: str):
    """刷新历史会话下拉列表"""
    return _dropdown_update(current_sid)


def on_load_session(selected_sid: str, current_sid: str):
    """点击“加载”：把选中历史还原到聊天窗口，并将活跃会话切到该历史 ID"""
    if not selected_sid:
        return current_sid, gr.skip()
    msgs = load_messages(selected_sid, as_tuples=False)
    return selected_sid, msgs


def on_delete_session(selected_sid: str, current_sid: str):
    """点击“删除”：删除选中本地记录；若删的是当前会话则自动切回全新会话"""
    if not selected_sid:
        return current_sid, gr.skip(), gr.skip()
    delete_session(selected_sid)
    if selected_sid == current_sid:
        return new_session_id(), [], _dropdown_update(None)
    return current_sid, gr.skip(), _dropdown_update(current_sid)


def unified_submit_pipeline(
    message: str,
    session_id: str,
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
    統一單步流式入口：彻底消除兩步 .then() 導致的清屏閃動與丟字，並自動落盤本地聊天記錄。
    輸出順序：(msg_box, session_state, history_dropdown, chatbot, staged_chart_state, submit_analysis_btn, *gen_outputs[1:])
    """
    clean_msg = (message or "").strip()
    current_history = list(history or [])
    active_sid = session_id or new_session_id()

    if not clean_msg:
        yield (
            "",
            active_sid,
            gr.skip(),
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
        active_sid = _persist_turn(active_sid, clean_msg, current_history)
        yield (
            "",
            active_sid,
            _dropdown_update(active_sid),
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
        active_sid,
        gr.skip(),
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
    final_history = current_history
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
        final_history = gen_out[0]
        yield (
            gr.skip(),
            active_sid,
            gr.skip(),
            gen_out[0],
            "",
            gr.skip(),
            *gen_out[1:],
        )

    # 4. 流式生成完整結束後，追加落盤並刷新下拉框
    active_sid = _persist_turn(active_sid, clean_msg, final_history)
    yield (
        gr.skip(),
        active_sid,
        _dropdown_update(active_sid),
        gr.skip(),
        gr.skip(),
        gr.skip(),
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


def unified_analysis_pipeline(
    session_id: str,
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
    """點擊【提交分析】的單步流式入口，完成後同步落盤。"""
    current_history = list(history or [])
    active_sid = session_id or new_session_id()

    if not (staged_chart or "").strip():
        yield (
            active_sid,
            gr.skip(),
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

    trigger_msg = "已核对暂存盘面，请提交分析。"
    current_history.append({"role": "user", "content": trigger_msg})
    yield (
        active_sid,
        gr.skip(),
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

    final_history = current_history
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
        final_history = gen_out[0]
        yield (
            active_sid,
            gr.skip(),
            gen_out[0],
            staged_chart,
            gr.skip(),
            *gen_out[1:],
        )

    active_sid = _persist_turn(active_sid, trigger_msg, final_history)
    yield (
        active_sid,
        _dropdown_update(active_sid),
        gr.skip(),
        gr.skip(),
        gr.skip(),
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


def bind_events(
    components: UIComponents,
    session_state: gr.State,
    new_session_btn: gr.Button,
    history_dropdown: gr.Dropdown,
    load_session_btn: gr.Button,
    refresh_session_btn: gr.Button,
    delete_session_btn: gr.Button,
) -> None:
    """串接 UI 元件事件至後端邏輯與工具函式。"""
    # 歷史會話控制事件
    new_session_btn.click(
        fn=on_new_session,
        inputs=[],
        outputs=[
            session_state,
            components.chatbot,
            components.staged_chart_state,
            components.submit_analysis_btn,
            history_dropdown,
        ],
    )
    refresh_session_btn.click(
        fn=on_refresh_sessions,
        inputs=[session_state],
        outputs=[history_dropdown],
    )
    load_session_btn.click(
        fn=on_load_session,
        inputs=[history_dropdown, session_state],
        outputs=[session_state, components.chatbot],
    )
    delete_session_btn.click(
        fn=on_delete_session,
        inputs=[history_dropdown, session_state],
        outputs=[session_state, components.chatbot, history_dropdown],
    )

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
        inputs=[components.msg_box, session_state, *gen_inputs],
        outputs=[
            components.msg_box,
            session_state,
            history_dropdown,
            components.chatbot,
            components.staged_chart_state,
            components.submit_analysis_btn,
            *gen_outputs[1:],
        ],
        concurrency_limit=1,
    )

    submit_analysis_ev = components.submit_analysis_btn.click(
        fn=unified_analysis_pipeline,
        inputs=[session_state, *gen_inputs],
        outputs=[
            session_state,
            history_dropdown,
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

    # 「清空对话」：封存当前会话到历史列表，开启新会话（绝不删除本地历史文件）
    components.clear_chat_btn.click(
        fn=lambda: (
            new_session_id(),
            gr.update(choices=list_session_choices(), value=None),
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
            session_state,
            history_dropdown,
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
        # 每次啟動網頁或重新整理頁面時，都會初始化一個全新的 session_id
        session_state = gr.State(value=new_session_id)

        with gr.Accordion("历史会话记录（点击展开加载 / 管理本地存档）", open=False):
            with gr.Row():
                new_session_btn = gr.Button("＋ 新建会话", size="sm", scale=1)
                history_dropdown = gr.Dropdown(
                    label="选择本地历史会话",
                    choices=list_session_choices(),
                    value=None,
                    interactive=True,
                    scale=5,
                )
                load_session_btn = gr.Button("加载", size="sm", scale=1)
                refresh_session_btn = gr.Button("刷新", size="sm", scale=1)
                delete_session_btn = gr.Button("删除", size="sm", variant="stop", scale=1)

        bind_events(
            components=components,
            session_state=session_state,
            new_session_btn=new_session_btn,
            history_dropdown=history_dropdown,
            load_session_btn=load_session_btn,
            refresh_session_btn=refresh_session_btn,
            delete_session_btn=delete_session_btn,
        )

    demo.app.include_router(create_skill_router(skill_registry))

    demo.launch(server_name=SERVER_NAME, server_port=SERVER_PORT, show_error=True)


if __name__ == "__main__":
    main()