# -*- coding: utf-8 -*-
"""
ui.py: 負責 Gradio 介面的 UI 佈局、元件定義。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import gradio as gr
from config import (
    BACKEND_CLOUD,
    BACKEND_LOCAL,
    CLOUD_API_KEY,
    CLOUD_BASE_URL,
    CLOUD_MODEL,
    DEFAULT_THINK,
    MODE_CHAT,
    MODE_CODE_FULL,
    MODE_CODE_PATCH,
    NUM_CTX,
)


@dataclass
class UIComponents:
    """封裝所有 Gradio UI 元件實例以供事件綁定。"""
    # 狀態管理
    candidate_code_state: gr.State
    syntax_err_state: gr.State
    staged_chart_state: gr.State

    # 左側欄：檔案管理元件
    file_path_box: gr.Textbox
    start_line_box: gr.Number
    end_line_box: gr.Number
    load_btn: gr.Button
    clear_file_btn: gr.Button
    file_status_md: gr.Markdown
    code_preview: gr.Code

    # 左側欄：模型與聯網設定
    backend_radio: gr.Radio
    mode_radio: gr.Radio
    think_cb: gr.Checkbox
    web_cb: gr.Checkbox
    tech_web_cb: gr.Checkbox
    cloud_url_box: gr.Textbox
    cloud_key_box: gr.Textbox
    cloud_model_box: gr.Textbox

    # 右側欄：對話區元件
    chatbot: gr.Chatbot
    msg_box: gr.Textbox
    send_btn: gr.Button
    submit_analysis_btn: gr.Button
    stop_btn: gr.Button
    clear_chat_btn: gr.Button

    # 右側欄：Diff 與安全寫入控制
    apply_status_md: gr.Markdown
    apply_btn: gr.Button
    autofix_btn: gr.Button
    restore_btn: gr.Button
    diff_preview: gr.Code

    def get_gen_inputs(self) -> List[gr.components.Component]:
        """返回核心推理生成時所需的所有輸入元件列表。"""
        return [
            self.chatbot,
            self.file_path_box,
            self.start_line_box,
            self.end_line_box,
            self.mode_radio,
            self.think_cb,
            self.web_cb,
            self.tech_web_cb,
            self.backend_radio,
            self.cloud_url_box,
            self.cloud_key_box,
            self.cloud_model_box,
            self.staged_chart_state,
        ]

    def get_gen_outputs(self) -> List[gr.components.Component]:
        """返回核心推理生成時所需更新的所有輸出元件列表。"""
        return [
            self.chatbot,
            self.code_preview,
            self.file_status_md,
            self.diff_preview,
            self.apply_status_md,
            self.apply_btn,
            self.autofix_btn,
            self.candidate_code_state,
            self.syntax_err_state,
        ]


def build_ui() -> Tuple[gr.Blocks, UIComponents]:
    """建立 Gradio 介面佈局並回傳 (demo, components)。"""
    with gr.Blocks(title="Qwen3.5 本地/云端双模代码工作台") as demo:
        candidate_code_state = gr.State("")
        syntax_err_state = gr.State("")
        staged_chart_state = gr.State("")

        gr.Markdown(
            f"## Qwen3.5 本地 / 云端双模代码工作台\n"
            f"本地上下文 **{NUM_CTX}** tokens（8G 显存安全线） | 支持一键切换云端 API | "
            "内置 Exa 高光联网 + AST 语法拦截 + Diff 对比 + `.workbench_bak` 自动备份回滚"
        )

        with gr.Row():
            # 左側欄：源碼掛載與雙模控台
            with gr.Column(scale=5):
                gr.Markdown("### 1. 本地源码挂载")
                file_path_box = gr.Textbox(
                    label="目标文件路径（支持相对路径或 Windows 复制文件地址）",
                    placeholder="例如：app.py 或 D:\\project\\main.py",
                )
                with gr.Row():
                    start_line_box = gr.Number(
                        label="起始行（0=从头）", value=0, precision=0, minimum=0
                    )
                    end_line_box = gr.Number(
                        label="结束行（0=到尾）", value=0, precision=0, minimum=0
                    )
                with gr.Row():
                    load_btn = gr.Button("读取 / 刷新源码", variant="secondary")
                    clear_file_btn = gr.Button("卸载文件")

                file_status_md = gr.Markdown(
                    "📄 **未挂载文件**（可直接对话，或输入本地文件路径直读源码）"
                )
                code_preview = gr.Code(
                    label="当前挂载源码预览（只读，带行号）",
                    language=None,
                    interactive=False,
                    lines=14,
                )

                gr.Markdown("### 2. 推理引擎与联网控台")
                backend_radio = gr.Radio(
                    choices=[BACKEND_LOCAL, BACKEND_CLOUD],
                    value=BACKEND_LOCAL,
                    label="推理引擎（本地零成本 / 云端强推理热切换）",
                )
                mode_radio = gr.Radio(
                    choices=[MODE_CHAT, MODE_CODE_PATCH, MODE_CODE_FULL],
                    value=MODE_CHAT,
                    label="工作模式",
                )
                with gr.Row():
                    think_cb = gr.Checkbox(label="本地深度思考 (Think)", value=DEFAULT_THINK)
                    web_cb = gr.Checkbox(label="开启 Exa 联网搜索", value=False)
                    tech_web_cb = gr.Checkbox(label="仅搜技术文档 (2025+)", value=False)

                with gr.Accordion("云端 API 参数设置（默认读取 .env 中的 CLOUD_*）", open=False):
                    cloud_url_box = gr.Textbox(
                        label="API Base URL",
                        value=CLOUD_BASE_URL,
                        placeholder="https://api.deepseek.com/v1",
                    )
                    cloud_key_box = gr.Textbox(
                        label="API Key",
                        value=CLOUD_API_KEY,
                        type="password",
                        placeholder="sk-...",
                    )
                    cloud_model_box = gr.Textbox(
                        label="云端模型名称",
                        value=CLOUD_MODEL,
                        placeholder="deepseek-chat / deepseek-reasoner",
                    )

            # 右側欄：對話區 + 安全寫入控制台
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    label="对话与代码生成区",
                    height=480,
                )
                msg_box = gr.Textbox(
                    placeholder="输入你的代码修改需求、生辰/四柱（先排盘暂存）或粘贴技术文档链接...",
                    lines=2,
                    show_label=False,
                )
                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary", scale=2)
                    submit_analysis_btn = gr.Button(
                        "提交分析", variant="secondary", scale=1, interactive=False
                    )
                    stop_btn = gr.Button("停止生成", variant="stop", scale=1)
                    clear_chat_btn = gr.Button("清空对话", scale=1)

                gr.Markdown("### 3. 差异对比（Diff）与安全写入")
                apply_status_md = gr.Markdown(
                    "ℹ️ 挂载文件并生成代码后，此处将自动显示 AST 语法预检结果与差异对比。"
                )
                with gr.Row():
                    apply_btn = gr.Button(
                        "✅ 应用修改到本地文件（自动备份）",
                        variant="primary",
                        interactive=False,
                    )
                    autofix_btn = gr.Button(
                        "🔧 一键发送语法报错给模型修复",
                        variant="secondary",
                        interactive=False,
                    )
                    restore_btn = gr.Button(
                        "↩️ 撤销上次写入（恢复备份）",
                        variant="secondary",
                    )

                with gr.Accordion("查看 Unified Diff 差异详情", open=True):
                    diff_preview = gr.Code(
                        label="修改前后差异对比（- 为原代码，+ 为新代码）",
                        language="markdown",
                        interactive=False,
                        lines=10,
                    )

        components = UIComponents(
            candidate_code_state=candidate_code_state,
            syntax_err_state=syntax_err_state,
            staged_chart_state=staged_chart_state,
            file_path_box=file_path_box,
            start_line_box=start_line_box,
            end_line_box=end_line_box,
            load_btn=load_btn,
            clear_file_btn=clear_file_btn,
            file_status_md=file_status_md,
            code_preview=code_preview,
            backend_radio=backend_radio,
            mode_radio=mode_radio,
            think_cb=think_cb,
            web_cb=web_cb,
            tech_web_cb=tech_web_cb,
            cloud_url_box=cloud_url_box,
            cloud_key_box=cloud_key_box,
            cloud_model_box=cloud_model_box,
            chatbot=chatbot,
            msg_box=msg_box,
            send_btn=send_btn,
            submit_analysis_btn=submit_analysis_btn,
            stop_btn=stop_btn,
            clear_chat_btn=clear_chat_btn,
            apply_status_md=apply_status_md,
            apply_btn=apply_btn,
            autofix_btn=autofix_btn,
            restore_btn=restore_btn,
            diff_preview=diff_preview,
        )

    return demo, components