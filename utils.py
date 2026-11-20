# -*- coding: utf-8 -*-
"""
utils.py: 處理檔案 I/O、自動備份（.workbench_bak）與通用工具函式。
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import socket
from datetime import datetime
from typing import Any, List, Optional, Tuple

import gradio as gr
from config import (
    BASE_DIR,
    BACKEND_CLOUD,
    CLOUD_PROMPT_CHARS,
    CONNECT_TIMEOUT,
    EXT_LANG_MAP,
    LOCAL_PROMPT_CHARS,
    MAX_MESSAGE_CHARS,
    READ_TIMEOUT,
    logger,
)

# ---------------------------------------------------------------------------
# 正則表達式與常用樣式
# ---------------------------------------------------------------------------
DETAILS_PATTERN = re.compile(r"<details(?:\s+open)?>.*?</details>\s*", re.DOTALL)
CODE_BLOCK_PATTERN = re.compile(r"```[a-zA-Z0-9_+-]*\r?\n(.*?)```", re.DOTALL)
LINE_PREFIX_PATTERN = re.compile(r"^\s*\d+\s*\|\s?")
URL_PATTERN = re.compile(r"https?://[^\s<>\"'\]\)]+")
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
SIMILAR_KEYWORDS = ("类似", "替代", "竞品", "平替", "同类型", "similar", "alternative")


# ---------------------------------------------------------------------------
# 檔案路徑與語言識別
# ---------------------------------------------------------------------------
def resolve_file_path(raw_path: str) -> Optional[str]:
    """解析使用者輸入的檔案路徑（支援相對路徑與絕對路徑）。"""
    cleaned = (raw_path or "").strip().strip('"').strip("'")
    if not cleaned:
        return None
    if os.path.isfile(cleaned):
        return os.path.abspath(cleaned)
    candidate = os.path.join(BASE_DIR, cleaned)
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)
    return None


def detect_language(path: str) -> Optional[str]:
    """根據副檔名識別程式語言類型。"""
    _, ext = os.path.splitext(path.lower())
    return EXT_LANG_MAP.get(ext)


def read_text_file(resolved_path: str) -> str:
    """自動處理 UTF-8 與 GBK 編碼讀取文字檔案。"""
    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(resolved_path, "r", encoding="gbk", errors="replace") as f:
            return f.read()


def compute_line_bounds(
    total_lines: int, start_line: Optional[float], end_line: Optional[float]
) -> Tuple[int, int]:
    """計算切片起止行邊界（1-based 索引）。"""
    s_idx = int(start_line or 0)
    e_idx = int(end_line or 0)
    actual_start = max(1, s_idx) if s_idx > 0 else 1
    actual_end = min(total_lines, e_idx) if e_idx > 0 else total_lines
    if actual_start > actual_end:
        return 1, total_lines
    return actual_start, actual_end


def load_source_slice(
    raw_path: str,
    start_line: Optional[float] = 0,
    end_line: Optional[float] = 0,
    budget_chars: int = LOCAL_PROMPT_CHARS,
) -> Tuple[str, str, str, Optional[str]]:
    """
    讀取目標檔案並按行號切片，返回：
    (numbered_code, plain_code, status_markdown, detected_language)
    """
    cleaned = (raw_path or "").strip().strip('"').strip("'")
    if not cleaned:
        return "", "", "📄 **未挂载文件**（可直接对话，或输入本地文件路径直读源码）", None

    resolved = resolve_file_path(cleaned)
    if not resolved:
        return "", "", f"❌ **找不到文件**：`{cleaned}`（请检查路径是否正确）", None

    try:
        raw_text = read_text_file(resolved)
    except Exception as exc:
        return "", "", f"❌ **读取失败**：`{exc}`", None

    all_lines = raw_text.splitlines()
    total_lines = len(all_lines)
    if total_lines == 0:
        return "", "", f"⚠️ **空文件**：`{os.path.basename(resolved)}`（0 行）", detect_language(resolved)

    actual_start, actual_end = compute_line_bounds(total_lines, start_line, end_line)
    selected = all_lines[actual_start - 1 : actual_end]
    width = len(str(actual_end))
    numbered_lines = [
        f"{line_no:>{width}} | {line_text}"
        for line_no, line_text in enumerate(selected, start=actual_start)
    ]
    numbered_code = "\n".join(numbered_lines)
    plain_code = "\n".join(selected)

    char_count = len(numbered_code)
    budget_ratio = int((char_count / max(1, budget_chars)) * 100)
    fname = os.path.basename(resolved)
    range_desc = f"第 {actual_start}~{actual_end} 行 / 共 {total_lines} 行"

    if budget_ratio > 80:
        status = (
            f"⚠️ **已挂载**：`{fname}`（{range_desc}，约 {char_count} 字符，占当前引擎预算 **{budget_ratio}%**）\n"
            f"> 提示：若使用本地 9B 模型，建议设置【起始行 / 结束行】截取局部修改区，或切换到云端 API。"
        )
    else:
        status = (
            f"✅ **已挂载**：`{fname}`（{range_desc}，约 {char_count} 字符，占当前引擎预算 **{budget_ratio}%**）"
        )

    return numbered_code, plain_code, status, detect_language(resolved)


# ---------------------------------------------------------------------------
# 檔案備份與寫入底層支援（.workbench_bak）
# ---------------------------------------------------------------------------
def backup_and_write_file(resolved_path: str, candidate_text: str) -> Tuple[bool, str, Optional[str]]:
    """
    安全備份當前檔案到同級目錄下的 .workbench_bak/，並寫入新代碼。
    返回 (is_success, message, bak_filename)
    """
    try:
        bak_dir = os.path.join(os.path.dirname(resolved_path), ".workbench_bak")
        os.makedirs(bak_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.basename(resolved_path)
        bak_path = os.path.join(bak_dir, f"{fname}.{ts}.bak")
        shutil.copy2(resolved_path, bak_path)

        with open(resolved_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(candidate_text)

        return True, "寫入成功", os.path.basename(bak_path)
    except Exception as exc:
        logger.exception("檔案寫入失敗: %s", exc)
        return False, str(exc), None


def restore_latest_backup(resolved_path: str) -> Tuple[bool, str, Optional[str]]:
    """
    從 .workbench_bak/ 找出最新歷史備份還原，並刪除該備份以保持備份棧乾淨。
    返回 (is_success, message, restored_bak_filename)
    """
    try:
        bak_dir = os.path.join(os.path.dirname(resolved_path), ".workbench_bak")
        fname = os.path.basename(resolved_path)
        pattern = os.path.join(bak_dir, f"{fname}.*.bak")
        backups = sorted(glob.glob(pattern))
        if not backups:
            return False, f"未找到 `{fname}` 的歷史備份記錄。", None

        latest_bak = backups[-1]
        shutil.copy2(latest_bak, resolved_path)
        os.remove(latest_bak)
        return True, "還原成功", os.path.basename(latest_bak)
    except Exception as exc:
        logger.exception("還原備份失敗: %s", exc)
        return False, str(exc), None


# ---------------------------------------------------------------------------
# 網絡端口可用性檢查
# ---------------------------------------------------------------------------
def port_available(host: str, port: int) -> bool:
    """檢查指定 host 與 port 是否可用。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


# ---------------------------------------------------------------------------
# Gradio 訊息提示與文字/代碼提取輔助
# ---------------------------------------------------------------------------
def notify(kind: str, text: str) -> None:
    """在 UI 彈出 Toast 提示，若非 Gradio 執行階段則寫入日誌。"""
    try:
        (gr.Warning if kind == "warning" else gr.Info)(text)
    except Exception:
        logger.info("[%s] %s", kind, text)


def render_response(answer: List[str], reasoning: List[str]) -> str:
    """將思考過程與回答正文組裝成 Markdown（包含摺疊思考標籤）。"""
    text = "".join(answer)
    if reasoning:
        thought_text = "".join(reasoning)
        if not text:
            return (
                f"<details open><summary>🧠 正在深度思考中（已生成 {len(thought_text)} 字）...</summary>\n\n"
                + thought_text
                + "\n\n</details>"
            )
        return (
            f"<details><summary>🧠 思考过程（共 {len(thought_text)} 字，点击展开）</summary>\n\n"
            + thought_text
            + "\n\n</details>\n\n"
            + text
        )
    return text


def friendly_error(exc: Exception, endpoint_label: str) -> str:
    """轉換常見網絡異常為友善提示。"""
    import requests
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return f"连接 {endpoint_label} 超时（{CONNECT_TIMEOUT:g}s），请检查网络或服务状态。"
    if isinstance(exc, requests.exceptions.ReadTimeout):
        return f"等待 {endpoint_label} 响应超时（{READ_TIMEOUT:g}s），可调大 OLLAMA_READ_TIMEOUT 后重试。"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return f"无法连接 {endpoint_label}，请确认服务地址与网络可用。"
    return f"{type(exc).__name__}: {str(exc)[:160]}"


def extract_text_from_content(content: Any, strip_thinking: bool = False) -> str:
    """從 Gradio 訊息物件或字串中提純純文字內容。"""
    raw_text = ""
    if isinstance(content, str):
        raw_text = content
    elif isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                txt = block.get("text")
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt.strip())
        raw_text = "\n".join(parts)

    if strip_thinking and raw_text:
        raw_text = DETAILS_PATTERN.sub("", raw_text)
    return raw_text.strip()


def extract_last_code_block(assistant_markdown: str) -> Optional[str]:
    """
    從模型回覆中提取最後一個 Markdown 程式碼區塊，
    並自動清除可能帶有的 '行號 | ' 前綴。
    """
    cleaned = DETAILS_PATTERN.sub("", assistant_markdown or "")
    matches = CODE_BLOCK_PATTERN.findall(cleaned)
    if not matches:
        return None
    code = matches[-1].rstrip("\r\n")
    lines = code.splitlines()
    if len(lines) >= 2 and sum(1 for ln in lines if LINE_PREFIX_PATTERN.match(ln)) >= len(lines) * 0.8:
        code = "\n".join(LINE_PREFIX_PATTERN.sub("", ln) for ln in lines)
    return code


# ---------------------------------------------------------------------------
# UI 元件互動處理函式（檔案載入、清空、寫入與備份還原）
# ---------------------------------------------------------------------------
def ui_load_file(raw_path: str, start_line: float, end_line: float, backend: str):
    """加載檔案並更新預覽區域與狀態條。"""
    budget = CLOUD_PROMPT_CHARS if backend == BACKEND_CLOUD else LOCAL_PROMPT_CHARS
    numbered_code, _plain, status, lang = load_source_slice(
        raw_path, start_line, end_line, budget_chars=budget
    )
    return gr.Code(value=numbered_code, language=lang), status


def ui_clear_file():
    """卸載檔案並重設所有相關元件狀態。"""
    return (
        "",
        0,
        0,
        gr.Code(value="", language=None),
        "📄 **未挂载文件**（可直接对话，或输入本地文件路径直读源码）",
        "",
        "ℹ️ 挂载文件后，此处将自动显示修改前后的 Diff 差异与 AST 语法检查结果。",
        gr.Button(interactive=False),
        gr.Button(interactive=False),
        "",
        "",
    )


def ui_apply_patch(
    raw_path: str,
    start_line: float,
    end_line: float,
    candidate_full_text: str,
    backend: str,
):
    """執行安全寫入，原檔案自動備份至 .workbench_bak/。"""
    resolved = resolve_file_path(raw_path)
    if not resolved:
        notify("warning", "未找到目标文件，无法写入。")
        return gr.skip(), gr.skip(), "❌ 写入失败：找不到目标文件。", gr.Button(interactive=False)

    if not candidate_full_text:
        notify("warning", "当前没有可写入的候选代码。")
        return gr.skip(), gr.skip(), "⚠️ 当前没有可写入的候选代码。", gr.Button(interactive=False)

    fname = os.path.basename(resolved)
    success, msg, bak_filename = backup_and_write_file(resolved, candidate_full_text)
    if success:
        budget = CLOUD_PROMPT_CHARS if backend == BACKEND_CLOUD else LOCAL_PROMPT_CHARS
        numbered_code, _plain, file_status, lang = load_source_slice(
            raw_path, start_line, end_line, budget_chars=budget
        )
        notify("info", f"已安全写入 {fname}，原文件已备份至 .workbench_bak/")
        return (
            gr.Code(value=numbered_code, language=lang),
            file_status,
            f"💾 **写入成功**：`{fname}` 已更新（自动备份：`{bak_filename}`；如需重新应用请先点右侧撤销）",
            gr.Button(interactive=False),
        )
    else:
        notify("warning", f"写入失败：{msg}")
        return gr.skip(), gr.skip(), f"❌ **写入失败**：`{msg}`", gr.skip()


def ui_restore_backup(
    raw_path: str,
    start_line: float,
    end_line: float,
    candidate_full_text: str,
    backend: str,
):
    """還原最近一次備份並動態重置應用按鈕。"""
    resolved = resolve_file_path(raw_path)
    if not resolved:
        notify("warning", "请先挂载目标文件。")
        return gr.skip(), gr.skip(), "⚠️ 请先在左侧挂载目标文件。", gr.skip()

    fname = os.path.basename(resolved)
    success, msg, bak_filename = restore_latest_backup(resolved)
    if success:
        budget = CLOUD_PROMPT_CHARS if backend == BACKEND_CLOUD else LOCAL_PROMPT_CHARS
        numbered_code, _plain, file_status, lang = load_source_slice(
            raw_path, start_line, end_line, budget_chars=budget
        )
        can_reapply = bool(candidate_full_text)
        extra_tip = "（左侧【应用修改】按钮已重新解锁，可随时再次应用）" if can_reapply else ""
        notify("info", f"已从 {bak_filename} 恢复原文件！")
        return (
            gr.Code(value=numbered_code, language=lang),
            file_status,
            f"↩️ **已回滚恢复**：还原自 `{bak_filename}` {extra_tip}",
            gr.Button(interactive=can_reapply),
        )
    else:
        notify("warning", f"未找到 {fname} 的历史备份文件。")
        return (
            gr.skip(),
            gr.skip(),
            f"ℹ️ {msg}",
            gr.Button(interactive=bool(candidate_full_text)),
        )


def append_user_message(user_text: str, history: Optional[List[dict]]):
    """將使用者訊息附加到歷史記錄中。"""
    history = list(history or [])
    text = (user_text or "").strip()
    if not text:
        return "", history
    if len(text) > MAX_MESSAGE_CHARS:
        text = text[:MAX_MESSAGE_CHARS]
        notify("warning", f"输入过长，已截断到 {MAX_MESSAGE_CHARS} 字符。")
    history.append({"role": "user", "content": text})
    return "", history


def prepare_autofix_prompt(syntax_err_detail: str) -> str:
    """自動組裝 AST 語法錯誤修復 Prompt。"""
    if not syntax_err_detail:
        return ""
    return (
        f"你上一轮输出的代码在 Python AST 静态语法检查中报错：{syntax_err_detail}。\n"
        "请仔细检查括号闭合、缩进一致性与语法结构，修复该错误并完整输出修正后的代码块。"
    )
