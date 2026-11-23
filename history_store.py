import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# 统一存放在工作台同级目录下的 chat_history 文件夹
DEFAULT_HISTORY_DIR = Path(__file__).resolve().parent / "chat_history"


def new_session_id() -> str:
    """生成新的会话 ID（仅在内存中，首条消息发送后才建文件）"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"


def _get_filepath(session_id: str, history_dir: Path = DEFAULT_HISTORY_DIR) -> Path:
    safe_id = Path(session_id).name
    return history_dir / f"{safe_id}.jsonl"


def append_message(
    session_id: str,
    role: str,
    content: str,
    history_dir: Path = DEFAULT_HISTORY_DIR,
) -> None:
    """追加写入一条消息到对应会话的 .jsonl 文件（出错静默降级，不阻断主对话）"""
    if not session_id or not content:
        return
    try:
        history_dir.mkdir(parents=True, exist_ok=True)
        fp = _get_filepath(session_id, history_dir)
        record = {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "role": role,
            "content": content,
        }
        with fp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[history_store] 写入聊天记录失败（已忽略）: {exc}")


def load_messages(
    session_id: str,
    as_tuples: bool = False,
    history_dir: Path = DEFAULT_HISTORY_DIR,
) -> list[Any]:
    """
    读取指定会话的全部消息：
    - as_tuples=False（默认）：返回 [{'role': 'user'|'assistant', 'content': '...'}, ...]
    - as_tuples=True（兼容旧版 Gradio Chatbot）：返回 [(user_text, bot_text), ...]
    """
    if not session_id:
        return []
    fp = _get_filepath(session_id, history_dir)
    if not fp.exists():
        return []

    raw_msgs: list[dict[str, str]] = []
    try:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                role = obj.get("role")
                content = obj.get("content")
                if role in ("user", "assistant") and isinstance(content, str):
                    raw_msgs.append({"role": role, "content": content})
    except Exception as exc:
        print(f"[history_store] 读取聊天记录失败: {exc}")
        return []

    if not as_tuples:
        return raw_msgs

    # 转换为旧版 Gradio [(user, assistant)] 元组列表
    pairs: list[tuple[str | None, str | None]] = []
    pending_user: str | None = None
    for m in raw_msgs:
        if m["role"] == "user":
            if pending_user is not None:
                pairs.append((pending_user, None))
            pending_user = m["content"]
        else:
            pairs.append((pending_user, m["content"]))
            pending_user = None
    if pending_user is not None:
        pairs.append((pending_user, None))
    return pairs


def list_session_choices(
    history_dir: Path = DEFAULT_HISTORY_DIR,
    limit: int = 60,
) -> list[tuple[str, str]]:
    """返回 Gradio Dropdown 选项 [(展示标题, session_id), ...]，按最新修改时间倒序"""
    if not history_dir.exists():
        return []
    files = sorted(
        history_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]

    choices: list[tuple[str, str]] = []
    for fp in files:
        sid = fp.stem
        preview = "无标题会话"
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("role") == "user":
                        text = str(obj.get("content", "")).replace("\r", " ").replace("\n", " ").strip()
                        preview = (text[:26] + "…") if len(text) > 26 else (text or "空提问")
                        break
        except Exception:
            preview = "读取异常"
        choices.append((f"{sid} | {preview}", sid))
    return choices


def delete_session(session_id: str, history_dir: Path = DEFAULT_HISTORY_DIR) -> bool:
    """删除指定的本地会话文件"""
    if not session_id:
        return False
    fp = _get_filepath(session_id, history_dir)
    try:
        if fp.exists():
            fp.unlink()
            return True
    except Exception as exc:
        print(f"[history_store] 删除会话失败: {exc}")
    return False


def build_recent_context(
    messages: list[dict[str, str]],
    max_turns: int = 4,
    max_chars_per_msg: int = 2000,
) -> list[dict[str, str]]:
    """
    截取最近 max_turns 轮对话，并对超长单条文本做截断，
    确保本地 8GB 显存运行 9B 模型时上下文不膨胀。
    """
    if max_turns <= 0 or not messages:
        return []
    sliced = messages[-(max_turns * 2) :]
    result: list[dict[str, str]] = []
    for m in sliced:
        content = m["content"]
        if len(content) > max_chars_per_msg:
            content = content[:max_chars_per_msg] + "\n...[已截断以控制上下文长度]..."
        result.append({"role": m["role"], "content": content})
    return result