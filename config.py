# -*- coding: utf-8 -*-
"""
config.py: 集中管理全域配置、模型參數、路徑與常數。
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 基礎路徑與環境變數載入
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), encoding="utf-8", override=True)


def get_env(name: str, default: Any, cast=str) -> Any:
    """安全解析環境變數，解析失敗時優雅回退默認值。"""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return cast(raw.strip())
    except (TypeError, ValueError):
        return default


def get_env_bool(name: str, default: bool = False) -> bool:
    """解析布林值環境變數。"""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


# ---------------------------------------------------------------------------
# 日誌配置
# ---------------------------------------------------------------------------
LOG_LEVEL = str(get_env("LOG_LEVEL", "INFO")).upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("qwen-workbench")

# ---------------------------------------------------------------------------
# 本地 Ollama 配置（守護顯存安全線）
# ---------------------------------------------------------------------------
OLLAMA_URL: str = str(get_env("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
CHAT_URL: str = f"{OLLAMA_URL}/api/chat"
TAGS_URL: str = f"{OLLAMA_URL}/api/tags"

MODEL_NAME: str = str(get_env("OLLAMA_MODEL", "qwen3.5:9b"))
NUM_CTX: int = int(get_env("OLLAMA_NUM_CTX", 16384, int))
NUM_PREDICT: int = int(get_env("OLLAMA_NUM_PREDICT", 2048, int))
TEMPERATURE: float = float(get_env("OLLAMA_TEMPERATURE", 0.7, float))
KEEP_ALIVE: str = str(get_env("OLLAMA_KEEP_ALIVE", "30m"))
DEFAULT_THINK: bool = get_env_bool("OLLAMA_THINK", False)

# ---------------------------------------------------------------------------
# 雲端 API 配置 (OpenAI 兼容規範，如 DeepSeek)
# ---------------------------------------------------------------------------
CLOUD_BASE_URL: str = str(get_env("CLOUD_BASE_URL", "https://api.deepseek.com/v1")).rstrip("/")
CLOUD_API_KEY: str = str(get_env("CLOUD_API_KEY", ""))
CLOUD_MODEL: str = str(get_env("CLOUD_MODEL", "deepseek-chat"))
CLOUD_PROMPT_CHARS: int = int(get_env("CLOUD_PROMPT_CHARS", 32000, int))

# ---------------------------------------------------------------------------
# 上下文與聯網檢索配置
# ---------------------------------------------------------------------------
MAX_HISTORY_MESSAGES: int = int(get_env("MAX_HISTORY_MESSAGES", 8, int))
MAX_MESSAGE_CHARS: int = int(get_env("MAX_MESSAGE_CHARS", 12000, int))
LOCAL_PROMPT_CHARS: int = int(get_env("MAX_PROMPT_CHARS", 32000, int))

EXA_API_KEY: str = str(get_env("EXA_API_KEY", "")).strip()
SEARCH_RESULTS: int = int(get_env("EXA_NUM_RESULTS", 3, int))
SEARCH_MAX_CHARS: int = int(get_env("EXA_MAX_CHARS", 650, int))
SEARCH_TIMEOUT: float = float(get_env("EXA_TIMEOUT", 20.0, float))
TECH_START_DATE: str = str(get_env("EXA_TECH_START_DATE", "2025-01-01"))

# ---------------------------------------------------------------------------
# 網絡超時與服務器設定
# ---------------------------------------------------------------------------
CONNECT_TIMEOUT: float = float(get_env("OLLAMA_CONNECT_TIMEOUT", 10.0, float))
READ_TIMEOUT: float = float(get_env("OLLAMA_READ_TIMEOUT", 300.0, float))
REQUEST_TIMEOUT: tuple[float, float] = (CONNECT_TIMEOUT, READ_TIMEOUT)

SERVER_NAME: str = str(get_env("GRADIO_SERVER_NAME", "127.0.0.1"))
SERVER_PORT: int = int(get_env("GRADIO_SERVER_PORT", 7860, int))

# ---------------------------------------------------------------------------
# 任務模式與引擎常量
# ---------------------------------------------------------------------------
BACKEND_LOCAL: str = "本地 Ollama (9B)"
BACKEND_CLOUD: str = "云端 API (OpenAI兼容)"

MODE_CHAT: str = "通用对话"
MODE_CODE_PATCH: str = "代码局部修改"
MODE_CODE_FULL: str = "完整文件重构"

MODE_PROFILES: Dict[str, Dict[str, Any]] = {
    MODE_CHAT: {
        "system_prompt": (
            "你是一个博学、严谨且具备多领域专业能力的智能 AI 专家与多技能协同 Agent。\n"
            "你精通传统数术哲学（如渊海子平八字、奇门遁甲排盘断局、玄空飞星风水堪舆等）、现代软件架构与代码工程、以及通用领域分析。\n"
            "【行为准则】：\n"
            "1. 当上下文挂载了特定专业技能（Skill）的推演规范或背景知识库时，你将自动担任该领域的顶尖专家，严格遵循其理论体系、排盘起局铁律、判定矩阵与标准输出格式开展推演；\n"
            "2. 在涉及排盘或推演的任务中，先呈现完整规范的盘面图表，再展开条理严密、层次清晰的专业深度剖析；\n"
            "3. 在非代码咨询中，提供详实、客观、富有洞察力的高质量解答，严禁死板限制为纯代码输出；在涉及代码任务时，同样保持高标准工程规范。"
        ),
        "temperature": TEMPERATURE,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.05,
        "num_predict": 4096,
    },
    MODE_CODE_PATCH: {
        "system_prompt": (
            "你是一个严谨的资深软件工程师与全能技术专家。当前处于【代码局部修改】模式：\n"
            "1. 针对用户需求或 Bug 进行精准修改，保持原有函数命名与结构一致性；\n"
            "2. 完整输出修改后的目标函数或代码块，严禁使用任何省略占位符；若参考源码带有行号，输出代码绝对不要包含行号前缀；\n"
            "3. 若当前任务激活了专业领域 Skill，请协同该 Skill 规范开展推理并给出专业成果；\n"
            "4. 先用一句话简述改动点，紧接着直接输出标准 Markdown 代码块。"
        ),
        "temperature": 0.25,
        "top_p": 0.85,
        "top_k": 20,
        "repeat_penalty": 1.05,
        "num_predict": 4096,
    },
    MODE_CODE_FULL: {
       "system_prompt": (
            "你是一个严谨的资深软件工程师与系统架构师。当前处于【完整文件重构/审查】模式：\n"
            "1. 【豁免铁律】若当前指令为代码审查/诊断，且经核查源码逻辑清晰、无明显 Bug 或安全隐患，请直接回复“代码逻辑完善，暂无明显问题，无需修改”并停止输出，严禁虚空找茬或输出任何代码块；\n"
            "2. 仅在确需修改或重构时，输出可直接运行的完整源文件代码，从头到尾一字不漏，严禁使用任何省略号占位符；\n"
            "3. 保留原有的依赖导入与接口兼容性；仔细检查括号闭合、缩进一致性与异常处理，确保语法零错误；\n"
            "4. 若当前任务涉及特定领域知识或专业 skill，请严格遵循其业务逻辑规范；\n"
            "5. 若输出了代码，先用一句话概括改动要点，紧接着直接输出完整 Markdown 代码块。"
        ),
        "temperature": 0.2,
        "top_p": 0.85,
        "top_k": 20,
        "repeat_penalty": 1.05,
        "num_predict": 4096,
    },
}

EXT_LANG_MAP: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "shell",
    ".bat": "shell",
    ".sql": "sql",
    ".md": "markdown",
}

TECH_DOMAINS: list[str] = [
    "github.com",
    "stackoverflow.com",
    "docs.python.org",
    "gradio.app",
    "huggingface.co",
    "pypi.org",
    "developer.mozilla.org",
    "docs.exa.ai",
]
