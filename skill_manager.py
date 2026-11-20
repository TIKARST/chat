# -*- coding: utf-8 -*-
"""
skill_manager.py: 輕量、乾淨的 Skill 自動掃描、本地執行與外部 HTTP API 調用管理器。
支援：
1. 資料夾封裝 Skill（含 SKILL.md，自動聚合 references/*.md 與 scripts/*.py，不再拆成多個頂層 Skill）
2. 單一檔案 Skill（skills/*.md 或 skills/*.py）
使用純原生 Python 實現，無額外冗餘框架依賴。
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from config import BASE_DIR, logger


# ---------------------------------------------------------------------------
# 資料模型：Skill 實體定義
# ---------------------------------------------------------------------------
@dataclass
class Skill:
    """代表一個已註冊的 Skill（可為 Markdown 提示詞資料夾包或 Python 邏輯函式）。"""
    name: str
    description: str = ""
    skill_type: str = "prompt"  # "prompt" (Markdown) 或 "python" (可執行模組)
    prompt_template: str = ""   # Markdown 內容或指示詞範本
    handler: Optional[Callable[..., Any]] = None  # Python 可呼叫對象
    file_path: Optional[str] = None
    parameters_schema: Dict[str, Any] = field(default_factory=dict)
    references: Dict[str, str] = field(default_factory=dict)  # 相對路徑 -> 參考字典正文
    scripts: Dict[str, str] = field(default_factory=dict)     # 相對路徑 -> 腳本絕對路徑

    def to_dict(self) -> Dict[str, Any]:
        """將 Skill 序列化為字典結構（供 API 或檢視使用）。"""
        return {
            "name": self.name,
            "description": self.description,
            "skill_type": self.skill_type,
            "prompt_template": self.prompt_template,
            "file_path": self.file_path,
            "parameters_schema": self.parameters_schema,
            "has_handler": self.handler is not None,
            "references": list(self.references.keys()),
            "scripts": list(self.scripts.keys()),
        }


# ---------------------------------------------------------------------------
# 輔助函式：純原生解析 Markdown Frontmatter 與讀取文字檔
# ---------------------------------------------------------------------------
_FRONTMATTER_PATTERN = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)


def _read_text_safe(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="gbk", errors="replace") as f:
            return f.read()


def _parse_markdown_skill(file_path: str) -> Skill:
    """純原生解析 Markdown Skill 檔案，支援 YAML 格式元數據或自動推導。"""
    content = _read_text_safe(file_path)

    file_stem = os.path.splitext(os.path.basename(file_path))[0]
    # 若檔名為 SKILL.md 或 README.md，則取上層目錄名作為預設 skill 名稱
    if file_stem.upper() in {"SKILL", "README"}:
        file_stem = os.path.basename(os.path.dirname(file_path))

    name = file_stem
    description = ""
    prompt_body = content

    match = _FRONTMATTER_PATTERN.match(content)
    if match:
        meta_block = match.group(1)
        prompt_body = content[match.end():].strip()
        for line in meta_block.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip().lower()
                v = v.strip().strip('"').strip("'")
                if k == "name" and v:
                    # 移除可能殘留的 namespace 前綴（如 user:）
                    name = v.split(":", 1)[-1] if ":" in v else v
                elif k in {"description", "desc"} and v:
                    description = v

    if not description:
        for line in prompt_body.splitlines():
            clean_line = line.strip().lstrip("#").strip()
            if clean_line:
                description = clean_line[:120]
                break

    return Skill(
        name=name,
        description=description,
        skill_type="prompt",
        prompt_template=prompt_body,
        file_path=os.path.abspath(file_path),
    )


def _load_folder_skill_bundle(skill_dir: str, skill_md_path: str) -> Skill:
    """
    載入完整資料夾封裝的 Skill：
    - 主文件：SKILL.md
    - 附帶參考字典：references/*.md（或其他子目錄 .md）
    - 附帶執行腳本：scripts/*.py（或其他子目錄 .py）
    """
    skill = _parse_markdown_skill(skill_md_path)
    abs_dir = os.path.abspath(skill_dir)

    for root, _dirs, files in os.walk(abs_dir):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, abs_dir).replace("\\", "/")
            if rel_path.lower() == "skill.md":
                continue
            if fname.lower().endswith(".md"):
                skill.references[rel_path] = _read_text_safe(fpath).strip()
            elif fname.lower().endswith(".py") and not fname.startswith("__"):
                skill.scripts[rel_path] = os.path.abspath(fpath)

    return skill


# ---------------------------------------------------------------------------
# 輔助函式：動態載入 Python 模組
# ---------------------------------------------------------------------------
def _load_python_skill(file_path: str) -> Optional[Skill]:
    """使用 importlib 動態載入 Python Skill 模組並提取處理常式。"""
    abs_path = os.path.abspath(file_path)
    file_stem = os.path.splitext(os.path.basename(file_path))[0]
    if file_stem in {"handler", "skill", "__init__"}:
        file_stem = os.path.basename(os.path.dirname(abs_path))

    module_name = f"skill_{file_stem}_{abs(hash(abs_path))}"

    try:
        spec = importlib.util.spec_from_file_location(module_name, abs_path)
        if not spec or not spec.loader:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        handler: Optional[Callable[..., Any]] = None
        for candidate_name in ["execute", "run", "handle", "main"]:
            fn = getattr(module, candidate_name, None)
            if callable(fn):
                handler = fn
                break

        skill_name = getattr(module, "SKILL_NAME", file_stem)
        skill_desc = getattr(module, "SKILL_DESCRIPTION", "")
        if not skill_desc and module.__doc__:
            skill_desc = module.__doc__.strip().splitlines()[0][:120]
        skill_prompt = getattr(module, "SKILL_PROMPT", "")
        skill_params = getattr(module, "SKILL_PARAMETERS", {})

        if handler and not skill_params:
            try:
                sig = inspect.signature(handler)
                skill_params = {
                    param_name: {
                        "default": None if param.default is inspect.Parameter.empty else str(param.default),
                        "required": param.default is inspect.Parameter.empty,
                    }
                    for param_name, param in sig.parameters.items()
                }
            except Exception:
                pass

        return Skill(
            name=skill_name,
            description=skill_desc or "Python 邏輯處理 Skill",
            skill_type="python",
            prompt_template=skill_prompt,
            handler=handler,
            file_path=abs_path,
            parameters_schema=skill_params,
        )
    except Exception as exc:
        logger.warning("載入 Python Skill 失敗 (%s): %s", file_path, exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# SkillRegistry: 統一註冊表與執行中心
# ---------------------------------------------------------------------------
class SkillRegistry:
    """管理 Skill 的掃描、註冊、本地調用與狀態查詢。"""

    def __init__(self, skills_dir: Optional[str] = None) -> None:
        self.skills_dir = skills_dir or os.path.join(BASE_DIR, "skills")
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """手動註冊或覆蓋一個 Skill。"""
        self._skills[skill.name] = skill
        extra = ""
        if skill.references or skill.scripts:
            extra = f" [refs={len(skill.references)}, scripts={len(skill.scripts)}]"
        logger.info("已註冊 Skill: %s (%s)%s", skill.name, skill.skill_type, extra)

    def get(self, skill_name: str) -> Optional[Skill]:
        """按名稱取得 Skill 實體。"""
        return self._skills.get(skill_name)

    def list_skills(self) -> List[Dict[str, Any]]:
        """列出所有已註冊 Skill 的摘要列表。"""
        return [
            {
                "name": s.name,
                "description": s.description,
                "skill_type": s.skill_type,
                "has_handler": s.handler is not None,
                "file_path": s.file_path,
                "references": list(s.references.keys()),
                "scripts": list(s.scripts.keys()),
            }
            for s in self._skills.values()
        ]

    def scan_skills(self, skills_dir: Optional[str] = None) -> int:
        """
        自動掃描目錄並註冊 Skills：
        1. 資料夾封裝：若目錄內含 SKILL.md，視為 1 個完整 Skill 包，
           自動聚合其 references/*.md 與 scripts/*.py，不再往下拆散註冊。
        2. 單一檔案：未封裝在 SKILL.md 目錄下的獨立 .md 或 .py 檔案仍照常註冊。
        """
        target_dir = skills_dir or self.skills_dir
        if not os.path.isdir(target_dir):
            try:
                os.makedirs(target_dir, exist_ok=True)
                logger.info("已自動建立 Skills 目錄: %s", target_dir)
            except OSError as err:
                logger.warning("無法建立 Skills 目錄 (%s): %s", target_dir, err)
                return 0

        self._skills.clear()
        loaded_count = 0

        for root, dirs, files in os.walk(target_dir):
            # 檢查當前目錄是否為一個標準資料夾 Skill 包（含 SKILL.md）
            skill_md_name = next((f for f in files if f.lower() == "skill.md"), None)
            if skill_md_name:
                skill_md_path = os.path.join(root, skill_md_name)
                skill = _load_folder_skill_bundle(root, skill_md_path)
                self.register(skill)
                loaded_count += 1
                # 剪枝：不再遞迴掃描該技能包內的 references/ 與 scripts/
                dirs[:] = []
                continue

            for file in sorted(files):
                file_lower = file.lower()
                full_path = os.path.join(root, file)

                if file_lower.endswith(".md") and file_lower != "readme.md":
                    skill = _parse_markdown_skill(full_path)
                    self.register(skill)
                    loaded_count += 1
                elif file_lower.endswith(".py") and not file_lower.startswith("__"):
                    skill = _load_python_skill(full_path)
                    if skill:
                        self.register(skill)
                        loaded_count += 1

        logger.info("SkillRegistry 掃描完畢，共載入 %d 個 Skills", loaded_count)
        return loaded_count

    def execute(self, skill_name: str, **kwargs: Any) -> Dict[str, Any]:
        """
        本地雙模調用核心進入點：
        - 若為 Python Skill 且具備 handler，直接傳遞 kwargs 執行；
        - 若為 Prompt Skill，以傳入參數格式化 Markdown 提示詞並回傳。
        """
        skill = self.get(skill_name)
        if not skill:
            raise KeyError(f"未找到名稱為 '{skill_name}' 的 Skill，可用 Skills: {list(self._skills.keys())}")

        if skill.skill_type == "python" and skill.handler:
            try:
                sig = inspect.signature(skill.handler)
                has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                filtered_kwargs = kwargs if has_var_kwargs else {k: v for k, v in kwargs.items() if k in sig.parameters}
                output = skill.handler(**filtered_kwargs)
                return {
                    "success": True,
                    "skill": skill.name,
                    "type": "python",
                    "result": output,
                }
            except Exception as exc:
                logger.exception("執行 Python Skill '%s' 時發生例外: %s", skill_name, exc)
                return {
                    "success": False,
                    "skill": skill.name,
                    "type": "python",
                    "error": str(exc),
                }

        formatted_prompt = skill.prompt_template
        if kwargs:
            try:
                formatted_prompt = formatted_prompt.format(**kwargs)
            except (KeyError, IndexError, ValueError):
                append_lines = ["\n\n### 輸入參數與上下文："]
                for k, v in kwargs.items():
                    append_lines.append(f"- **{k}**: {v}")
                formatted_prompt += "\n".join(append_lines)

        return {
            "success": True,
            "skill": skill.name,
            "type": "prompt",
            "prompt": formatted_prompt,
            "references": skill.references,
            "scripts": list(skill.scripts.keys()),
            "params": kwargs,
        }


# 全域單例
skill_registry = SkillRegistry()


# ---------------------------------------------------------------------------
# 外部 HTTP API: FastAPI 輕量路由
# ---------------------------------------------------------------------------
def create_skill_router(registry: Optional[SkillRegistry] = None) -> APIRouter:
    reg = registry or skill_registry
    router = APIRouter(prefix="/api/skills", tags=["Skills"])

    @router.get("", summary="列出所有可用的 Skills")
    def list_all_skills():
        return {
            "count": len(reg.list_skills()),
            "skills": reg.list_skills(),
        }

    @router.post("/reload", summary="重新掃描並載入 Skills")
    def reload_skills():
        count = reg.scan_skills()
        return {
            "success": True,
            "message": f"成功重新掃描，當前共有 {count} 個 Skills。",
            "skills": reg.list_skills(),
        }

    @router.get("/{skill_name}", summary="獲取指定 Skill 詳情")
    def get_skill_detail(skill_name: str):
        skill = reg.get(skill_name)
        if not skill:
            raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' 不存在")
        return skill.to_dict()

    @router.post("/{skill_name}/invoke", summary="外部調用指定 Skill")
    async def invoke_skill(skill_name: str, request: Request):
        skill = reg.get(skill_name)
        if not skill:
            raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' 不存在")

        body: Dict[str, Any] = {}
        try:
            body = await request.json()
        except Exception:
            body = {}

        params = body.get("params", body) if isinstance(body, dict) else {}
        if not isinstance(params, dict):
            params = {"input": params}

        return reg.execute(skill_name, **params)

    return router