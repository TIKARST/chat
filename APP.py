import os
import json
import requests
import gradio as gr
from dotenv import load_dotenv

try:
    from exa_py import Exa
except ImportError:
    Exa = None

# 强行锁定从脚本同级目录加载 .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path, override=True)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3.5:9b"

# 直接从环境变量获取 Key 并初始化
EXA_API_KEY = os.getenv("EXA_API_KEY")
exa = Exa(api_key=EXA_API_KEY) if (Exa and EXA_API_KEY) else None

def search_web(query):
    if not exa:
        return "", ""
    try:
        # 使用最稳妥的调用方式，并设置合理的超时
        res = exa.search(query, num_results=3, timeout=6)
        results = getattr(res, "results", [])
        if not results:
            return "", ""
        snippets = [f"- [{r.title}]({r.url}): {r.text}" for r in results if getattr(r, "text", None)]
        web_text = "\n".join(snippets)
        if web_text:
            return web_text, "🌐 **[已成功通过 Exa 联网检索]**\n\n"
    except Exception:
        # 联网失败时静默降级，不抛出异常影响主流程
        pass
    return "", ""

def chat_fn(message, history, use_web):
    messages = []
    for item in history[-4:]:
        if isinstance(item, dict):
            role = item.get("role", "user")
            content = item.get("content", "")
            if role in ["user", "assistant"] and content:
                messages.append({"role": role, "content": str(content)})
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            if item[0]:
                messages.append({"role": "user", "content": str(item[0])})
            if item[1]:
                messages.append({"role": "assistant", "content": str(item[1])})

    prompt = message
    prefix = ""
    if use_web:
        web_info, prefix = search_web(message)
        if web_info:
            prompt = f"请参考以下联网搜索资料回答问题，并在适当位置标注参考链接：\n\n{web_info}\n\n问题：{message}"

    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True,
        "options": {"num_ctx": 4096}
    }

    partial_text = prefix
    yield partial_text  # 率先输出前缀，确保界面立刻有响应，绝不空白

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=60) as r:
            if r.status_code != 200:
                yield f"{partial_text}❌ Ollama 响应错误 (状态码 {r.status_code})"
                return
            for line in r.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        partial_text += chunk
                        yield partial_text
    except Exception as e:
        yield f"{partial_text}\n❌ 请求 Ollama 失败：{e}"

demo = gr.ChatInterface(
    fn=chat_fn,
    additional_inputs=[gr.Checkbox(label="开启 Exa 联网搜索", value=False)],
    title="Qwen3.5-9B 极简工作台",
    description="显存锁定 4K 上下文 | 保留最近 4 轮对话 | 支持自动降级联网检索"
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)