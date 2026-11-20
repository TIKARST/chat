# Qwen3.5-9B 极简本地联网与技能工作台

基于 **Ollama + Gradio + Exa** 构建的轻量化本地大模型交互工作台，专为 8GB 显存显卡（如 RTX 2060 Super）优化。采用模块化架构设计，支持思考过程折叠、层级上下文管理、实时联网检索与本地技能包（Skills）扩展。

## ✨ 核心特性

- **显存友好与层级上下文管理**：支持轮数与字符预算双层级上下文裁剪，显式化生成参数配置，8GB 显存流畅运行不爆显存。
- **思考模型（Thinking）适配**：支持思考模式开关，开启时以可折叠面板独立展示推理过程，避免“空白等待”与思维链污染上下文。
- **本地技能系统（Skills）**：内置 `skill_manager` 技能管理器，支持按需加载本地扩展技能包（`.md` 规范与 `.py` 工具脚本），灵活拓展垂直领域能力。
- **实时联网检索与容错降级**：集成 Exa Search API，一键勾选即可获取最新网页信息辅助回答；具备连接与读取超时保护，未配置密钥或网络波动时自动切换为纯本地推理。
- **全环境变量驱动**：核心模型地址、上下文预算、超时参数等均可通过 `.env` 环境变量覆盖，无需修改源码。

## 📁 项目结构

```text
chat/
├── APP.py              # 程序总入口，负责初始化、串接 UI 与 Backend 并启动服务
├── backend.py          # 大模型对话流式推理、联网检索与异常降级逻辑
├── ui.py               # Gradio 极简交互界面与思考面板布局
├── config.py           # 全局配置与环境变量加载管理
├── skill_manager.py    # 本地技能包扫描、解析与上下文注入管理
├── utils.py            # 上下文裁剪、文本清洗等通用工具函数
├── requirements.txt    # Python 依赖清单
└── skills/             # （本地忽略目录）存放自定义技能包与扩展脚本
🛠️ 环境准备与启动
1. 安装 Python 依赖
Bash
pip install -r requirements.txt
2. 配置本地模型与环境变量
确保本地已启动 Ollama 并拉取对应模型：

Bash
ollama pull qwen3.5:9b
在项目根目录创建 .env 文件，按需配置参数（如不需要联网功能可跳过 EXA_API_KEY）：

程式碼片段
EXA_API_KEY=你的_EXA_API_KEY
OLLAMA_URL=http://localhost:11434/api/chat
MODEL_NAME=qwen3.5:9b
3. 启动工作台
Bash
python APP.py
启动后，在浏览器打开终端提示的本地地址（默认 http://127.0.0.1:7860）即可开始使用。


---

### 更新步骤
1. 把上面的内容粘贴覆盖进 `README.md` 并保存[cite: 3]。
2. 切回 **GitHub Desktop**，左侧会多出 `README.md` 的黄色修改标记[cite: 2]。
3. 在 Summary 填 `docs: update README for modular architecture and skills`，点击 **Commi