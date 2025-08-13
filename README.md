# Selfy AI — 移动版（主/辅/基推导版）

**后端**：`fastapi_app.py`（runtime v3.8.0）  
**前端**：`index_mobile.html`（移动端优化 + 复制全文整洁输出）

## 本地启动
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=你的密钥   # Windows: $env:OPENAI_API_KEY="..."
uvicorn fastapi_app:app --reload
# 访问
# http://127.0.0.1:8000/mobile
# http://127.0.0.1:8000/docs
```

## Render 启动命令
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT`
- 环境变量：`OPENAI_API_KEY`（必须），`DEBUG=1`（可选）
