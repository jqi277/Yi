# Selfy AI v3.7.5（移动路由版）

- 访问前端：`/mobile`（例：`https://YOUR-RENDER-DOMAIN/mobile`）
- 把 `index_mobile.html` 与 `fastapi_app.py` 放在仓库根目录（同级），Render 部署后即可访问。
- `index_mobile.html` 会自动使用 `location.origin` 作为 API 地址，无需手改。

## 部署
1. 设置环境变量：`OPENAI_API_KEY`（必要），`DEBUG=1`（可选）。
2. Start Command：`uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT --proxy-headers`
3. 打开 `/mobile` 进行前端上传与分析。

## 本地开发
```bash
pip install fastapi uvicorn openai
export OPENAI_API_KEY=你的Key
uvicorn fastapi_app:app --host 0.0.0.0 --port 10000 --reload
# 浏览器访问 http://127.0.0.1:10000/mobile
```