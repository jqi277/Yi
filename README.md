# Selfy AI v3.7.2 Logic + Mobile Layout

- 后端运行版本：3.7.6（仅运行时版本号）；分析文风：**3.7.2**（保留“性格倾向”独立字段，卦象组合允许适度重合，domains_detail 60–90字）。
- 移动端前端：`/mobile` 路由直接返回 `index_mobile.html`（一列纵向布局）。
- `API_BASE = location.origin`，无需手动改域名。

## 部署
1. 将 `fastapi_app.py` 与 `index_mobile.html` 放到仓库根目录（与 `requirements.txt` 同级）。
2. Render 启动命令保持：
   ```bash
   uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT --proxy-headers
   ```
3. 环境变量：
   - `OPENAI_API_KEY`（必填）
   - `DEBUG=1`（可选，开启调试日志）
   - `ANALYSIS_VERSION=372`（默认即372）

## 使用
部署完成后，访问：`https://你的域名/mobile`  
上传图片 → 点击“开始分析” → 返回卡片式结果。

## 目录
- `fastapi_app.py`：后端（372逻辑，包含 /mobile、/upload、/health、/version 等）
- `index_mobile.html`：移动端页面（上传、展示）
