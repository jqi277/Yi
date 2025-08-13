# Selfy AI v3.7.2 Logic + Mobile Layout (Refined)
- 后端：`fastapi_app.py`（v3.7.7 runtime，v3.7.2 分析风格，三分象优化/卦象组合融合/事业与感情状态&建议分离）
- 前端：`index_mobile.html`（移动端一列）

## 部署
1. 把这三个文件放到仓库根目录；Render 启动命令：
   ```bash
   uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT --proxy-headers
   ```
2. 环境变量：
   - `OPENAI_API_KEY`（必填）
   - `DEBUG=1`（可选）

## 推送到 GitHub
```bash
cd D:\Project\Android_Projects\selfy_ai
git add fastapi_app.py index_mobile.html README.md
git commit -m "v3.7.2 mobile refined: section wording, combo synthesis, status/suggestion split"
git pull --rebase origin main
git push origin main
```

## 使用
部署后访问：`https://你的域名/mobile` 上传图片即可。
