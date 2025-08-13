# Selfy AI v3.7.2 Mobile + Post-processing
- 后端：`fastapi_app.py`（v3.7.10 runtime，v3.7.2 生成风格 + 文本后处理：
  - 合并三分象“说明+解读”成一句，并去重去符号；
  - “五官细节”分号→句号；
  - 卦象组合依据三象卦意重写总结（60–90字），避免机械拼接；
  - 事业/感情：状态合并分析，建议抽取命令式语句。)
- 前端：`index_mobile.html`（移动端一列，适配新句式与“五官细节”）

## 部署
1. 覆盖到仓库根目录；Render 启动命令：
   ```bash
   uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT --proxy-headers
   ```
2. 环境变量：
   - `OPENAI_API_KEY`（必填）
   - `DEBUG=1`（可选）

## 推送
```bash
cd D:\Project\Android_Projects\selfy_ai
git add fastapi_app.py index_mobile.html README.md
git commit -m "v3.7.2 mobile postproc2: clean punctuation, merge sentences, synthesize combo, refine status/suggestion"
git pull --rebase origin main
git push origin main
```

## 使用
部署后访问：`/mobile`。
