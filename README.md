# Selfy AI — YiJing Inference API

一个最小可部署的 FastAPI 服务：
- 多模态：上传图片 -> 走 OpenAI gpt-4o -> 工具函数返回 JSON
- 词库热加载：同目录 `yijing_lexicon.json`
- Render 兼容：`/health`、`/` 的 `HEAD` 返回 200；支持 `$PORT`

## 本地运行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
export DEBUG=1
python fastapi_app.py
# 浏览 http://127.0.0.1:10000/mobile
```

## Render 部署

1. 仓库根目录包含本项目文件（本 zip 解压后的全部内容）。
2. Render 仪表盘创建 Web Service：选择此仓库。
3. **Start Command** 填：
   ```bash
   uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT
   ```
4. 环境变量：
   - `OPENAI_API_KEY`：你的 OpenAI 密钥
   - 可选 `DEBUG=1`
5. Health Check：路径 `/` 或 `/health` 均可。我们显式处理了 `HEAD /` 返回 200。

## 词库格式（示例）

参考 `yijing_lexicon.json`，结构包含：
- `persona`、`wuxing`、`sheng`、`ke`、`fuse`、`rel_influence`
- `domains.career|love.{卦名}.state|advice`

你也可以只用最小词库，程序会兜底，但效果会更朴素。
