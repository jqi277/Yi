# Selfy AI v3.7.2 Logic + Mobile Layout + 近期状态

- 运行版本：3.7.6；分析文风：3.7.2；移动端一列布局。
- `/mobile` 路由直接返回 `index_mobile.html`；前端自动使用 `location.origin` 作为 API 地址。

## 部署
1. 把 `fastapi_app.py` 和 `index_mobile.html` 放到仓库根目录。
2. Render 启动命令：
   ```bash
   uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT --proxy-headers
   ```
3. 环境变量：
   - `OPENAI_API_KEY`（必填）
   - `DEBUG=1`（可选）

## 使用
部署完成后访问 `https://你的域名/mobile`。上传图片即可得到结果：
- 三象展示为：
  - 第二行：`卦象：乾卦（天）`
  - 第三行：`说明 —— 解读；性格倾向`
- 卦象组合：一段式总述（优先采用 `meta.triple_analysis.总结`）。
- 事业/感情：补充“近期状态”，建议仍取 `meta.domains_detail`。
