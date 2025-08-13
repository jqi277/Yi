Selfy AI v3.9.0 · 易经推导版（带词库热加载）

1) 运行
   uvicorn fastapi_app:app --reload
   打开 http://127.0.0.1:8000/mobile

2) 词库
   - 同目录下 yijing_lexicon.json 为实时热加载；编辑保存后刷新页面生效。
   - 结构：persona(人物画像)、domains(事业/感情：state/advice)、wuxing/sheng/ke、fuse、rel_influence。

3) 输出规范
   - 🔮 卦象组合：纯人物画像（主/辅/基 + 生克），不带建议，不用“这类人/这种人”的指代词。
   - 💼 / 💗：近期状态（bullet 3–4条）+ 近期建议（bullet 2–3条），均按卦象与生克推导。

4) 注意
   - 最大上传 15MB；仅接受 image/*。
   - 若 OpenAI key 未配置，接口会报错（需要在环境变量中设置）。