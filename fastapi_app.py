from fastapi import FastAPI, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, shutil
# ... 省略不变的 import 与初始化 ...

UPLOAD_DIR = "uploaded_photos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 静态文件挂载（已在你代码里）
app.mount("/images", StaticFiles(directory=UPLOAD_DIR), name="images")

@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"message": "🎉 Selfy AI 易经分析接口已上线！请通过 POST /upload 上传图片。"}

# ✅ 同时支持 /upload 与 /upload/
@app.post("/upload")
@app.post("/upload/")
async def analyze_with_vision(request: Request, file: UploadFile = File(...)):
    # 保存上传文件
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # ✅ 动态生成公网图片 URL（不再写死 localhost 或固定域名）
    image_url = str(request.url_for("images", path=file.filename))

    prompt = """
你是一位结合《易经》六十四卦、五行、象数、心理学与图像观察的智慧分析师。
请你根据用户上传的照片，结合其神情、面部结构、姿态、气场，参考易经卦象与象义进行分析。

请提取并逐项分析：
1. 姿态（如头部朝向、身体倾斜、肢体动作） → 对应卦象与解释
2. 神情（情绪主导与眼神状态） → 卦象与象义
3. 面部结构（脸型、鼻形、眼距、眉眼等） → 象征性格趋势与命运走向

输出结构：
- 每项卦象的名称、象征、来源说明、性格/运势解读
- 三卦组合推演总体气质
- 金钱财富方面的潜力与建议
- 情感关系中的倾向与适配性
- 提供简洁而富象意的总结
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一位融合传统易经与现代图像观察的分析师，专精六十四卦、五行哲理与心理解读。"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt.strip()},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        max_tokens=1200,
        temperature=0.9
    )

    result = response.choices[0].message.content.strip()
    return {"analysis": [line for line in result.split("\n") if line.strip()],
            "hexagrams": "由 GPT-4o 自动生成的卦象分析"}
