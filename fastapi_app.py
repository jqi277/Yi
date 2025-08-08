# 📌 Selfy AI 易经分析说明文案
# 本系统结合现代AI图像理解与传统易经智慧，直接将用户上传的照片交由 GPT-4o 视觉模型解读，
# 提取面部与姿态特征，结合《易经》象数理占推演性格、事业与情感分析。

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import shutil, os
from dotenv import load_dotenv
import openai

load_dotenv(dotenv_path=".env")

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_photos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/images/{filename}")
async def get_uploaded_image(filename: str):
    return FileResponse(path=os.path.join(UPLOAD_DIR, filename))

@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"message": "🎉 Selfy AI 易经分析接口已上线！请通过 POST /upload/ 上传图片。"}

@app.post("/upload/")
async def analyze_with_vision(file: UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        image_url = f"https://yi-t31x.onrender.com/images/{file.filename}"  # 部署时替换为公网地址

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

        return {
            "analysis": result.split("\n"),
            "hexagrams": "由 GPT-4o 自动生成的卦象分析"
        }

    except Exception as e:
        return {"error": str(e)}
