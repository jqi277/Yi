# fastapi_app.py
import os
import re
import uuid
import shutil
import logging
from typing import Optional, List, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
import orjson

# ---- 初始化应用（必须先于 app.mount/路由定义）----
app = FastAPI(title="Selfy AI API", version="0.2.0")

# CORS（看你需求，默认全开，前端调试方便）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---- 环境 & 日志 ----
load_dotenv()
API_KEY = os.getenv("API_KEY", "").strip()
BASE_URL = os.getenv("BASE_URL", "").strip()  # 可选：固定你的 https 域名
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
UPLOAD_DIR = "uploaded_photos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("selfy")

# ---- 静态图片挂载（/images/xxx.jpg）----
app.mount("/images", StaticFiles(directory=UPLOAD_DIR), name="images")

# ---- 输出结构（严格 JSON）----
class IChingAnalysis(BaseModel):
    hexagram: str                 # 卦名，如「乾」「坤」「既济」...
    hexagram_number: int          # 卦序 1~64
    changing_lines: List[int]     # 变爻（1~6），无则 []
    confidence: float             # 0~1
    cues: List[str]               # 触发该判断的五官/面相线索
    advice: str                   # 总体建议（面向用户）
    domains: Dict[str, str]       # {"金钱与事业": "...", "配偶与感情": "..."}

    @field_validator("confidence")
    @classmethod
    def _clip_conf(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]")

def sanitize_filename(name: str) -> str:
    name = (name or "").strip().replace(" ", "_")
    name = SAFE_NAME_RE.sub("", name)
    if not name or name.startswith("."):
        name = f"img_{uuid.uuid4().hex}.jpg"
    return name

def ensure_unique_path(dir_: str, name: str) -> str:
    root, ext = os.path.splitext(name)
    cand = name
    i = 1
    while os.path.exists(os.path.join(dir_, cand)):
        cand = f"{root}_{i}{ext}"
        i += 1
    return os.path.join(dir_, cand)

def get_absolute_base_url(request: Request) -> str:
    """
    Render 反代场景：优先 BASE_URL；否则从 X-Forwarded-* 推断 https 域名
    """
    if BASE_URL:
        return BASE_URL.rstrip("/")
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
    return f"{proto}://{host}"

def build_image_url(request: Request, filename: str) -> str:
    return f"{get_absolute_base_url(request)}/images/{filename}"

def check_api_key(x_api_key: Optional[str]) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ---- 健康/版本/根路由 ----
@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"message": "🎉 Selfy AI 易经分析接口在线。POST /upload 上传图片。"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"version": app.version}

# ---- OpenAI 客户端 ----
# 需要 requirements: openai>=1.30
from openai import OpenAI
client = OpenAI()  # OPENAI_API_KEY 从环境获得（Render 上配置）

# ---- 上传并分析（同时支持 /upload 与 /upload/）----
@app.post("/upload")
@app.post("/upload/")
async def analyze_with_vision(
    request: Request,
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None),
):
    # 简单鉴权（可在 .env 置空 API_KEY 关闭）
    check_api_key(x_api_key)

    # 基本校验
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image/* is supported.")
    size_header = request.headers.get("content-length")
    if size_header and int(size_header) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (> {MAX_UPLOAD_MB}MB).")

    # 保存文件
    try:
        clean_name = sanitize_filename(file.filename)
        save_path = ensure_unique_path(UPLOAD_DIR, clean_name)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        logger.exception("Saving file failed")
        raise HTTPException(status_code=500, detail=f"Save failed: {e}")

    filename = os.path.basename(save_path)
    public_url = build_image_url(request, filename)  # 确保是 https 外链

    # 结构化提示（严格 JSON 输出）
    system_prompt = (
        "你是《易经》面相与五官关系的专业分析师。"
        "仅基于面部结构（忽略背景/服饰/灯光），"
        "从五官比例、对称性、骨量与肉量、眉眼口鼻的关系、"
        "神态等抽象出稳定的人格与运势线索，"
        "并据此映射到最贴切的一卦（1~64），可含变爻。"
        "务必只返回 JSON，字段：hexagram, hexagram_number, changing_lines, confidence, cues, advice, domains。"
        "避免健康/种族/性别偏见，不输出任何个人身份信息。"
    )

    user_prompt = (
        "只返回 JSON，不要多余文本。若不确定也要给出 best-effort JSON，并降低 confidence。"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": public_url}},
                    ],
                },
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        data = orjson.loads(raw)  # 容错解析
        result = IChingAnalysis.model_validate(data)  # 结构校验
    except Exception as e:
        logger.exception("OpenAI vision failed")
        raise HTTPException(status_code=502, detail=f"Vision error: {e}")

    return JSONResponse(
        {
            "image_url": public_url,
            "analysis": result.model_dump(),
        }
    )

# ---- 可选：本地调试入口 ----
if __name__ == "__main__":
    import uvicorn
    # 本地启动：python fastapi_app.py
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
