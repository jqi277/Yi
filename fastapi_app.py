# fastapi_app.py
import os
import re
import uuid
import shutil
import time
import logging
import base64
from io import BytesIO
from typing import Optional, List, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
import orjson
from openai import OpenAI
from PIL import Image

# ---------------- 初始化 ----------------
app = FastAPI(title="Selfy AI API", version="0.2.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

load_dotenv()
API_KEY = os.getenv("API_KEY", "").strip()
BASE_URL = os.getenv("BASE_URL", "").strip()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
UPLOAD_DIR = "uploaded_photos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("selfy")

# 静态文件挂载（供前端直接访问图片）
app.mount("/images", StaticFiles(directory=UPLOAD_DIR), name="images")

# ---------------- 数据模型 ----------------
class IChingAnalysis(BaseModel):
    hexagram: str                 # 卦名
    hexagram_number: int          # 1~64
    changing_lines: List[int]     # 变爻（可空列表）
    confidence: float             # 0~1
    cues: List[str]               # 触发线索
    advice: str                   # 总体建议
    domains: Dict[str, str]       # {"金钱与事业": "...", "配偶与感情": "..."}

    @field_validator("confidence")
    @classmethod
    def _clip_conf(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

# ---------------- 工具函数 ----------------
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

def file_to_data_url(path: str, max_side: int = 1600, quality: int = 85) -> str:
    """
    读取本地图片，做一次轻压缩（最长边不超过 max_side，JPEG 质量 quality），
    返回 data:image/jpeg;base64,... 的 Data URL，用于 OpenAI 视觉输入。
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_side / float(max(w, h)))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)))
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

def extract_json_str(text: str) -> str:
    """
    从模型返回里尽量提取纯 JSON：
    - 去掉 ```json/``` 代码块包裹
    - 去掉前后空白
    - 截取第一个 '{' 到最后一个 '}' 的子串
    失败则抛出 ValueError
    """
    if not text:
        raise ValueError("empty content")
    s = text.strip()

    # 去代码围栏
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()

    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no-brace-json")
    return s[start:end+1]

def call_openai_with_retry(messages, model="gpt-4o", temperature=0.2, retries: int = 1, backoff: float = 0.8):
    """
    对 OpenAI 调用做一次轻量重试（默认重试 1 次，退避 backoff 秒）
    """
    client = OpenAI()  # OPENAI_API_KEY 从环境变量读取
    last_err = None
    for attempt in range(retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
            )
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
            else:
                raise last_err

# ---------------- 根/健康/版本 ----------------
@app.get("/", operation_id="root_get")
def root_get():
    return {"message": "🎉 Selfy AI 易经分析接口在线。POST /upload 上传图片。"}

@app.head("/", include_in_schema=False)
def root_head():
    return {"message": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"version": app.version}

# ---------------- 上传并分析 ----------------
@app.post("/upload")
@app.post("/upload/")
async def analyze_with_vision(
    request: Request,
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None),
):
    # 鉴权（可选）
    check_api_key(x_api_key)

    # 校验
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
    public_url = build_image_url(request, filename)  # 方便前端直接查看
    data_url = file_to_data_url(save_path)           # ✅ 提供给 OpenAI，避免外链超时

    # 结构化提示
    system_prompt = (
        "你是《易经》面相与五官关系的专业分析师。"
        "仅基于面部结构（忽略背景/服饰/灯光），"
        "从五官比例、对称性、骨量与肉量、眉眼口鼻关系、神态等抽象线索，"
        "映射到最贴切的一卦（1~64），可含变爻。"
        "务必只返回 JSON，字段：hexagram, hexagram_number, changing_lines, confidence, cues, advice, domains。"
        "避免任何基于健康/种族/性别的偏见描述。"
        "只输出原始 JSON，不得包含代码块标记或任何说明文本。"
    )
    user_prompt = "只返回 JSON，不要多余文本。若不确定也要给出 best-effort JSON，并降低 confidence。"

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    try:
        # 带重试的 OpenAI 调用（默认重试 1 次）
        resp = call_openai_with_retry(messages, retries=1)
        raw = (resp.choices[0].message.content or "").strip()
        logger.info("LLM raw length=%s", len(raw))

        try:
            json_str = extract_json_str(raw)
            data = orjson.loads(json_str)
        except Exception as e:
            logger.error("JSON parse failed: %s ; head=%r", e, raw[:200])
            raise HTTPException(status_code=502, detail="Vision error: invalid JSON from model")

        result = IChingAnalysis.model_validate(data)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("OpenAI vision failed")
        raise HTTPException(status_code=502, detail=f"Vision error: {e}")

    return JSONResponse({"image_url": public_url, "analysis": result.model_dump()})

# ---------------- 本地调试 ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
