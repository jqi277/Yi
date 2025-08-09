# fastapi_app.py
import os
import re
import uuid
import shutil
import time
import logging
from typing import Optional, List, Dict

import requests
from requests.exceptions import RequestException
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
import orjson
from openai import OpenAI

# ---------------- 初始化 ----------------
app = FastAPI(title="Selfy AI API", version="0.2.1")

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

# 静态文件挂载
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
    # Render 场景：优先 BASE_URL；否则用代理头推断 https
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

def wait_until_public(url: str, timeout_total: float = 4.0, step: float = 0.4) -> None:
    """
    轮询公网上是否能 GET 到图片。用于避免 OpenAI 'Timeout while downloading'。
    总等待 <= timeout_total。用 GET（非 HEAD）以确保真实可读。
    """
    deadline = time.time() + timeout_total
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=(1.0, 2.5))  # 连接1s，读取2.5s
            if r.status_code == 200 and int(r.headers.get("Content-Length", "1")) > 0:
                return
        except RequestException:
            pass
        time.sleep(step)
    # 不抛错，后续还有 OpenAI 重试；你也可改成直接 503 更显性

def call_openai_with_retry(messages, model="gpt-4o", temperature=0.3, retries: int = 1, backoff: float = 0.8):
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
    public_url = build_image_url(request, filename)

    # 关键：OpenAI 调用前预热图片（避免 timeout while downloading）
    wait_until_public(public_url)

    # 结构化提示
    system_prompt = (
        "你是《易经》面相与五官关系的专业分析师。"
        "仅基于面部结构（忽略背景/服饰/灯光），"
        "从五官比例、对称性、骨量与肉量、眉眼口鼻关系、神态等抽象线索，"
        "映射到最贴切的一卦（1~64），可含变爻。"
        "务必只返回 JSON，字段：hexagram, hexagram_number, changing_lines, confidence, cues, advice, domains。"
        "避免任何基于健康/种族/性别的偏见描述。"
    )
    user_prompt = "只返回 JSON，不要多余文本。若不确定也要给出 best-effort JSON，并降低 confidence。"

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": public_url}},
            ],
        },
    ]

    try:
        # 带重试的 OpenAI 调用（默认重试 1 次）
        resp = call_openai_with_retry(messages, retries=1)
        raw = resp.choices[0].message.content or "{}"
        data = orjson.loads(raw)  # 容错解析
        result = IChingAnalysis.model_validate(data)  # 结构校验
    except Exception as e:
        logger.exception("OpenAI vision failed")
        raise HTTPException(status_code=502, detail=f"Vision error: {e}")

    return JSONResponse({"image_url": public_url, "analysis": result.model_dump()})

# ---------------- 本地调试 ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
