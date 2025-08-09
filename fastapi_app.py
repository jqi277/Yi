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
app = FastAPI(title="Selfy AI API", version="0.3.0")

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

# 静态文件（给前端看图用，不再给 OpenAI 抓）
app.mount("/images", StaticFiles(directory=UPLOAD_DIR), name="images")

# ---------------- 数据模型（输出结构） ----------------
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
    读取本地图片，轻压缩 -> data URL（避免外链超时）
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

def call_openai_with_retry(payload: dict, retries: int = 1, backoff: float = 0.8):
    """
    payload 示例：
      {
        "messages": [...],
        "model": "gpt-4o",
        "temperature": 0.1,
        "tools": [...],
        "tool_choice": {"type":"function","function":{"name":"submit_analysis"}}
      }
    """
    client = OpenAI()  # 读取 OPENAI_API_KEY
    last_err = None
    for attempt in range(retries + 1):
        try:
            return client.chat.completions.create(**payload)
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
    public_url = build_image_url(request, filename)  # 仅供前端查看
    data_url = file_to_data_url(save_path)           # ✅ 发给 OpenAI

    # ---- System / User 提示（简洁）----
    system_prompt = (
        "你是《易经》面相与五官关系的专业分析师。仅基于面部结构（忽略背景/服饰/灯光），"
        "从五官比例、对称性、骨量与肉量、眉眼口鼻关系、神态等抽象线索映射到最贴切的一卦（1~64），可含变爻。"
        "必须通过函数（tools）提交结果，严禁输出除函数参数外的任何文本。避免健康/种族/性别偏见。"
    )
    user_prompt = "请按函数 schema 返回严谨 JSON 参数，不要输出普通文本。"

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

    # ---- 定义函数（工具）Schema，强制结构化 ----
    tools = [
        {
            "type": "function",
            "function": {
                "name": "submit_analysis",
                "description": "提交严格结构化的易经分析结果",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "hexagram": {"type": "string"},
                        "hexagram_number": {"type": "integer", "minimum": 1, "maximum": 64},
                        "changing_lines": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 1, "maximum": 6},
                            "default": []
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "cues": {"type": "array", "items": {"type": "string"}},
                        "advice": {"type": "string"},
                        "domains": {
                            "type": "object",
                            "properties": {
                                "金钱与事业": {"type": "string"},
                                "配偶与感情": {"type": "string"}
                            },
                            "required": ["金钱与事业", "配偶与感情"]
                        }
                    },
                    "required": ["hexagram", "hexagram_number", "changing_lines", "confidence", "cues", "advice", "domains"],
                    "additionalProperties": False
                },
            },
        }
    ]

    # ---- 强制调用该函数并解析参数 ----
    try:
        resp = call_openai_with_retry(
            {
                "messages": messages,
                "model": "gpt-4o",
                "temperature": 0.1,
                "tools": tools,
                "tool_choice": {"type": "function", "function": {"name": "submit_analysis"}},
            },
            retries=1,
        )

        choice = resp.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", None)
        if not tool_calls or tool_calls[0].function.name != "submit_analysis":
            logger.error("No tool call returned. raw=%r", getattr(choice.message, "content", "")[:200])
            raise HTTPException(status_code=502, detail="Vision error: tool call missing")

        args_str = tool_calls[0].function.arguments or "{}"
        try:
            data = orjson.loads(args_str)
        except Exception as e:
            logger.error("Tool JSON parse failed: %s ; head=%r", e, args_str[:200])
            raise HTTPException(status_code=502, detail="Vision error: invalid tool JSON")

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
