# fastapi_app.py  (v3.5 → v3.5-ui-r1)
import os
import base64
import json
import logging
import traceback
from typing import Dict, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI

VERSION = "3.5"
SCHEMA_ID = "selfy.v3"
DEBUG = str(os.getenv("DEBUG", "0")).strip() in ("1", "true", "True", "YES", "yes")

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("selfy-ai")

app = FastAPI(title="Selfy AI - YiJing Analysis API", version=VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if DEBUG else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    client = OpenAI()
except Exception as e:
    logger.error("OpenAI client init failed: %s", e)
    client = None


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", include_in_schema=False)
def root():
    return HTMLResponse("""
        <h3>Selfy AI - YiJing Analysis API</h3>
        <ul>
          <li><a href="/docs">/docs (Swagger)</a></li>
          <li><a href="/health">/health</a></li>
          <li><a href="/version">/version</a></li>
        </ul>
    """)

@app.head("/", include_in_schema=False)
def root_head():
    return Response(status_code=200)

@app.get("/version")
def version():
    return {"version": VERSION, "debug": DEBUG, "schema": SCHEMA_ID}


def _to_data_url(content: bytes, content_type: str) -> str:
    b64 = base64.b64encode(content).decode("utf-8")
    return f"data:{content_type};base64,{b64}"


def _build_tools_schema() -> List[Dict[str, Any]]:
    # 与旧版完全兼容，meta 允许扩展（用于 combo / summary_rich / sections_titles / top_tag）
    return [
        {
            "type": "function",
            "function": {
                "name": "submit_analysis_v3",
                "description": "Return end-user facing JSON for Selfy AI YiJing analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "archetype": {"type": "string"},
                        "confidence": {"type": "number"},
                        "sections": {
                            "type": "object",
                            "properties": {
                                "姿态": {"type": "string"},
                                "神情": {"type": "string"},
                                "面相": {"type": "string"},
                            },
                            "required": ["姿态", "神情", "面相"],
                            "additionalProperties": False,
                        },
                        "domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Only from ['金钱与事业','配偶与感情']",
                        },
                        "meta": {
                            "type": "object",
                            "description": "Optional metadata for debugging or triple-analysis rich content",
                            "additionalProperties": True,
                        },
                    },
                    "required": ["summary", "archetype", "confidence", "sections", "domains"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _prompt_for_image() -> List[Dict[str, Any]]:
    # 说明：保留你原来的三象→组合→总结→domains 顺序；强调组合 bullets 与卦名含“象字”；
    sys = (
      "你是 Selfy AI 的易经观相助手。必须使用 submit_analysis_v3 工具返回严格 JSON。"
      "【分析范围】仅“姿态/神情/面相”，忽略服饰与环境。"
      "【三象四段式】每一象写：说明/卦象/解读/性格倾向；卦象只写 1 个（艮/离/兑/乾/坤/震/巽/坎），必要时在文本中可带象字（如 艮（山））。"
      "【卦象组合】写 meta.combo："
      "  - gua_list：三象卦名（不带括号）；"
      "  - bullets：2–4 条短句（7–12 字），如“外冷内热”“独立审美”“稳重理智”“交际选择性”；"
      "  - summary：40–80 字小段，概括外在/内在/对人关系；"
      "  - 另生成 meta.combo_title='姿态卦 + 神情卦 + 面相卦'。"
      "【summary】用两段式：第一行“这个人给人的感觉是：”下一行给一段带引号的总印象（30–50字）；"
      "第二段为意境句：“在易经意境中，像是 ‘X’ —— Y。”（20–40字）。"
      "【domains】数组仅从 ['金钱与事业','配偶与感情'] 取；详细建议放 meta.domains_detail，每个 160–220 字，给 2 个可执行动作（含时机/频率）+ 1 个失败预警信号。"
      "【triple_analysis】包含'姿态','神情','面容','组合意境','总结'，三象各含 说明/卦象/解读/性格倾向。"
      "【严格】只通过工具输出，禁止自由文本。"
    )
    user = (
        "请分析这张图片：先三象四段式，再写卦象组合（含 bullets 与 summary），再写两段式总结，最后给事业/感情建议。"
        "返回严格符合 schema 的工具 JSON。"
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]


def _call_gpt_tool_with_image(data_url: str) -> Dict[str, Any]:
    if client is None:
        raise RuntimeError("OpenAI client is not initialized. Check OPENAI_API_KEY.")

    messages = _prompt_for_image()
    messages[-1]["content"] = [
        {"type": "text", "text": messages[-1]["content"]},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    logger.debug("[OAI] Sending messages with image (Data URL)")

    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        tools=_build_tools_schema(),
        tool_choice={"type": "function", "function": {"name": "submit_analysis_v3"}},
        response_format={"type": "json_object"},
        messages=messages,
    )

    if DEBUG:
        try:
            logger.debug("[OAI] raw response (pass1): %s", resp)
        except Exception:
            pass

    choice = resp.choices[0]
    tool_calls = getattr(choice.message, "tool_calls", None)

    if tool_calls:
        tool = tool_calls[0]
        if tool.function.name != "submit_analysis_v3":
            raise RuntimeError(f"Unexpected tool called: {tool.function.name}")
        try:
            args = json.loads(tool.function.arguments)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Tool arguments JSON decode failed: {e}")
        return {"tool_args": args, "oai_raw": resp if DEBUG else None}

    content = getattr(choice.message, "content", None)
    if isinstance(content, str) and content.strip().startswith("{"):
