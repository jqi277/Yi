# fastapi_app.py  (v3.4 - fixed)
import os
import base64
import json
import logging
import traceback
from typing import Dict, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from openai import OpenAI

VERSION = "3.4"
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

@app.get("/version")
def version():
    return {"version": VERSION, "debug": DEBUG, "schema": SCHEMA_ID}


def _to_data_url(content: bytes, content_type: str) -> str:
    b64 = base64.b64encode(content).decode("utf-8")
    return f"data:{content_type};base64,{b64}"


def _build_tools_schema() -> List[Dict[str, Any]]:
    """
    工具约束为 C 端扁平结构；允许将更丰富的内容放在 meta 中。
    （注意括号/花括号的闭合顺序，已校对）
    """
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
    """
    三象拆分法：姿态 / 神情 / 环境 → 卦象组合 → 总结
    保持对外 JSON 扁平，细节放 meta.triple_analysis。
    """
    sys = (
        "你是 Selfy AI 的易经分析助理。"
        "请使用“三象拆分法”先进行结构化分析："
        "1. 姿态 → 给出对应卦象、特征（列表）、解读、性格倾向；"
        "2. 神情 → 给出对应卦象、特征（列表）、解读、性格倾向；"
        "3. 环境 → 给出对应卦象、特征（列表）、解读、性格倾向；"
        "4. 卦象组合：融合三象，给出整体意境的1-2句描述；"
        "5. 总结性格印象：用一句意境化中文总结。"
        "然后将上述内容映射到工具函数 submit_analysis_v3 的 JSON："
        "- summary：写“总结性格印象”；"
        "- archetype：意境化标签，如“外冷内热”等；"
        "- sections：将“姿态/神情/面相(若无面相则用神情替代)”各压缩为一句中文；"
        "- domains：仅从 ['金钱与事业','配偶与感情'] 中选择相关项；"
        "- meta：把完整“三象拆分法”的结构体放在 meta.triple_analysis；如有更细特征也放在 meta.sections_detail。"
        "严格通过工具函数 submit_analysis_v3 返回，不要产生其它输出。"
    )
    user = (
        "请分析这张图片，结合易经/面相/五官关系，但忽略背景与服饰。"
        "语言：中文。保证工具 JSON 严格符合 schema，并包含 meta.triple_analysis。"
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

    # 第一次：强制指定函数（不要用 "auto"）
    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        tools=_build_tools_schema(),
        tool_choice={"type": "function", "function": {"name": "submit_analysis_v3"}},
        messages=messages,
    )

    if DEBUG:
        try:
            logger.debug("[OAI] raw response (pass1): %s", resp)
        except Exception:
            pass

    choice = resp.choices[0]
    tool_calls = getattr(choice.message, "tool_calls", None)

    # ✅ 正常走工具
    if tool_calls:
        tool = tool_calls[0]
        if tool.function.name != "submit_analysis_v3":
            raise RuntimeError(f"Unexpected tool called: {tool.function.name}")
        try:
            args = json.loads(tool.function.arguments)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Tool arguments JSON decode failed: {e}")
        return {"tool_args": args, "oai_raw": resp if DEBUG else None}

    # ❗兜底：有些时候模型会直接把 JSON 写进 content，不走工具；我们尝试解析
    content = getattr(choice.message, "content", None)
    if isinstance(content, str) and content.strip().startswith("{"):
        try:
            args = json.loads(content)
            return {"tool_args": args, "oai_raw": resp if DEBUG else None}
        except Exception:
            pass

    # ❗兜底重试：再发一次，附加一句强提示，并启用 JSON 响应格式
    harder_messages = messages + [
        {"role": "system", "content": "你必须通过函数 submit_analysis_v3 返回结果，严格符合 schema。不要直接输出文本。"}
    ]
    resp2 = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.1,
        tools=_build_tools_schema(),
        tool_choice={"type": "function", "function": {"name": "submit_analysis_v3"}},
        response_format={"type": "json_object"},
        messages=harder_messages,
    )

    if DEBUG:
        try:
            logger.debug("[OAI] raw response (pass2): %s", resp2)
        except Exception:
            pass

    choice2 = resp2.choices[0]
    tool_calls2 = getattr(choice2.message, "tool_calls", None)
    if tool_calls2:
        tool2 = tool_calls2[0]
        try:
            args2 = json.loads(tool2.function.arguments)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Tool arguments JSON decode failed (pass2): {e}")
        return {"tool_args": args2, "oai_raw": resp2 if DEBUG else None}

    # 仍失败：尽可能把内容带回 debug
    msg = "Model did not return tool_calls after forced attempt."
    if DEBUG:
        raw1 = getattr(choice, "message", None)
        raw2 = getattr(choice2, "message", None)
        logger.error("[OAI] %s\npass1=%s\npass2=%s", msg, raw1, raw2)
    raise RuntimeError(msg)



def _join_cn(items: List[str]) -> str:
    items = [s for s in items if isinstance(s, str) and s.strip()]
    if not items:
        return ""
    return "、".join(items)


def _coerce_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    兼容“富结构”并压平：
    - sections.* 保证为字符串；如果收到对象{features,hexagram,meaning,advice}，拼句并下沉到 meta.sections_detail
    - domains 支持对象/数组两种；对象时将键数组化，并把详文存到 meta.domains_detail
    - 若不存在 meta.triple_analysis 但能从 sections_detail 推断，自动补一份
    """
    allowed_domains = {"金钱与事业", "配偶与感情"}
    out = dict(data)  # 浅拷贝

    # 确保 meta
    meta = out.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    out["meta"] = meta

    # sections 处理
    sections = out.get("sections") or {}
    if isinstance(sections, dict):
        detail_bucket = meta.setdefault("sections_detail", {})
        for k in ["姿态", "神情", "面相"]:
            v = sections.get(k)
            if isinstance(v, dict):
                detail_bucket[k] = v
                features = v.get("features") if isinstance(v.get("features"), list) else []
                features_txt = _join_cn(features)
                parts = []
                if features_txt:
                    parts.append(f"特征：{features_txt}")
                if v.get("hexagram"):
                    parts.append(f"卦象：{v.get('hexagram')}")
                if v.get("meaning"):
                    parts.append(f"含义：{v.get('meaning')}")
                if v.get("advice"):
                    parts.append(f"建议：{v.get('advice')}")
                sections[k] = "；".join([p for p in parts if p])
            elif isinstance(v, str):
                pass
            else:
                sections[k] = sections.get(k) or ""
        out["sections"] = sections

    # domains 处理
    domains = out.get("domains")
    if isinstance(domains, dict):
        domain_keys = [k for k in domains.keys() if k in allowed_domains]
        out["domains"] = domain_keys
        meta["domains_detail"] = {k: domains[k] for k in domain_keys}
    elif isinstance(domains, list):
        out["domains"] = [d for d in domains if d in allowed_domains]
    else:
        out["domains"] = []

    # 必填兜底
    out["summary"] = out.get("summary") or ""
    out["archetype"] = out.get("archetype") or ""
    try:
        out["confidence"] = float(out.get("confidence", 0.0))
    except Exception:
        out["confidence"] = 0.0

    # 自动补 meta.triple_analysis（如果缺失）
    triple = meta.get("triple_analysis")
    if not isinstance(triple, dict):
        sd = meta.get("sections_detail") or {}
        if isinstance(sd, dict) and any(isinstance(sd.get(x), dict) for x in ["姿态", "神情"]):
            def _mk(seg):
                segd = sd.get(seg) or {}
                return {
                    "卦象": segd.get("hexagram", ""),
                    "特征": segd.get("features", []),
                    "解读": segd.get("meaning", ""),
                    "性格倾向": segd.get("advice", ""),
                }
            triple = {
                "姿态": _mk("姿态"),
                "神情": _mk("神情"),
                "环境": _mk("环境"),
                "组合意境": "",
                "总结": out.get("summary", ""),
            }
            meta["triple_analysis"] = triple

    return out


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded.")

        content_type = file.content_type or ""
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=415, detail=f"Unsupported content type: {content_type}")

        raw = await file.read()
        if not raw or len(raw) == 0:
            raise HTTPException(status_code=400, detail="Empty file.")

        if len(raw) > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (>15MB).")

        data_url = _to_data_url(raw, content_type)
        logger.info("[UPLOAD] file=%s size=%d type=%s", file.filename, len(raw), content_type)

        result = _call_gpt_tool_with_image(data_url)
        tool_args = result["tool_args"]

        # 统一规范化输出（兼容富结构 & 注入 triple_analysis）
        final_out = _coerce_output(tool_args)

        # DEBUG 附加
        if DEBUG:
            meta = final_out.setdefault("meta", {})
            meta.setdefault("debug", {})
            meta["debug"]["debug_mode"] = True
            meta["debug"]["file_info"] = {
                "filename": file.filename,
                "content_type": content_type,
                "size": len(raw),
            }
            if result.get("oai_raw") is not None:
                try:
                    meta["debug"]["oai_choice_finish_reason"] = result["oai_raw"].choices[0].finish_reason
                    meta["debug"]["oai_has_tool_calls"] = bool(result["oai_raw"].choices[0].message.tool_calls)
                except Exception:
                    meta["debug"]["oai_choice_finish_reason"] = "n/a"
                    meta["debug"]["oai_has_tool_calls"] = "n/a"

        return JSONResponse(content=final_out, status_code=200)

    except HTTPException as he:
        if DEBUG:
            return JSONResponse(
                status_code=he.status_code,
                content={
                    "error": he.detail,
                    "debug": {"trace": traceback.format_exc()}
                }
            )
        raise
    except Exception as e:
        logger.exception("[ERROR] /upload failed: %s", e)
        body = {"error": "Internal Server Error"}
        if DEBUG:
            body["debug"] = {"message": str(e), "trace": traceback.format_exc()}
        return JSONResponse(status_code=500, content=body)
