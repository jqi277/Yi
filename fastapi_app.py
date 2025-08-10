# fastapi_app.py  (v3.5+ui)
import os
import base64
import json
import logging
import traceback
from typing import Dict, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, Response
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


# ===== Bagua mapping =====
BAGUA_SYMBOLS = {
    "艮": "山",
    "离": "火",
    "兑": "泽",
    "乾": "天",
    "坤": "地",
    "震": "雷",
    "巽": "风",
    "坎": "水",
}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root():
    return HTMLResponse(
        """
        <h3>Selfy AI - YiJing Analysis API</h3>
        <ul>
          <li><a href="/docs">/docs (Swagger)</a></li>
          <li><a href="/health">/health</a></li>
          <li><a href="/version">/version</a></li>
        </ul>
    """
    )


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
                    "required": [
                        "summary",
                        "archetype",
                        "confidence",
                        "sections",
                        "domains",
                    ],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _prompt_for_image() -> List[Dict[str, Any]]:
    sys = (
        "你是 Selfy AI 的易经观相助手。必须先用“三象四段式”分析："
        "【姿态/神情/面容】三部分，每部分包含："
        "1) 说明：1句，描绘该面向的具体外观/动作/气质；"
        "2) 卦象：仅写一个卦名（如 艮、离、兑、乾、坤、震、巽、坎）；"
        "3) 解读：1–2句，解释该卦在此面向的含义；"
        "4) 性格倾向：1–2句，把“特征”合并成倾向，总结性格走向。"
        "然后给出："
        "5) 卦象组合：标题=三象卦名相加（如“艮 + 离 + 兑”），正文90–150字；"
        "6) 总结性格印象：20–40字的意境化总结。"
        "将结果通过 submit_analysis_v3 工具返回，字段要求："
        "- summary：第6条“总结性格印象”；"
        "- archetype：意境化标签（如“外冷内热”等）；"
        "- sections：把三象各压成一句中文（姿态/神情/面相）；"
        "- domains：仅从 ['金钱与事业','配偶与感情'] 选择；"
        "- meta.triple_analysis：需包含键：'姿态','神情','面容','组合意境','总结'；"
        "  其中每个三象对象含：'说明','卦象','解读','性格倾向'；"
        "- meta.domains_detail：对'金钱与事业'与'配偶与感情'分别给出60–90字建议；"
        "禁止使用“环境”作为第三象。语言：中文。禁止输出除工具调用以外的任何自由文本。"
        "八卦参考：艮=止=稳重/边界；离=火=明亮/表达；兑=泽=交流/愉悦；乾=天=领导/自信；坤=地=包容/承载；震=雷=行动；巽=风=协商；坎=水=谨慎/深度。"
    )
    user = (
        "请分析这张图片，结合易经/面相/五官关系。"
        "返回严格符合 schema 的工具 JSON，并包含 meta.triple_analysis（姿态/神情/面容四段式、组合意境、总结）与 meta.domains_detail。"
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
        try:
            args = json.loads(content)
            return {"tool_args": args, "oai_raw": resp if DEBUG else None}
        except Exception:
            pass

    harder_messages = messages + [
        {
            "role": "system",
            "content": "你必须通过函数 submit_analysis_v3 返回结果，严格符合 schema。不要直接输出文本。",
        }
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

    raise RuntimeError("Model did not return tool_calls after forced attempt.")


def _join_cn(items: List[str]) -> str:
    items = [s for s in items if isinstance(s, str) and s.strip()]
    if not items:
        return ""
    return "、".join(items)


def _coerce_output(data: Dict[str, Any]) -> Dict[str, Any]:
    allowed_domains = {"金钱与事业", "配偶与感情"}

    out = dict(data)
    meta = out.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    out["meta"] = meta

    sections = out.get("sections") or {}
    if not isinstance(sections, dict):
        sections = {}

    ta = meta.get("triple_analysis") if isinstance(meta.get("triple_analysis"), dict) else {}

    def _mk_line(name_cn: str, fallback_key: str) -> str:
        o = ta.get(name_cn) or {}
        desc = o.get("说明") or ""
        hexg = o.get("卦象") or ""
        mean = o.get("解读") or ""
        tend = o.get("性格倾向") or ""
        parts = [p for p in [desc, f"卦象：{hexg}" if hexg else "", mean, tend] if p]
        line = "；".join(parts)
        return line or (sections.get(fallback_key) or "")

    sections["姿态"] = _mk_line("姿态", "姿态")
    sections["神情"] = _mk_line("神情", "神情")
    sections["面相"] = _mk_line("面容", "面相")
    out["sections"] = sections

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
    out["sections"] = sections

    domains = out.get("domains")
    if isinstance(domains, dict):
        domain_keys = [k for k in domains.keys() if k in allowed_domains]
        out["domains"] = domain_keys
        meta["domains_detail"] = {k: domains[k] for k in domain_keys}
    elif isinstance(domains, list):
        out["domains"] = [d for d in domains if d in allowed_domains]
    else:
        out["domains"] = []

    out["summary"] = out.get("summary") or ""
    out["archetype"] = out.get("archetype") or ""
    try:
        out["confidence"] = float(out.get("confidence", 0.0))
    except Exception:
        out["confidence"] = 0.0

    if not isinstance(meta.get("triple_analysis"), dict):
        sd = meta.get("sections_detail") or {}
        if isinstance(sd, dict) and any(isinstance(sd.get(x), dict) for x in ["姿态", "神情", "面相"]):
            def _mk(sd_key):
                segd = sd.get(sd_key) or {}
                return {
                    "说明": "",
                    "卦象": segd.get("hexagram", ""),
                    "特征": segd.get("features", []),
                    "解读": segd.get("meaning", ""),
                    "性格倾向": segd.get("advice", ""),
                }

            meta["triple_analysis"] = {
                "姿态": _mk("姿态"),
                "神情": _mk("神情"),
                "面容": _mk("面相"),
                "组合意境": "",
                "总结": out.get("summary", ""),
            }

    ta2 = meta.get("triple_analysis") or {}
    hexes = [
        ta2.get("姿态", {}).get("卦象", ""),
        ta2.get("神情", {}).get("卦象", ""),
        ta2.get("面容", {}).get("卦象", ""),
    ]
    combo_title = " + ".join([h for h in hexes if h])
    if combo_title:
        meta["combo_title"] = combo_title

    # === Build UI helpers for frontend ===

    # 1) 顶部 tag：性格标签 + 可信度
    meta["headline"] = {
        "tag": out.get("archetype", ""),
        "confidence": out.get("confidence", 0.0),
    }

    # 2) 分象标题：显示“姿态 → 艮卦（山）”等
    def _title_with_hex(section_key: str, ta_key: str):
        hexname = (ta2.get(ta_key, {}) or {}).get("卦象", "")
        symbol = BAGUA_SYMBOLS.get(hexname, "")
        if hexname and symbol:
            return f"{section_key} → {hexname}卦（{symbol}）"
        elif hexname:
            return f"{section_key} → {hexname}卦"
        else:
            return section_key

    meta["sections_titles"] = {
        "姿态": _title_with_hex("姿态", "姿态"),
        "神情": _title_with_hex("神情", "神情"),
        "面相": _title_with_hex("面相", "面容"),
    }

    # 3) 卦象组合：标题 + 要点（供第一排 box 列出）
    combo_points = []
    for k in ("姿态", "神情", "面容"):
        tend = (ta2.get(k, {}) or {}).get("性格倾向", "")
        if isinstance(tend, str) and tend.strip():
            combo_points.append(tend.strip())

    combo_yijing = (ta2.get("组合意境", "") or "").strip()
    if combo_yijing:
        combo_points.append(combo_yijing)

    combo_title_txt = meta.get("combo_title", "").strip()
    combo_full_title = f"🔮 卦象组合：{combo_title_txt}" if combo_title_txt else "🔮 卦象组合"

    meta["combo_detail"] = {
        "title": combo_full_title,
        "bullets": combo_points[:6],
    }

    # 4) 总结性格：加一行意境句
    h1 = (ta2.get("姿态", {}) or {}).get("卦象", "")
    h2 = (ta2.get("神情", {}) or {}).get("卦象", "")
    h3 = (ta2.get("面容", {}) or {}).get("卦象", "")
    s1, s2, s3 = BAGUA_SYMBOLS.get(h1, ""), BAGUA_SYMBOLS.get(h2, ""), BAGUA_SYMBOLS.get(h3, "")
    imagery = ""
    if s1 and s2 and s3:
        imagery = f"“{s1}中有{s2}，{s2}映{s3}面”"
    elif s1 and s2:
        imagery = f"“{s1}映{s2}光”"
    elif s2 and s3:
        imagery = f"“{s2}照{s3}容”"

    meta["summary_rich"] = {
        "lead": out.get("summary", ""),
        "imagery": f"在易经意境中，像是 {imagery} —— 内藏光芒，择人而耀。" if imagery else "",
    }

    out["meta"] = meta
    return out


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded.")

        content_type = file.content_type or ""
        if not content_type.startswith("image/"):
            raise HTTPException(
                status_code=415, detail=f"Unsupported content type: {content_type}"
            )

        raw = await file.read()
        if not raw or len(raw) == 0:
            raise HTTPException(status_code=400, detail="Empty file.")

        if len(raw) > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (>15MB).")

        data_url = _to_data_url(raw, content_type)
        logger.info("[UPLOAD] file=%s size=%d type=%s", file.filename, len(raw), content_type)

        result = _call_gpt_tool_with_image(data_url)
        tool_args = result["tool_args"]

        final_out = _coerce_output(tool_args)

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
                    meta["debug"]["oai_choice_finish_reason"] = result["oai_raw"].choices[
                        0
                    ].finish_reason
                    meta["debug"]["oai_has_tool_calls"] = bool(
                        result["oai_raw"].choices[0].message.tool_calls
                    )
                except Exception:
                    meta["debug"]["oai_choice_finish_reason"] = "n/a"
                    meta["debug"]["oai_has_tool_calls"] = "n/a"

        return JSONResponse(content=final_out, status_code=200)

    except HTTPException as he:
        if DEBUG:
            return JSONResponse(
                status_code=he.status_code,
                content={"error": he.detail, "debug": {"trace": traceback.format_exc()}},
            )
        raise
    except Exception as e:
        logger.exception("[ERROR] /upload failed: %s", e)
        body = {"error": "Internal Server Error"}
        if DEBUG:
            body["debug"] = {"message": str(e), "trace": traceback.format_exc()}
        return JSONResponse(status_code=500, content=body)
