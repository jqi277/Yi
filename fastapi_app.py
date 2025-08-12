# fastapi_app.py  (v3.6-ui-plus)
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

VERSION = "3.6"
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

BAGUA_TRAITS = {
    "艮": "稳重/定界",
    "离": "明晰/表达",
    "兑": "亲和/交流",
    "乾": "自信/主导",
    "坤": "包容/承载",
    "震": "果断/行动",
    "巽": "圆融/协商",
    "坎": "谨慎/深思",
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
                            "description": "Optional metadata for debugging or rich content",
                            "additionalProperties": True,
                        },
                    },
                    "required": [
                        "summary","archetype","confidence","sections","domains"
                    ],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _prompt_for_image() -> List[Dict[str, Any]]:
    sys = (
        "你是 Selfy AI 的易经观相助手。"
        "必须先用“三象四段式”分析：【姿态/神情/面容】三部分，每部分包含："
        "1) 说明：1句，描绘该面向的具体外观/动作/气质；"
        "2) 卦象：仅写一个卦名（艮/离/兑/乾/坤/震/巽/坎）；"
        "3) 解读：1–2句，解释该卦在此面向的含义；"
        "4) 性格倾向：1–2句，总结该面的性格走向。"
        "—— 面相部分需“拆解五官”，给出【眉/眼/鼻/嘴/颧或下巴】各1句具体特征，并基于易经作解读（映射到‘艮离兑乾坤震巽坎’之一），形成 meta.face_parts。"
        "然后给出："
        "5) 卦象组合：标题=三象卦名相加（如“艮 + 离 + 兑”），正文为4–6条要点（用短句），避免空泛；"
        "6) 总结性格印象：20–40字，必须结合三卦特征形成“独特且相关”的总结；"
        "7) 人格标签 archetype：必须根据三卦的主调自动生成（例如 乾+坤→“外刚内柔”，艮+离→“外稳内明” 等），禁止使用固定套话。"
        "将结果通过 submit_analysis_v3 工具返回，字段要求："
        "- summary：第6条“总结性格印象”；"
        "- archetype：第7条生成的人格标签；"
        "- sections：三象各压成一句中文（姿态/神情/面相）；"
        "- domains：仅从 ['金钱与事业','配偶与感情'] 选择；"
        "- meta.triple_analysis：含键'姿态','神情','面容','组合意境','总结'；每个三象含'说明','卦象','解读','性格倾向'；"
        "- meta.face_parts：键为'眉','眼','鼻','嘴','颧/下巴'，每个值含'特征','卦象','解读'；"
        "- meta.domains_detail：对'金钱与事业'与'配偶与感情'分别给出尽量“单行可读”的建议（各40–70字）。"
        "语言：中文。禁止输出除工具调用以外的任何自由文本。"
    )
    user = (
        "请严格按要求分析这张图片，避免模板化措辞。"
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


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
        temperature=0.4,  # 提高一点多样性
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
        {"role": "system", "content": "你必须通过函数 submit_analysis_v3 返回结果，严格符合 schema。不要直接输出文本。"}
    ]
    resp2 = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.35,
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


def _compose_auto_archetype(hexes: List[str]) -> str:
    # 依据三卦主调组合一个标签
    tags = [BAGUA_TRAITS.get(h, "") for h in hexes if h]
    tags = [t.split("/")[0] for t in tags if t]
    if not tags:
        return ""
    if len(tags) >= 2:
        return f"外{tags[0]}内{tags[1]}"
    return f"{tags[0]}取向"


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

    # 面相细分（透传）
    face_parts = meta.get("face_parts")
    if not isinstance(face_parts, dict):
        meta["face_parts"] = {}

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
    # archetype 若缺失则自动组合一个
    try:
        out["confidence"] = float(out.get("confidence", 0.0))
    except Exception:
        out["confidence"] = 0.0

    if not out.get("archetype"):
        ta2 = meta.get("triple_analysis") or {}
        hexes = [
            ta2.get("姿态", {}).get("卦象", ""),
            ta2.get("神情", {}).get("卦象", ""),
            ta2.get("面容", {}).get("卦象", ""),
        ]
        out["archetype"] = _compose_auto_archetype(hexes)

    # combo title
    ta2 = meta.get("triple_analysis") or {}
    hexes = [
        ta2.get("姿态", {}).get("卦象", ""),
        ta2.get("神情", {}).get("卦象", ""),
        ta2.get("面容", {}).get("卦象", ""),
    ]
    combo_title = " + ".join([h for h in hexes if h])
    if combo_title:
        meta["combo_title"] = combo_title

    # === UI helpers ===
    meta["headline"] = {"tag": out.get("archetype", ""), "confidence": out.get("confidence", 0.0)}

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

    # 组合要点：从性格倾向提要 + 组合意境
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
    meta["combo_detail"] = {"title": combo_full_title, "bullets": combo_points[:6]}

    # 总结 + 意境
    h1, h2, h3 = hexes + ["", "", ""][:max(0, 3 - len(hexes))]
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
                content={"error": he.detail, "debug": {"trace": traceback.format_exc()}},
            )
        raise
    except Exception as e:
        logger.exception("[ERROR] /upload failed: %s", e)
        body = {"error": "Internal Server Error"}
        if DEBUG:
            body["debug"] = {"message": str(e), "trace": traceback.format_exc()}
        return JSONResponse(status_code=500, content=body)
