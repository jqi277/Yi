# fastapi_app.py  (v3.5 → v3.5-ui-r2)
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
    # 保持你的旧逻辑，只强调“卦象组合 bullets/summary、卦名含象字、两段式总结”
    sys = (
      "你是 Selfy AI 的易经观相助手。必须先用“三象四段式”分析："
      "【姿态/神情/面容】三部分，每部分包含："
      "1) 说明：1句，描绘该面向的具体外观/动作/气质；"
      "2) 卦象：仅写一个卦名（如 艮/离/兑/乾/坤/震/巽/坎），必要时可写为“艮（山）”等；"
      "3) 解读：1–2句，解释该卦在此面向的含义；"
      "4) 性格倾向：1–2句，总结性格走向。"
      "然后给出："
      "5) 卦象组合：生成 meta.combo：gua_list（三卦名，不带括号）/ bullets（2–4条短句，如“外冷内热”“独立审美”“稳重理智”“交际选择性”）/ summary（40–80字）；"
      "   并生成 meta.combo_title='姿态卦 + 神情卦 + 面相卦'（不带括号）。"
      "6) 总结性格印象：两段式——"
      "   第一行：'这个人给人的感觉是：' 换行后一段带引号的总印象（30–50字）；"
      "   第二段：'在易经意境中，像是 “X” —— Y。'（20–40字）。"
      "将结果通过 submit_analysis_v3 工具返回，字段要求："
      "- summary：按第6条两段式；"
      "- archetype：意境化标签（如“外冷内热”等）；"
      "- sections：把三象各压成一句中文（姿态/神情/面相）；"
      "- domains：仅从 ['金钱与事业','配偶与感情'] 选择；"
      "- meta.triple_analysis：需包含键：'姿态','神情','面容','组合意境','总结'；"
      "  其中每个三象对象含：'说明','卦象','解读','性格倾向'；"
      "- meta.domains_detail：对'金钱与事业'与'配偶与感情'分别给出60–90字建议；"
      "语言：中文。禁止输出除工具调用以外的任何自由文本。"
      "八卦参考：艮=山，离=火，兑=泽，乾=天，坤=地，震=雷，巽=风，坎=水。"
    )
    user = (
        "请分析这张图片，返回严格符合 schema 的工具 JSON（含三象四段式、组合 bullets+summary、两段式总结、domains 建议）。"
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

    # 兜底重试
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

    raise RuntimeError("Model did not return tool_calls after forced attempt.")


def _join_cn(items: List[str]) -> str:
    items = [s for s in items if isinstance(s, str) and s.strip()]
    if not items:
        return ""
    return "、".join(items)


def _coerce_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    UI 适配（不破坏旧结构）：
    - meta.top_tag：只显示 archetype + confidence（给页面最上面那排）
    - meta.combo {gua_list, bullets, summary} + meta.combo_title
    - meta.sections_titles：'姿态 → 艮卦（山）' 这种标题
    - meta.summary_rich {impression, imagery}；summary 写成两段合并文本
    - 保留旧 sections 一句话；domains/domains_detail 兼容
    """
    allowed_domains = {"金钱与事业", "配偶与感情"}
    GUA_TO_XIANG = {"乾":"天","坤":"地","震":"雷","巽":"风","坎":"水","离":"火","艮":"山","兑":"泽"}

    out = dict(data) if isinstance(data, dict) else {}
    meta = out.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    out["meta"] = meta

    # ---------- sections 一句话（优先 triple_analysis） ----------
    sections = out.get("sections") or {}
    if not isinstance(sections, dict):
        sections = {}

    ta = meta.get("triple_analysis")
    if not isinstance(ta, dict):
        ta = {}

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

    # ---------- 顶部 tag ----------
    out["archetype"] = out.get("archetype") or "外冷内热"
    try:
        out["confidence"] = float(out.get("confidence", out.get("可信度", 0.88)))
    except Exception:
        out["confidence"] = 0.88
    meta["top_tag"] = {"personality_tag": out["archetype"], "confidence": out["confidence"]}

    # ---------- 分象标题：'姿态 → 艮卦（山）' ----------
    def _core_gua(g: str) -> str:
        if not isinstance(g, str): return ""
        return g.replace("（","(").split("(")[0].strip()

    def _title_of(seg_key: str, ta_key: str) -> str:
        g = _core_gua((ta.get(ta_key) or {}).get("卦象",""))
        if g and g in GUA_TO_XIANG:
            return f"{seg_key} → {g}卦（{GUA_TO_XIANG[g]}）"
        return seg_key

    meta["sections_titles"] = {
        "姿态": _title_of("姿态", "姿态"),
        "神情": _title_of("神情", "神情"),
        "面相": _title_of("面相", "面容"),
    }

    # ---------- 卦象组合（bullets + summary） ----------
    combo = meta.get("combo")
    if not isinstance(combo, dict):
        combo = {}

    def _safe_list(x):
        return x if isinstance(x, list) else []

    gua_list = _safe_list(combo.get("gua_list"))
    if not gua_list:
        gl = [
            _core_gua((ta.get("姿态") or {}).get("卦象","")),
            _core_gua((ta.get("神情") or {}).get("卦象","")),
            _core_gua((ta.get("面容") or {}).get("卦象","")),
        ]
        gua_list = [g for g in gl if g]

    bullets = _safe_list(combo.get("bullets"))
    if not bullets:
        bullets = ["外冷内热","独立审美","稳重理智","交际选择性"]

    combo_summary = (combo.get("summary") or "").strip() or "这种组合显示：外表克制沉稳，内心保有热度与理想；对人际更重深度与真诚连接。"

    meta["combo"] = {"gua_list": gua_list, "bullets": bullets, "summary": combo_summary}
    meta["combo_title"] = " + ".join([g for g in gua_list if g])

    # ---------- 两段式总结 ----------
    impression = "这个人给人的感觉是：\n“外在沉稳冷艳，内心热情坚定，重视自我独立与美感，对人际关系有选择性。”"
    # 用卦映射成意境，如“山中有火，火映泽面”
    xiangs = [GUA_TO_XIANG.get(x, "") for x in gua_list]
    xiangs = [x for x in xiangs if x]
    if len(xiangs) >= 2:
        imagery = f"在易经意境中，像是 “{xiangs[0]}中有{xiangs[1]}，{xiangs[1]}映{xiangs[0]}面” —— 内藏光芒，择人而耀。"
    else:
        imagery = "在易经意境中，像是 “山中有火，火映泽面” —— 内藏光芒，择人而耀。"
    meta["summary_rich"] = {"impression": impression, "imagery": imagery}
    out["summary"] = impression + "\n" + imagery

    # ---------- domains / domains_detail ----------
    allowed = {"金钱与事业", "配偶与感情"}
    domains = out.get("domains")
    if isinstance(domains, dict):
        domain_keys = [k for k in domains.keys() if k in allowed]
        out["domains"] = domain_keys
        meta["domains_detail"] = {k: domains[k] for k in domain_keys}
    elif isinstance(domains, list):
        out["domains"] = [d for d in domains if d in allowed]
    else:
        out["domains"] = []

    dd = meta.get("domains_detail")
    if isinstance(dd, str):
        try:
            dd = json.loads(dd)
        except Exception:
            dd = {}
    if not isinstance(dd, dict):
        dd = {}

    def _ensure_heavy(key: str):
        txt = (dd.get(key) or "").strip()
        if len(txt) < 140:
            if key == "金钱与事业":
                dd[key] = (
                    "稳重理智，适合承担重要任务。建议：①小步试错+两周复盘，固定记录与SOP迭代；"
                    "②设置对外协作位（技术/渠道/财务其一），每周30分钟沟通同步。"
                    "预警：若出现一次性押注或长期闭环不汇报，立即缩小试错规模并公开里程碑。"
                )
            else:
                dd[key] = (
                    "外冷内热，重边界与真诚。建议：①每周一次30分钟节律沟通，只谈事实—感受—需求；"
                    "②重要信息4小时内先回应，晚些再细聊。"
                    "预警：若连续两次回避表达或过度理性化，对方会感到疏离，需要以共享计划或情感回应补偿。"
                )
        if key not in out["domains"]:
            out["domains"].append(key)

    _ensure_heavy("金钱与事业")
    _ensure_heavy("配偶与感情")
    meta["domains_detail"] = dd

    out["meta"] = meta
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
