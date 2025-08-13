# fastapi_app.py  (runtime v3.7.10, analysis logic v3.7.2 + post-processing)
import os, base64, json, logging, traceback, re
from typing import Dict, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI

RUNTIME_VERSION = "3.7.10"
ANALYSIS_VERSION = os.getenv("ANALYSIS_VERSION", "372").strip()
SCHEMA_ID = "selfy.v3"
DEBUG = str(os.getenv("DEBUG","0")).strip() in ("1","true","True","YES","yes")

logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("selfy-ai")

app = FastAPI(title="Selfy AI - YiJing Analysis API", version=RUNTIME_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

try:
    client = OpenAI()
except Exception as e:
    logger.error("OpenAI client init failed: %s", e); client=None

BAGUA_SYMBOLS = {"艮":"山","离":"火","兑":"泽","乾":"天","坤":"地","震":"雷","巽":"风","坎":"水"}

# ---------------- helpers ----------------
def _to_data_url(content: bytes, content_type: str) -> str:
    return f"data:{content_type};base64,{base64.b64encode(content).decode('utf-8')}"

def _build_tools_schema() -> List[Dict[str, Any]]:
    return [{
      "type":"function",
      "function":{
        "name":"submit_analysis_v3",
        "description":"Return end-user facing JSON for Selfy AI YiJing analysis.",
        "parameters":{
          "type":"object",
          "properties":{
            "summary":{"type":"string"},
            "archetype":{"type":"string"},
            "confidence":{"type":"number"},
            "sections":{"type":"object","properties":{"姿态":{"type":"string"},"神情":{"type":"string"},"面相":{"type":"string"}},"required":["姿态","神情","面相"],"additionalProperties":False},
            "domains":{"type":"array","items":{"type":"string"}},
            "meta":{"type":"object","additionalProperties":True}
          },
          "required":["summary","archetype","confidence","sections","domains"],
          "additionalProperties":False
        }
      }
    }]

def _json_hint() -> str:
    return ("只以 JSON object 返回（必须 JSON）。示例:{\"summary\":\"…\",\"archetype\":\"…\",\"confidence\":0.9,"
            "\"sections\":{\"姿态\":\"…\",\"神情\":\"…\",\"面相\":\"…\"},"
            "\"domains\":[\"金钱与事业\",\"配偶与感情\"],"
            "\"meta\":{\"triple_analysis\":{\"姿态\":{\"说明\":\"…\",\"卦象\":\"艮\",\"解读\":\"…\",\"性格倾向\":\"…\"},\"神情\":{…},\"面容\":{…},\"组合意境\":\"…\",\"总结\":\"…\"},"
            "\"face_parts\":{\"眉\":{\"特征\":\"…\",\"卦象\":\"…\",\"解读\":\"…\"},\"眼\":{…},\"鼻\":{…},\"嘴\":{…},\"颧/下巴\":{…}},"
            "\"domains_detail\":{\"金钱与事业\":\"…(60–90字)\",\"配偶与感情\":\"…(60–90字)\"}}}")

def _prompt_for_image_v372():
    sys = (
      "你是 Selfy AI 的易经观相助手（v3.7.2 风格）。"
      "严格按“三象四段式”分析：【姿态/神情/面容】三部分。每部分必须包含："
      "1) 说明：1句，客观描绘外观/动作/气质；"
      "2) 卦象：仅写一个卦名（艮/离/兑/乾/坤/震/巽/坎）；"
      "3) 解读：1–2句，基于卦象与观察做含义阐释；"
      "4) 性格倾向：1–2句，独立成段，不要与“解读”重复措辞。"
      "然后给出："
      "5) 卦象组合：标题=三象卦名相加（如“艮 + 离 + 兑”），正文 90–150 字（可与三象结论适度重合）；"
      "6) 总结性格印象：20–40字，避免模板化；"
      "7) 人格标签 archetype：2–5字中文，如“外冷内热/主导型/谨慎型”。"
      "面相需拆成五官：在 meta.face_parts 中，给【眉/眼/鼻/嘴/颧/下巴】（任选5项覆盖）各写“特征（外观）”与“解读（基于易经）”。"
      "domains 仅从 ['金钱与事业','配偶与感情'] 选择；在 meta.domains_detail 中分别写 60–90 字建议文本。"
      "将结果通过 submit_analysis_v3 工具返回，并"+_json_hint()+"。语言：中文。本消息含“JSON”以满足 API 要求。"
    )
    user = "请按 3.7.2 风格分析图片，严格通过函数返回 JSON（不要输出自由文本）。"
    return [{"role":"system","content":sys},{"role":"user","content":user}]

def _call_openai(messages):
    return client.chat.completions.create(
        model="gpt-4o",
        temperature=0.4,
        tools=_build_tools_schema(),
        tool_choice={"type":"function","function":{"name":"submit_analysis_v3"}},
        response_format={"type":"json_object"},
        messages=messages
    )

# ---------------- text post-processing ----------------
def _clean_punct(s: str) -> str:
    if not s: return s
    s = s.replace("——", "，")
    s = re.sub(r"[；;]+\s*", "；", s)          # 压缩连续分号
    s = re.sub(r"[。\.]{2,}", "。", s)        # 压缩连续句号
    s = re.sub(r"\s*；\s*", "；", s)          # 分号两侧空格
    s = re.sub(r"(，|、)\\1+", r"\\1", s)     # 压缩连续逗号/顿号
    s = re.sub(r"；\s*$", "。", s)            # 句尾分号→句号
    return s.strip()

def _dedup_sentence_fragments(s: str) -> str:
    if not s: return s
    parts = re.split(r"[。；]", s)
    seen = set(); filtered = []
    for p in parts:
        p = p.strip()
        if not p: continue
        if p in seen: 
            continue
        seen.add(p); filtered.append(p)
    s = "。".join(filtered) + "。"
    clauses = [c.strip() for c in re.split(r"[，,]", s) if c.strip()]
    uniq = []; seen2 = set()
    for c in clauses:
        if c not in seen2:
            seen2.add(c); uniq.append(c)
    return "，".join(uniq).rstrip("，").replace("。。","。").rstrip("；") + "。"

def _merge_observe_and_interp(desc: str, interp: str) -> str:
    desc = (desc or "").strip().rstrip("；。")
    interp = (interp or "").strip().rstrip("；。")
    if not desc and not interp: return ""
    s = f"{desc}，{interp}" if desc and interp else (desc or interp)
    return _clean_punct(_dedup_sentence_fragments(s))

def _hex_meaning(h: str) -> List[str]:
    m = {
        "乾": ["自信","主导","果断"],
        "坤": ["包容","稳重","承载"],
        "兑": ["亲和","交流","悦人"],
        "离": ["明晰","表达","洞察"],
        "艮": ["稳守","边界","定力"],
        "震": ["行动","突破","启动"],
        "巽": ["协商","渗透","整合"],
        "坎": ["谨慎","深度","风险意识"],
    }
    return m.get(h, [])

def _compose_combo_summary(hexes: List[str], traits: List[str]) -> str:
    kws = []
    for h in hexes:
        kws += _hex_meaning(h)
    # 去重保序
    seen = set(); kws2 = []
    for k in kws:
        if k not in seen:
            seen.add(k); kws2.append(k)
    head = "、".join(kws2[:6]) or "综合均衡"
    tail = "；".join([t for t in traits if t][:2])  # 少量融合
    sent = f"整体气质以{head}为主"
    if tail:
        sent += f"；{tail}"
    sent += "。"
    return _clean_punct(sent)

def _merge_status_and_detail(status: str, detail: str) -> str:
    first = ""
    if detail:
        first = detail.split("。")[0].strip()
        if re.match(r"^(建议|可|宜|尽量|避免|优先|不妨|尝试|保持|加强|明确|制定|多与|多用|提前|关注|留意|主动)", first):
            first = ""
    chunks = [ch for ch in [status, first] if ch]
    s = "；".join(chunks)
    return _clean_punct(_dedup_sentence_fragments(s))

def _imperative_suggestion(detail: str) -> str:
    if not detail: return ""
    sents = [x.strip() for x in re.split(r"[。;；]", detail) if x.strip()]
    sugg = []
    for s in sents:
        if re.match(r"^(建议|可|宜|尽量|避免|优先|不妨|尝试|保持|加强|明确|制定|多与|多用|提前|关注|留意|主动)", s):
            sugg.append(s)
    if not sugg:
        s = detail.strip()
        s = re.sub(r"(可以考虑|可以|适合|能够|可能会|需要|应当|有助于)", "建议", s)
        sugg = [s]
    text = "；".join(sugg)
    return _clean_punct(_dedup_sentence_fragments(text))

def _sanitize_block_text(s: str) -> str:
    return _clean_punct(_dedup_sentence_fragments(s or ""))

def _inflate_dotted_keys(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict): return obj
    out: Dict[str, Any] = {}
    for k,v in obj.items():
        if "." not in k:
            out[k] = _inflate_dotted_keys(v) if isinstance(v, dict) else v
    for k,v in obj.items():
        if isinstance(k, str) and "." in k:
            head, tail = k.split(".", 1)
            base = out.setdefault(head, {})
            if not isinstance(base, dict): base = {}; out[head] = base
            cur = base
            parts = tail.split(".")
            for i, p in enumerate(parts):
                if i == len(parts)-1:
                    cur[p] = v
                else:
                    cur = cur.setdefault(p, {})
    for k in list(out.keys()):
        if isinstance(out[k], dict):
            out[k] = _inflate_dotted_keys(out[k])
    return out

# ---------------- main coercer ----------------
def _coerce_output_v372(data: Dict[str,Any]) -> Dict[str,Any]:
    data = _inflate_dotted_keys(data)
    out = dict(data)
    meta = out.get("meta") or {}
    if not isinstance(meta, dict): meta = {}
    out["meta"] = meta

    ta = meta.get("triple_analysis") or {}
    # 合并说明+解读为一句，并清理重复标点；同时收集“性格倾向”
    traits = []
    for key in ["姿态","神情","面容"]:
        block = (ta.get(key) or {}).copy()
        if not isinstance(block, dict): block = {}
        desc = block.get("说明","")
        interp = block.get("解读","")
        merged = _merge_observe_and_interp(desc, interp)
        tend = (block.get("性格倾向") or "").strip()
        if tend: traits.append(tend)
        block["解读"] = merged
        ta[key] = block
    meta["triple_analysis"] = ta

    # 组合卦
    hexes = [(ta.get("姿态") or {}).get("卦象",""),
             (ta.get("神情") or {}).get("卦象",""),
             (ta.get("面容") or {}).get("卦象","")]
    combo_title = " + ".join([h for h in hexes if h]) if any(hexes) else ""
    if combo_title:
        meta["combo_title"] = combo_title

    raw_summary = (ta.get("总结") or out.get("summary","")).strip()
    combo_summary = _compose_combo_summary(hexes, traits)
    if raw_summary:
        raw_summary = _sanitize_block_text(raw_summary)
        if re.search(r"(姿态|神情|面容|双手|目光|五官|眉|眼|鼻|嘴|下巴)", raw_summary):
            overview_text = combo_summary
        else:
            overview_text = _sanitize_block_text(raw_summary + "；" + combo_summary)
    else:
        overview_text = combo_summary
    meta["overview_card"] = {
        "title": f"🔮 卦象组合：{combo_title}" if combo_title else "🔮 卦象组合",
        "summary": overview_text
    }

    # 中文 archetype 兜底
    arch = (out.get("archetype") or "").strip()
    if arch and not any('\u4e00' <= ch <= '\u9fff' for ch in arch):
        s = set([h for h in hexes if h])
        if "乾" in s and "兑" in s: arch = "主导·亲和型"
        elif "乾" in s and "离" in s: arch = "主导·表达型"
        elif "艮" in s and "坤" in s: arch = "稳重·包容型"
        elif "坎" in s and "离" in s: arch = "谨慎·表达型"
        elif "震" in s and "兑" in s: arch = "行动·亲和型"
        else: arch = "综合型"
        out["archetype"] = arch

    # 事业/感情：状态与建议
    def_map = meta.get("domains_detail") or {}
    status_map = _insight_for_domains(hexes)
    merged_status = {
        "事业": _merge_status_and_detail(status_map.get("事业",""), def_map.get("金钱与事业","")),
        "感情": _merge_status_and_detail(status_map.get("感情",""), def_map.get("配偶与感情","")),
    }
    meta["domains_status"] = {
        "事业": _sanitize_block_text(merged_status["事业"]),
        "感情": _sanitize_block_text(merged_status["感情"]),
    }
    meta["domains_suggestion"] = {
        "事业": _sanitize_block_text(_imperative_suggestion(def_map.get("金钱与事业",""))),
        "感情": _sanitize_block_text(_imperative_suggestion(def_map.get("配偶与感情",""))),
    }

    # 五官细节兜底清理：分号→句号
    fp = meta.get("face_parts") or {}
    for part in fp:
        for sub in ["特征","解读"]:
            if isinstance(fp[part], dict) and fp[part].get(sub):
                fp[part][sub] = _clean_punct(fp[part][sub]).replace("；","。")
    meta["face_parts"] = fp

    # 可信度兜底
    try:
        out["confidence"] = float(out.get("confidence",0.0))
    except Exception:
        out["confidence"] = 0.0
    meta["headline"] = {"tag": out.get("archetype",""), "confidence": out["confidence"]}

    out["meta"] = meta
    return out

# ---------------- routes ----------------
@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/", include_in_schema=False)
def root():
    return HTMLResponse("<h3>Selfy AI</h3><a href='/docs'>/docs</a> · <a href='/mobile'>/mobile</a>")

@app.head("/", include_in_schema=False)
def root_head():
    return Response(status_code=200)

@app.get("/version")
def version(): return {"runtime":RUNTIME_VERSION,"analysis":ANALYSIS_VERSION,"schema":SCHEMA_ID,"debug":DEBUG}

@app.get("/mobile", include_in_schema=False)
def mobile():
    path = os.path.join(os.path.dirname(__file__), "index_mobile.html")
    try:
        html = open(path, "r", encoding="utf-8").read()
    except Exception as e:
        return HTMLResponse(f"<pre>index_mobile.html not found: {e}</pre>", status_code=500)
    return HTMLResponse(html)

def _call_gpt(messages):
    if client is None:
        raise RuntimeError("OpenAI client not initialized")
    return client.chat.completions.create(
        model="gpt-4o",
        temperature=0.4,
        tools=_build_tools_schema(),
        tool_choice={"type":"function","function":{"name":"submit_analysis_v3"}},
        response_format={"type":"json_object"},
        messages=messages
    )

def _call_gpt_tool_with_image(data_url: str) -> Dict[str,Any]:
    messages = _prompt_for_image_v372()
    messages[-1]["content"] = [
        {"type":"text","text":messages[-1]["content"]},
        {"type":"image_url","image_url":{"url":data_url}}
    ]
    resp = _call_gpt(messages)
    choice = resp.choices[0]
    tool_calls = getattr(choice.message, "tool_calls", None)
    if tool_calls:
        args = json.loads(tool_calls[0].function.arguments)
    else:
        content = getattr(choice.message, "content", None)
        if isinstance(content, str) and content.strip().startswith("{"):
            args = json.loads(content)
        else:
            raise RuntimeError("Model did not return tool_calls.")
    return {"tool_args": args, "oai_raw": resp if DEBUG else None}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        if not file: raise HTTPException(400,"No file")
        ct = file.content_type or ""
        if not ct.startswith("image/"): raise HTTPException(415,f"Unsupported content type: {ct}")
        raw = await file.read()
        if not raw: raise HTTPException(400,"Empty file")
        if len(raw) > 15*1024*1024: raise HTTPException(413,"File too large (>15MB)")

        data_url = _to_data_url(raw, ct)
        logger.info("[UPLOAD] %s %dB %s", file.filename, len(raw), ct)

        result = _call_gpt_tool_with_image(data_url)
        tool_args = result["tool_args"]
        final_out = _coerce_output_v372(tool_args)

        if DEBUG:
            meta = final_out.setdefault("meta",{}).setdefault("debug",{})
            meta["file_info"]={"filename":file.filename,"content_type":ct,"size":len(raw)}
            try:
                meta["oai_choice_finish_reason"]=result["oai_raw"].choices[0].finish_reason
            except Exception:
                meta["oai_choice_finish_reason"]="n/a"

        return JSONResponse(content=final_out, status_code=200)
    except HTTPException as he:
        if DEBUG:
            return JSONResponse(status_code=he.status_code, content={"error":he.detail,"debug":{"trace":traceback.format_exc()}})
        raise
    except Exception as e:
        logging.exception("upload failed: %s", e)
        body={"error":"Internal Server Error"}
        if DEBUG: body["debug"]={"message":str(e),"trace":traceback.format_exc()}
        return JSONResponse(status_code=500, content=body)
