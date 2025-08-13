# fastapi_app.py  (runtime v3.7.6, analysis logic v3.7.2, with /mobile route, status insights)
import os, base64, json, logging, traceback
from typing import Dict, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI

RUNTIME_VERSION = "3.7.6"
ANALYSIS_VERSION = os.getenv("ANALYSIS_VERSION", "372").strip()  # default 372
SCHEMA_ID = "selfy.v3"
DEBUG = str(os.getenv("DEBUG","0")).strip() in ("1","true","True","YES","yes")

logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("selfy-ai")

app = FastAPI(title="Selfy AI - YiJing Analysis API", version=RUNTIME_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# OpenAI client
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
    # 包含“JSON”字样以满足 response_format=json_object 的要求
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

def _call_openai(messages):
    return client.chat.completions.create(
        model="gpt-4o",
        temperature=0.4,
        tools=_build_tools_schema(),
        tool_choice={"type":"function","function":{"name":"submit_analysis_v3"}},
        response_format={"type":"json_object"},
        messages=messages
    )

def _insight_for_domains(hexes: List[str]) -> Dict[str, str]:
    s = set([h for h in hexes if h])
    biz = []
    if "乾" in s or "震" in s: biz.append("推进力强、目标感明确")
    if "坤" in s or "艮" in s: biz.append("稳健务实、执行到位")
    if "离" in s or "兑" in s: biz.append("表达协作顺畅、善于影响")
    if "坎" in s: biz.append("风险意识较强、节奏更稳")
    if "巽" in s: biz.append("擅协调资源、善整合")
    love = []
    if "兑" in s: love.append("互动亲和、沟通自然")
    if "坤" in s: love.append("重承诺与包容")
    if "离" in s: love.append("表达清晰、善于共情")
    if "坎" in s: love.append("安全感需求偏高、较敏感")
    if "震" in s or "乾" in s: love.append("主动靠近、决断力较强")
    return {"事业": "；".join(biz), "感情": "；".join(love)}

def _synthesize_titles(ta: Dict[str, Any]) -> Dict[str,str]:
    # 仍提供，但前端不再使用此拼接形式
    def _title(section: str, key: str) -> str:
        hx = (ta.get(key) or {}).get("卦象","")
        sym = BAGUA_SYMBOLS.get(hx,"")
        return f"{section} → {hx}卦（{sym}）" if hx else section
    return {"姿态":_title("姿态","姿态"), "神情":_title("神情","神情"), "面相":_title("面相","面容")}

def _coerce_output_v372(data: Dict[str,Any]) -> Dict[str,Any]:
    data = _inflate_dotted_keys(data)
    out = dict(data)
    meta = out.get("meta") or {}
    if not isinstance(meta, dict): meta = {}
    out["meta"] = meta

    ta = meta.get("triple_analysis") or {}

    # 组合卦
    hexes = [(ta.get("姿态") or {}).get("卦象",""),
             (ta.get("神情") or {}).get("卦象",""),
             (ta.get("面容") or {}).get("卦象","")]
    combo_title = " + ".join([h for h in hexes if h])
    if combo_title:
        meta["combo_title"] = combo_title

    # 卦象组合卡：优先采用 ta["总结"] 作为“一段式总述”
    one = (ta.get("总结") or out.get("summary",""))
    meta["overview_card"] = {
        "title": f"🔮 卦象组合：{combo_title}" if combo_title else "🔮 卦象组合",
        "summary": one
    }

    # 三象标题（带卦）供前端备用
    meta["sections_titles"] = _synthesize_titles(ta)

    # 标签中文兜底
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

    # 新增：近期状态
    meta["domains_status"] = _insight_for_domains(hexes)

    try:
        out["confidence"] = float(out.get("confidence",0.0))
    except Exception:
        out["confidence"] = 0.0
    meta["headline"] = {"tag": out["archetype"], "confidence": out["confidence"]}

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

def _call_openai(messages):
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
    resp = _call_openai(messages)
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
