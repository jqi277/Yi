# fastapi_app.py  (v3.7.5 - mobile route)
import os, base64, json, logging, traceback
from typing import Dict, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI

VERSION = "3.7.5"
SCHEMA_ID = "selfy.v3"
DEBUG = str(os.getenv("DEBUG","0")).strip() in ("1","true","True","YES","yes")

logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("selfy-ai")

app = FastAPI(title="Selfy AI - YiJing Analysis API", version=VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

try:
    client = OpenAI()
except Exception as e:
    logger.error("OpenAI client init failed: %s", e); client=None

BAGUA_SYMBOLS = {"艮":"山","离":"火","兑":"泽","乾":"天","坤":"地","震":"雷","巽":"风","坎":"水"}

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

def _json_hint():
    return ("只以 JSON object 返回（必须 JSON）。示例:{\"summary\":\"…\",\"archetype\":\"…\",\"confidence\":0.9,"
            "\"sections\":{\"姿态\":\"…\",\"神情\":\"…\",\"面相\":\"…\"},"
            "\"domains\":[\"金钱与事业\",\"配偶与感情\"],"
            "\"meta\":{\"triple_analysis\":{\"姿态\":{\"说明\":\"…\",\"卦象\":\"艮\",\"解读\":\"…（将性格倾向自然融入解读内）\"},\"神情\":{…},\"面容\":{…},\"组合意境\":\"…\",\"总结\":\"…\"},"
            "\"face_parts\":{\"眉\":{\"特征\":\"…\",\"卦象\":\"…\",\"解读\":\"…\"},\"眼\":{…},\"鼻\":{…},\"嘴\":{…},\"颧/下巴\":{…}},"
            "\"domains_detail\":{\"金钱与事业\":\"…\",\"配偶与感情\":\"…\"}}}")

def _prompt_for_image():
    sys = (
      "你是 Selfy AI 的易经观相助手。"
      "严格用“三象四段式”分析：【姿态/神情/面容】。每部分含：说明(1句)；卦象(艮/离/兑/乾/坤/震/巽/坎)；解读(1–3句)；性格倾向(1–2句)。"
      "重要：在输出时，把“性格倾向”自然地**融入解读**中（解读可相应加长），前端不单独展示“性格倾向”。"
      "面相必须拆解五官：给【眉/眼/鼻/嘴/颧或下巴】各1句具体特征，并为每项标注一个卦象并解读，写入 meta.face_parts。"
      "【避免重复】“解读”不得简单复述“特征”的字词；如“神情”已描述某五官动态/风格，则“面相-五官”应换角度（形态/比例/纹理/功能感等）。"
      "然后：5) 卦象组合：基于三卦“综合推理”写 4–6 条要点（不得逐字重复三象原句）；"
      "6) 总结性格印象：20–40字，与三卦强相关，避免模板化；"
      "7) 人格标签 archetype：中文标签；若内部推导是英文，也要给出中文意境词。"
      "明令禁止：出现“五官端正/整体面容和谐/面容和谐”等套话。"
      "将结果通过 submit_analysis_v3 工具返回，并"+_json_hint()+"。语言：中文。本消息含“JSON”以满足 API 要求。"
    )
    user = "请严格按要求分析图片，并只以 JSON 格式通过函数返回。"
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
        temperature=0.45,
        tools=_build_tools_schema(),
        tool_choice={"type":"function","function":{"name":"submit_analysis_v3"}},
        response_format={"type":"json_object"},
        messages=messages
    )

def _synthesize_combo(ta: Dict[str, Any]):
    hexes = [
        (ta.get("姿态") or {}).get("卦象",""),
        (ta.get("神情") or {}).get("卦象",""),
        (ta.get("面容") or {}).get("卦象","")
    ]
    traits_map = {"艮":"稳重","离":"表达","兑":"亲和","乾":"主导","坤":"包容","震":"行动","巽":"协调","坎":"谨慎"}
    bullets: List[str] = []
    traits = [traits_map.get(h,"") for h in hexes if h]
    if len(traits) >= 2:
        bullets.append(f"外在表现偏{traits[0]}，内在驱动更{traits[1]}。")
    if "兑" in hexes: bullets.append("沟通风格亲和而直接，重视真实与愉悦的互动。")
    if "坎" in hexes: bullets.append("决策前会评估风险与后果，偏稳健。")
    if "震" in hexes: bullets.append("遇事行动果断，推进节奏快。")
    if "离" in hexes: bullets.append("表达清晰，擅长信息提炼与呈现。")
    if "乾" in hexes: bullets.append("具备主导性与目标感，愿意承担责任。")
    if "坤" in hexes: bullets.append("处事包容稳妥，善于托底与承载团队。")
    if "艮" in hexes: bullets.append("有边界感与秩序感，做事沉稳可靠。")
    if "巽" in hexes: bullets.append("倾向协商与整合资源，善做协调者。")
    seen=set(); out=[]
    for b in bullets:
        if b not in seen:
            seen.add(b); out.append(b)
        if len(out)>=5: break
    return hexes, out

def _insight_for_domains(hexes: List[str]):
    sets = set(hexes); lines: Dict[str,str] = {}
    segs = []
    if "乾" in sets or "震" in sets: segs.append("具推进力与目标感")
    if "坤" in sets or "艮" in sets: segs.append("稳健度与执行力兼备")
    if "离" in sets or "兑" in sets: segs.append("擅表达与协作")
    if "坎" in sets: segs.append("风险意识较强")
    if "巽" in sets: segs.append("善于协调资源")
    lines["事业"] = "；".join(segs) if segs else "以稳中求进为主，兼顾沟通与执行。"
    segs = []
    if "兑" in sets: segs.append("互动亲和")
    if "坤" in sets: segs.append("重承诺与包容")
    if "离" in sets: segs.append("表达明确")
    if "坎" in sets: segs.append("安全感需求较高")
    if "震" in sets or "乾" in sets: segs.append("主动追求与决断")
    lines["感情"] = "；".join(segs) if segs else "重视稳定关系，沟通直接。"
    return lines

def _archetype_cn_fallback(archetype: str, hexes: List[str]) -> str:
    if archetype and any('\u4e00' <= ch <= '\u9fff' for ch in archetype):
        return archetype
    has = lambda h: h in hexes
    if has("乾") and has("兑"): return "主导·亲和型"
    if has("乾") and has("离"): return "主导·表达型"
    if has("艮") and has("坤"): return "稳重·包容型"
    if has("坎") and has("离"): return "谨慎·表达型"
    if has("震") and has("兑"): return "行动·亲和型"
    return archetype or "综合型"

def _coerce_output(data: Dict[str,Any]) -> Dict[str,Any]:
    data = _inflate_dotted_keys(data)
    out = dict(data)
    meta = out.get("meta") or {}
    if not isinstance(meta, dict): meta = {}
    out["meta"] = meta

    ta = meta.get("triple_analysis") or {}
    hexes, bullets = _synthesize_combo(ta)
    combo_title = " + ".join([h for h in hexes if h])
    if combo_title: meta["combo_title"] = combo_title
    meta["overview_card"] = {
        "title": f"🔮 卦象组合：{combo_title}" if combo_title else "🔮 卦象组合",
        "summary": out.get("summary",""),
        "bullets": bullets
    }

    raw_dd = meta.get("domains_detail") or {}
    status = _insight_for_domains(hexes)
    meta["domains_status"] = {"事业": status.get("事业",""), "感情": status.get("感情","")}
    def _expand(txt, fallback):
        if not isinstance(txt, str) or len(txt) < 80:
            return (fallback or "") + " 倾向将优势场景与风险点成对管理：用优势覆盖关键节点，同时设置检查点与反馈机制，以保证节奏与质量。"
        return txt
    meta["domains_detail_long"] = {
        "事业": _expand(raw_dd.get("金钱与事业",""), "在事业中建议把主导性与稳健度结合，先定清晰目标与边界，再逐步推进"),
        "感情": _expand(raw_dd.get("配偶与感情",""), "在关系中保持真诚表达与稳固承诺，关注对方节奏与需求差异，营造可预期的安全感")
    }

    def _title_with_hex(section_key: str, ta_key: str) -> str:
        hexname = (ta.get(ta_key) or {}).get("卦象","")
        symbol = {"艮":"山","离":"火","兑":"泽","乾":"天","坤":"地","震":"雷","巽":"风","坎":"水"}.get(hexname,"")
        return f"{section_key} → {hexname}卦（{symbol}）" if hexname and symbol else (f"{section_key} → {hexname}卦" if hexname else section_key)
    meta["sections_titles"] = {
        "姿态": _title_with_hex("姿态","姿态"),
        "神情": _title_with_hex("神情","神情"),
        "面相": _title_with_hex("面相","面容")
    }

    out["archetype"] = _archetype_cn_fallback(out.get("archetype",""), hexes)
    try: out["confidence"] = float(out.get("confidence",0.0))
    except Exception: out["confidence"] = 0.0
    meta["headline"] = {"tag": out["archetype"], "confidence": out["confidence"]}

    out["meta"] = meta
    return out

@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/", include_in_schema=False)
def root():
    return HTMLResponse("<h3>Selfy AI</h3><a href='/docs'>/docs</a> · <a href='/mobile'>/mobile</a>")

@app.head("/", include_in_schema=False)
def root_head():
    return Response(status_code=200)

@app.get("/version")
def version(): return {"version":VERSION,"schema":SCHEMA_ID,"debug":DEBUG}

def _call_gpt_tool_with_image(data_url: str) -> Dict[str,Any]:
    if client is None: raise RuntimeError("OpenAI client not initialized")
    messages = _prompt_for_image()
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
        final_out = _coerce_output(tool_args)

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

# Serve mobile page at /mobile
@app.get("/mobile", include_in_schema=False)
def mobile():
    try:
        html = open("index_mobile.html", "r", encoding="utf-8").read()
    except Exception:
        html = "<h4>index_mobile.html not found</h4>"
    return HTMLResponse(html)