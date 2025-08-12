
# fastapi_app.py  (v3.7, clean build)
import os, base64, json, logging, traceback, statistics
from typing import Dict, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI

VERSION = "3.7"
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
BAGUA_TRAITS = {"艮":"稳重/定界","离":"明晰/表达","兑":"亲和/交流","乾":"自信/主导","坤":"包容/承载","震":"果断/行动","巽":"圆融/协商","坎":"谨慎/深思"}
GENERIC_FACE_LINES = [
  "五官端正，整体面容和谐", "整体面容协调", "五官比例协调", "整体给人亲和稳定的感觉"
]

def _to_data_url(content: bytes, content_type: str) -> str:
    return f"data:{content_type};base64,{base64.b64encode(content).decode('utf-8')}"

def _build_tools_schema():
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
            "meta":{"type":"object","additionalProperties":True},
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
            "\"meta\":{\"triple_analysis\":{\"姿态\":{\"说明\":\"…\",\"卦象\":\"艮\",\"解读\":\"…\",\"性格倾向\":\"…\"},\"神情\":{…},\"面容\":{…},\"组合意境\":\"…\",\"总结\":\"…\"},"
            "\"face_parts\":{\"眉\":{\"特征\":\"…\",\"卦象\":\"…\",\"解读\":\"…\"},\"眼\":{…},\"鼻\":{…},\"嘴\":{…},\"颧/下巴\":{…}},"
            "\"domains_detail\":{\"金钱与事业\":\"…\",\"配偶与感情\":\"…\"}}}")

def _prompt_for_image():
    sys = (
      "你是 Selfy AI 的易经观相助手。"
      "严格用“三象四段式”分析：【姿态/神情/面容】。每部分含：说明(1句)；卦象(艮/离/兑/乾/坤/震/巽/坎)；解读(1–2句)；性格倾向(1–2句)。"
      "面相必须拆解五官：给【眉/眼/鼻/嘴/颧或下巴】各1句具体特征，并为每项标注一个卦象并解读，写入 meta.face_parts。"
      "然后：5) 卦象组合：基于三卦“综合推理”写 4–6 条要点（不得逐字重复三象原文；要合成新的洞见，如外在呈现/内在驱动/沟通风格/决策风格/风险偏好等）；"
      "6) 总结性格印象：20–40字，必须与三卦强相关，避免模板化；"
      "7) 人格标签 archetype：根据三卦主调自动生成（避免“外柔内刚/外冷内热”等陈词，如确需类似结构请替换为更具体的词）。"
      "明令禁止：出现“五官端正/整体面容和谐/面容和谐”等套话；卦象组合中禁止仅复制三象‘性格倾向’原句。"
      "将结果通过 submit_analysis_v3 工具返回，并"+_json_hint()+"。语言：中文。本消息含“JSON”以满足 API 要求。"
    )
    user = "请严格按要求分析图片，并只以 JSON 格式通过函数返回。"
    return [{"role":"system","content":sys},{"role":"user","content":user}]

def _inflate_dotted_keys(obj):
    if not isinstance(obj, dict): return obj
    out = {}
    for k,v in obj.items():
        if "." not in k: out[k]=_inflate_dotted_keys(v) if isinstance(v,dict) else v
    for k,v in obj.items():
        if isinstance(k,str) and "." in k:
            head,tail=k.split(".",1)
            base = out.setdefault(head, {})
            if not isinstance(base, dict): base = {}; out[head]=base
            cur=base
            parts=tail.split(".")
            for i,p in enumerate(parts):
                if i==len(parts)-1: cur[p]=v
                else: cur=cur.setdefault(p,{})
    for k in list(out.keys()):
        if isinstance(out[k], dict): out[k]=_inflate_dotted_keys(out[k])
    return out

def _call_openai(messages):
    return client.chat.completions.create(
        model="gpt-4o",
        temperature=0.45,
        tools=_build_tools_schema(),
        tool_choice={"type":"function","function":{"name":"submit_analysis_v3"}},
        response_format={"type":"json_object"},
        messages=messages,
    )

def _synthesize_combo(ta):
    hexes = [(ta.get("姿态") or {}).get("卦象",""),
             (ta.get("神情") or {}).get("卦象",""),
             (ta.get("面容") or {}).get("卦象","")]
    traits = [BAGUA_TRAITS.get(h,"").split("/")[0] for h in hexes if h]
    bullets = []
    if len(traits)>=2:
        bullets.append(f"外在表现偏{traits[0]}，内在驱动更{traits[1]}。")
    if "兑" in hexes:
        bullets.append("沟通风格亲和而直接，重视真实与愉悦的互动。")
    if "坎" in hexes:
        bullets.append("决策前会评估风险与后果，偏稳健。")
    if "震" in hexes:
        bullets.append("遇事行动果断，推进节奏快。")
    if "离" in hexes:
        bullets.append("表达清晰，擅长信息提炼与呈现。")
    if "乾" in hexes:
        bullets.append("具备主导性与目标感，愿意承担责任。")
    if "坤" in hexes:
        bullets.append("处事包容稳妥，善于托底与承载团队。")
    if "艮" in hexes:
        bullets.append("有边界感与秩序感，做事沉稳可靠。")
    if "巽" in hexes:
        bullets.append("倾向协商与整合资源，善做协调者。")
    # 去重并限长
    seen=set(); out=[]
    for b in bullets:
        if b not in seen: seen.add(b); out.append(b)
        if len(out)>=5: break
    return hexes, out

def _is_generic_face_text(text:str)->bool:
    if not isinstance(text,str): return False
    return any(key in text for key in GENERIC_FACE_LINES)

def _compose_auto_archetype(hexes):
    tags=[BAGUA_TRAITS.get(h,"").split("/")[0] for h in hexes if h]
    if len(tags)>=2: return f"外{tags[0]}内{tags[1]}"
    return tags[0]+"取向" if tags else ""

def _coerce_output(data: Dict[str,Any]) -> Dict[str,Any]:
    data = _inflate_dotted_keys(data)
    out = dict(data)
    meta = out.get("meta") or {}
    if not isinstance(meta, dict): meta = {}
    out["meta"]=meta

    ta = meta.get("triple_analysis") or {}
    face_parts = meta.get("face_parts") or {}

    # 面相概览：若文本太泛，则用五官投票合成
    dominant = ""
    if isinstance(face_parts, dict):
        hex_list=[(v or {}).get("卦象","") for v in face_parts.values() if isinstance(v,dict)]
        if hex_list:
            try:
                dominant = statistics.mode([h for h in hex_list if h])
            except Exception:
                dominant = hex_list[0] if hex_list else ""
    if _is_generic_face_text(out.get("sections",{}).get("面相","")) and dominant:
        sym=BAGUA_SYMBOLS.get(dominant,"")
        rep=f"面相整体偏{dominant}（{sym}）之象，五官呈现出与之相符的气质。"
        out.setdefault("sections",{}); out["sections"]["面相"] = rep

    # 组合标题+要点（综合，而非复读）
    hexes, synth_bullets = _synthesize_combo(ta)
    combo_title = " + ".join([h for h in hexes if h])
    if combo_title: meta["combo_title"]=combo_title

    # 总览卡：合并 summary + 组合要点
    lead = out.get("summary","")
    meta["overview_card"] = {"title": f"🔮 卦象组合：{combo_title}" if combo_title else "🔮 卦象组合",
                             "summary": lead,
                             "bullets": synth_bullets}

    # 长版领域建议（各 100–160 字）兼容旧字段
    dd = meta.get("domains_detail") or {}
    def _expand(txt, fallback):
        if not isinstance(txt,str) or len(txt)<80:
            return (fallback or "") + " 倾向将优势场景与风险点成对管理：用优势覆盖关键节点，同时设置检查点与反馈机制，以保证节奏与质量。"
        return txt
    meta["domains_detail_long"]={
        "金钱与事业": _expand(dd.get("金钱与事业",""), "在事业中建议把主导性与稳健度结合，先定清晰目标与边界，再逐步推进"),
        "配偶与感情": _expand(dd.get("配偶与感情",""), "在关系中保持真诚表达与稳固承诺，关注对方节奏与需求差异，营造可预期的安全感")
    }

    # 标题辅助
    def _title_with_hex(section_key: str, ta_key: str):
        hexname = (ta.get(ta_key) or {}).get("卦象","")
        symbol = BAGUA_SYMBOLS.get(hexname,"")
        return f"{section_key} → {hexname}卦（{symbol}）" if hexname and symbol else (f"{section_key} → {hexname}卦" if hexname else section_key)
    meta["sections_titles"]={"姿态":_title_with_hex("姿态","姿态"),"神情":_title_with_hex("神情","神情"),"面相":_title_with_hex("面相","面容")}

    # 顶部标签
    arch = out.get("archetype") or _compose_auto_archetype(hexes)
    out["archetype"]=arch
    try: out["confidence"]=float(out.get("confidence",0.0))
    except: out["confidence"]=0.0
    meta["headline"]={"tag":out["archetype"],"confidence":out["confidence"]}

    out["meta"]=meta
    return out

@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/", include_in_schema=False)
def root():
    return HTMLResponse("<h3>Selfy AI</h3><a href='/docs'>/docs</a>")

@app.get("/version")
def version(): return {"version":VERSION,"schema":SCHEMA_ID,"debug":DEBUG}

def _call_gpt_tool_with_image(data_url: str) -> Dict[str,Any]:
    if client is None: raise RuntimeError("OpenAI client not initialized")
    messages = _prompt_for_image()
    messages[-1]["content"]=[{"type":"text","text":messages[-1]["content"]},{"type":"image_url","image_url":{"url":data_url}}]
    resp=_call_openai(messages)
    choice=resp.choices[0]
    tool_calls=getattr(choice.message,"tool_calls",None)
    if tool_calls:
        args=json.loads(tool_calls[0].function.arguments)
    else:
        content=getattr(choice.message,"content",None)
        if isinstance(content,str) and content.strip().startswith("{"):
            args=json.loads(content)
        else:
            raise RuntimeError("Model did not return tool_calls.")
    return {"tool_args":args, "oai_raw": resp if DEBUG else None}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        if not file: raise HTTPException(400,"No file")
        ct=file.content_type or ""
        if not ct.startswith("image/"): raise HTTPException(415,f"Unsupported content type: {ct}")
        raw=await file.read()
        if not raw: raise HTTPException(400,"Empty file")
        if len(raw)>15*1024*1024: raise HTTPException(413,"File too large (>15MB)")

        data_url=_to_data_url(raw, ct)
        logger.info("[UPLOAD] %s %dB %s", file.filename, len(raw), ct)

        result=_call_gpt_tool_with_image(data_url)
        tool_args=result["tool_args"]

        final_out=_coerce_output(tool_args)

        if DEBUG:
            meta=final_out.setdefault("meta",{}).setdefault("debug",{})
            meta["file_info"]={"filename":file.filename,"content_type":ct,"size":len(raw)}
            try:
                meta["oai_choice_finish_reason"]=result["oai_raw"].choices[0].finish_reason
            except Exception:
                meta["oai_choice_finish_reason"]="n/a"

        return JSONResponse(content=final_out, status_code=200)
    except HTTPException as he:
        if DEBUG: return JSONResponse(status_code=he.status_code, content={"error":he.detail,"debug":{"trace":traceback.format_exc()}})
        raise
    except Exception as e:
        logging.exception("upload failed: %s", e)
        body={"error":"Internal Server Error"}
        if DEBUG: body["debug"]={"message":str(e),"trace":traceback.format_exc()}
        return JSONResponse(status_code=500, content=body)
