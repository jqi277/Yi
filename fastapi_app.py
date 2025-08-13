# fastapi_app.py  (runtime v3.9.0 · YiJing inference)
# - Tri-image (姿态/神情/面相) stays intact from upstream (model v3.7.2 style)
# - NEW: 卦象组合(三合象) = 人物画像（主/辅/基 + 五行生克，严禁建议口吻）
# - NEW: 事业/感情 = 近期状态(bullets) + 近期建议(bullets)，完全按词库与生克推导
# - Lexicon hot-load: yijing_lexicon.json in same folder; reload on every request
# - Render-compatible; no external state required
import os, json, re, base64, logging, traceback
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI

RUNTIME_VERSION = "3.9.0"
ANALYSIS_VERSION = os.getenv("ANALYSIS_VERSION", "372")  # model prompt profile
SCHEMA_ID = "selfy.v3"
DEBUG = str(os.getenv("DEBUG","0")).strip() in ("1","true","True","YES","yes")

logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("selfy-ai")

app = FastAPI(title="Selfy AI — YiJing Inference API", version=RUNTIME_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

try:
    client = OpenAI()
except Exception as e:
    logger.error("OpenAI client init failed: %s", e); client=None

# --- Utils ---
BAGUA_SYMBOLS = {"艮":"山","离":"火","兑":"泽","乾":"天","坤":"地","震":"雷","巽":"风","坎":"水"}

def _to_data_url(content: bytes, content_type: str) -> str:
    return f"data:{content_type};base64,{base64.b64encode(content).decode('utf-8')}"

# --- Lexicon ---
def load_lexicon() -> Dict[str, Any]:
    """Hot-load lexicon every call. Must be placed next to this file."""
    path = os.path.join(os.path.dirname(__file__), "yijing_lexicon.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Lexicon missing or invalid, using minimal fallback: %s", e)
        # Minimal fallback for safety
        return {
            "persona": {"乾":"刚健自强","坤":"厚德载物","离":"明辨表达","兑":"亲和交流","震":"行动开局","巽":"协调入微","坎":"谨慎求证","艮":"止定稳守"},
            "domains": {"career":{},"love":{}},
            "wuxing":{"乾":"金","兑":"金","离":"火","震":"木","巽":"木","坎":"水","艮":"土","坤":"土"},
            "sheng":{"木":"火","火":"土","土":"金","金":"水","水":"木"},
            "ke":{"木":"土","土":"水","水":"火","火":"金","金":"木"},
            "fuse":{"生":"特质互补，更容易形成合力","克":"风格有冲突，需要先对齐方式再推进","比":"风格一致，优势被放大，同时要留意盲区","并":"各有侧重，分工清晰，互不干扰"},
            "rel_influence":{"career":{},"love":{}}
        }

# --- OpenAI tool schema ---
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
            "meta":{"type":"object","additionalProperties":True}
          },
          "required":["summary","archetype","confidence","sections","domains"],
          "additionalProperties":False
        }
      }
    }]

def _prompt_for_image_v372():
    sys = (
      "你是 Selfy AI 的易经观相助手（v3.7.2 风格）。"
      "严格按“三象四段式”分析：【姿态/神情/面容】三部分。每部分包含："
      "1) 说明：1句，客观描绘；2) 卦象：仅写一个八卦名；3) 解读：1–2句；4) 性格倾向：1–2句。"
      "随后：5) 卦象组合（90–150字）；6) 总结性格印象（20–40字）；7) archetype（2–5字）。"
      "同时在 meta.face_parts 中给【眉/眼/鼻/嘴/颧/下巴】（覆盖5项）特征与解读。"
      "domains 仅从 ['金钱与事业','配偶与感情'] 选择；在 meta.domains_detail 中各写 60–90 字建议。"
      "通过 submit_analysis_v3 工具以 JSON 返回，不要输出自由文本。"
    )
    user = "请按 3.7.2 风格对图像进行分析，并严格以工具返回 JSON。"
    return [{"role":"system","content":sys},{"role":"user","content":user}]

def _call_oai(messages):
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

# --- Text cleaners ---
DOMAIN_LEADS = r"(在(金钱与事业|配偶与感情|事业|感情)(方面|中|里)?|目前|近期|当下)"
_STOPWORDS = r"(姿态|神情|面容|整体|气质|形象|给人以|一种|以及|并且|而且|更显|显得|展现出|流露出|透露出)"
def _depronoun(s:str)->str:
    if not isinstance(s,str): return s
    s = re.sub(r"^(他|她|TA|你|对方|其)(的)?[，、： ]*", "", s.strip())
    s = re.sub(r"^(在(事业|感情|生活)[上中]|目前|近期)[，、： ]*", "", s)
    return s
def _neutralize(s:str)->str:
    if not isinstance(s,str): return s
    s = re.sub(r"(他|她|TA|对方|其)(的)?", "", s.strip())
    s = re.sub(DOMAIN_LEADS + r"[，、： ]*", "", s)
    s = re.sub(r"(可能|或许|也许)[，、 ]*", "", s)
    s = re.sub(r"[，,]{2,}", "，", s)
    s = re.sub(r"[；;]{2,}", "；", s)
    return s.strip("；，。 ")
def _canon_key(s:str)->str:
    if not isinstance(s,str): return ""
    k = re.sub(_STOPWORDS, "", s)
    k = re.sub(r"[的地得]", "", k); k = re.sub(r"\s+", "", k)
    return k
def _dedupe_smart(s:str)->str:
    if not isinstance(s,str): return s
    s = s.strip("。；，,; ")
    sentences = re.split(r"[。！？]", s)
    out = []
    for sen in sentences:
        sen = sen.strip("，,;； ")
        if not sen: continue
        parts = re.split(r"[，,；;]", sen)
        seen, kept = set(), []
        for p in parts:
            t = p.strip(); 
            if not t: continue
            ck = _canon_key(t)
            if ck and ck not in seen:
                seen.add(ck); kept.append(t)
        out.append("，".join(kept))
    return "。".join(out) + ("。" if out else "")

# --- YiJing inference helpers ---
def _rel(el_a:str, el_b:str, sheng:Dict[str,str], ke:Dict[str,str])->str:
    if not el_a or not el_b: return "并"
    if el_a == el_b: return "比"
    if sheng.get(el_a) == el_b: return "生"
    if ke.get(el_a) == el_b: return "克"
    return "并"

def _synthesize_combo(lex:Dict[str,Any], h1:str, h2:str, h3:str)->str:
    """三合象：人物画像（无建议）"""
    persona = lex.get("persona",{})
    wuxing  = lex.get("wuxing",{})
    sheng   = lex.get("sheng",{})
    ke      = lex.get("ke",{})
    fuse    = lex.get("fuse",{})
    parts = []
    for role,h in (("主",h1),("辅",h2),("基",h3)):
        if not h: continue
        sym = BAGUA_SYMBOLS.get(h,"")
        per = persona.get(h,"")
        el  = wuxing.get(h,"")
        p = f"{role}{h}（{sym}，{el}），{per}" if per else f"{role}{h}（{sym}，{el}）"
        parts.append(p)
    # relations
    r1 = _rel(wuxing.get(h1,""), wuxing.get(h2,""), sheng, ke) if h1 and h2 else ""
    r2 = _rel(wuxing.get(h3,""), wuxing.get(h1,""), sheng, ke) if h3 and h1 else ""
    r_texts = []
    if r1: r_texts.append(f"主与辅：{fuse.get(r1,'')}")
    if r2: r_texts.append(f"基与主：{fuse.get(r2,'')}")
    out = "；".join(parts) + "。" + " ".join([t for t in r_texts if t])
    # 去“这类人/这种人”类指代：本段只陈述特质
    return _dedupe_smart(_neutralize(_depronoun(out)))

def _collect_bullets(source: Dict[str,Any], keys: List[str], limit:int=4)->List[str]:
    seen, bullets = set(), []
    for k in keys:
        arr = (source.get(k) or {}).get("state") if isinstance(source.get(k), dict) else None
        if not arr: continue
        for item in arr:
            t = _dedupe_smart(_neutralize(_depronoun(str(item)))).strip("。")
            ck = _canon_key(t)
            if ck and ck not in seen:
                seen.add(ck); bullets.append("· " + t)
            if len(bullets) >= limit: break
        if len(bullets) >= limit: break
    return bullets

def _collect_advices(source: Dict[str,Any], keys: List[str], limit:int=3)->List[str]:
    seen, bullets = set(), []
    for k in keys:
        arr = (source.get(k) or {}).get("advice") if isinstance(source.get(k), dict) else None
        if not arr: continue
        for item in arr:
            t = _dedupe_smart(_neutralize(_depronoun(str(item)))).strip("。")
            ck = _canon_key(t)
            if ck and ck not in seen:
                seen.add(ck); bullets.append("· " + t)
            if len(bullets) >= limit: break
        if len(bullets) >= limit: break
    return bullets

def _relation_influence(lex:Dict[str,Any], domain:str, el_main:str, el_fu:str, el_base:str)->List[str]:
    sheng, ke = lex.get("sheng",{}), lex.get("ke",{})
    rel_text = lex.get("rel_influence",{}).get(domain,{})
    bullets = []
    if el_main and el_fu:
        r1 = _rel(el_main, el_fu, sheng, ke)
        if rel_text.get(r1): bullets.append("· " + rel_text[r1])
    if el_base and el_main:
        r2 = _rel(el_base, el_main, sheng, ke)
        if rel_text.get(r2): bullets.append("· " + rel_text[r2])
    # de-dup
    seen, out = set(), []
    for b in bullets:
        ck = _canon_key(b)
        if ck not in seen:
            seen.add(ck); out.append(b)
    return out[:2]

def _make_domains(lex:Dict[str,Any], h1:str, h2:str, h3:str)->Dict[str,Dict[str,List[str]]]:
    doms = {"career":{}, "love":{}}
    el = lex.get("wuxing",{})
    keys = [k for k in [h1,h2,h3] if k]
    for domain in ["career","love"]:
        source = lex.get("domains",{}).get(domain,{}) or {}
        status = _collect_bullets(source, keys, limit=4)
        status += _relation_influence(lex, domain, el.get(h1,""), el.get(h2,""), el.get(h3,""))
        # re-trim to <=4
        status = status[:4]
        advice = _collect_advices(source, keys, limit=3)
        doms[domain] = {"status": status, "advice": advice}
    return doms

# --- Post-processing of model output ---
def _combine_sentence(desc:str, interp:str)->str:
    if not desc and not interp: return ""
    desc  = _neutralize(_depronoun((desc or "").strip().rstrip("；;。")))
    interp = _neutralize(_depronoun((interp or "").strip().lstrip("——").lstrip("- ").strip().rstrip("；;。")))
    interp = re.sub(r"^(这种|此类|这类|其|这种姿态|这种神情|这种面容)[，、： ]*", "", interp)
    s = f"{desc}，{interp}" if (desc and interp) else (desc or interp)
    s = re.sub(r"[；;]+", "；", s); s = re.sub(r"，，+", "，", s)
    return _dedupe_smart(s)

def _coerce_output(tool_args: Dict[str,Any])->Dict[str,Any]:
    out = dict(tool_args)
    meta = out.get("meta") or {}
    if not isinstance(meta, dict): meta = {}
    out["meta"] = meta

    # Merge triple analysis parts
    ta = meta.get("triple_analysis") or {}
    def _apply(o):
        desc = (o.get("说明") or ""); inter = (o.get("解读") or "")
        merged = _combine_sentence(desc, inter)
        o["解读"] = merged; return o
    for k in ["姿态","神情","面容"]:
        if isinstance(ta.get(k), dict): ta[k] = _apply(ta[k])
    meta["triple_analysis"] = ta

    # Trio hexes
    h1 = (ta.get("姿态") or {}).get("卦象","") or ""
    h2 = (ta.get("神情") or {}).get("卦象","") or ""
    h3 = (ta.get("面容") or {}).get("卦象","") or ""

    # Load lexicon and synthesize
    lex = load_lexicon()
    combo_title = " + ".join([h for h in [h1,h2,h3] if h])
    combo_summary = _synthesize_combo(lex, h1, h2, h3)
    meta["overview_card"] = {"title": f"🔮 卦象组合：{combo_title}" if combo_title else "🔮 卦象组合",
                             "summary": combo_summary}

    doms = _make_domains(lex, h1, h2, h3)
    meta["domains_status"] = {"事业": "\n".join(doms["career"]["status"]),
                              "感情": "\n".join(doms["love"]["status"])}
    meta["domains_suggestion"] = {"事业": "\n".join(doms["career"]["advice"]),
                                  "感情": "\n".join(doms["love"]["advice"])}

    # Headline
    try:
        conf = float(out.get("confidence",0.0))
    except Exception: conf = 0.0
    meta["headline"] = {"tag": (out.get("archetype") or "").strip(), "confidence": conf}

    # Top-level clean
    def _clean(s):
        if not isinstance(s,str): return s
        s = s.replace("——","，")
        s = re.sub(r"[；;]+","；", s)
        s = re.sub(r"；([。！])", r"\1", s)
        s = re.sub(r"([。！？])；", r"\1", s)
        s = _depronoun(s); s = _neutralize(s)
        return _dedupe_smart(s)
    out["summary"] = _clean(out.get("summary",""))
    out["archetype"] = _clean(out.get("archetype",""))
    # deep clean meta
    def _deep(x):
        if isinstance(x, dict): return {k:_deep(v) for k,v in x.items()}
        if isinstance(x, list): return [_deep(v) for v in x]
        return _clean(x)
    out["meta"] = _deep(meta)
    return out

# --- Routes ---
@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/", include_in_schema=False)
def root():
    return HTMLResponse("<h3>Selfy AI</h3><a href='/docs'>/docs</a> · <a href='/mobile'>/mobile</a>")

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

        msgs = _prompt_for_image_v372()
        msgs[-1]["content"] = [
            {"type":"text","text":msgs[-1]["content"]},
            {"type":"image_url","image_url":{"url":data_url}}
        ]
        resp = _call_oai(msgs)
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
        final_out = _coerce_output(args)

        if DEBUG:
            final_out.setdefault("meta",{}).setdefault("debug",{})["oai_finish"] = resp.choices[0].finish_reason

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
