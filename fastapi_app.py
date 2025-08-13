# fastapi_app.py  (runtime v3.8.1, analysis logic v3.7.2, humanized phrasing + 主/辅/基 synthesis • Yijing combo independent)
import os, base64, json, logging, traceback, re, math
from typing import Dict, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI

RUNTIME_VERSION = "3.8.1"
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
    return ("只以 JSON object 返回（必须 JSON）。示例:{\"summary\":\"…\",\"archetype\":\"…\",\"confidence\":0.9,"
            "\"sections\":{\"姿态\":\"…\",\"神情\":\"…\",\"面相\":\"…\"},"
            "\"domains\":[\"金钱与事业\",\"配偶与感情\"],"
            "\"meta\":{\"triple_analysis\":{\"姿态\":{\"说明\":\"…\",\"卦象\":\"艮\",\"解读\":\"…\",\"性格倾向\":\"…\"},\"神情\":{…},\"面容\":{…},\"组合意境\":\"…\",\"总结\":\"…\"},"
            "\"face_parts\":{\"眉\":{\"特征\":\"…\",\"卦象\":\"…\",\"解读\":\"…\"},\"眼\":{…},\"鼻\":{…},\"嘴\":{…},\"颧/下巴\":{…}},"
            "\"domains_detail\":{\"金钱与事业\":\"…(60–90字)\",\"配偶与感情\":\"…(60–90字)\"}}}")

def _prompt_for_image_v372():
    # 说明：保留 3.7.2 的核心判定逻辑，不动分析，只规范输出结构
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

# ===== 文本后处理：去代词 / 去复读 / 人话化 =====

DOMAIN_LEADS = r"(在(金钱与事业|配偶与感情|事业|感情)(方面|中|里)?|目前|近期|当下)"

def _depronoun(s: str) -> str:
    """去掉“他/她/TA/你/其/对方/在…上/中/目前/近期”等口头起句，使语句客观中性"""
    if not isinstance(s, str): return s
    s = s.strip()
    s = re.sub(r"^(他|她|TA|你|对方|其)(的)?[，、： ]*", "", s)
    s = re.sub(r"^(在(事业|感情|生活)[上中]|目前|近期)[，、： ]*", "", s)
    return s

def _neutralize(s: str) -> str:
    """全局去人称代词/场景口头语/弱化词；把'她可能/他会'等改为中性判断"""
    if not isinstance(s, str): return s
    s = s.strip()
    s = re.sub(r"(他|她|TA|对方|其)(的)?", "", s)
    s = re.sub(DOMAIN_LEADS + r"[，、： ]*", "", s)
    s = re.sub(r"(可能|或许|也许)[，、 ]*", "", s)
    s = re.sub(r"[；;]+", "；", s)
    s = re.sub(r"[，,]{2,}", "，", s)
    return s.strip("；，。 ")

def _dedupe_phrase(s: str) -> str:
    """以逗号/句号切分做有序去重，避免“复读机”"""
    if not isinstance(s, str): return s
    parts = re.split(r"[，,。\.]", s)
    seen, kept = set(), []
    for p in parts:
        t = p.strip()
        if not t: continue
        if t not in seen:
            seen.add(t)
            kept.append(t)
    out = "，".join(kept)
    out = re.sub(r"(，){2,}", "，", out).strip("，")
    return out

def _strip_domain_lead(s: str) -> str:
    """专门去掉领域口头化引导语：在金钱与事业方面… / 在感情中…"""
    if not isinstance(s, str): return s
    return re.sub("^" + DOMAIN_LEADS + r"[，、： ]*", "", s.strip())

def _first_clause(s: str, maxlen: int = 20) -> str:
    """取一个完整子句（到第一个标点为止），最长 maxlen，避免半句被截断"""
    if not isinstance(s, str): return s
    s = s.strip()
    m = re.split(r"[。；；;，,]", s, maxsplit=1)
    head = (m[0] or "").strip()
    if len(head) > maxlen:
        head = head[:maxlen].rstrip("，,。；； ")
    return head

# ----- OpenAI 调用 -----

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

# ---------- Synthesis helpers ----------
# 注：以下词义用于“组合推导”的语义库（非模板）。
HEX_SUMMARY = {
    "乾":"自信·领导·果断",     # 乾为天：刚健、自强、主导
    "坤":"包容·稳定·承载",     # 坤为地：厚德、柔顺、承载
    "震":"行动·突破·起势",     # 震为雷：发动、开拓、决断
    "巽":"协调·渗透·说服",     # 巽为风：入而不争、调和、影响
    "坎":"谨慎·探深·智谋",     # 坎为水：险中求、谋略、求证
    "离":"明晰·表达·洞察",     # 离为火：明辨、洞察、表达
    "艮":"止定·边界·稳重",     # 艮为山：当止则止、立界、稳守
    "兑":"亲和·交流·悦人"      # 兑为泽：说也、和悦、沟通
}

# 五行&阴阳（用于卦间关系）
WUXING = {
    "乾":{"element":"金","polarity":"阳","virtue":"刚健自强、御领局面"},
    "兑":{"element":"金","polarity":"阴","virtue":"和悦亲和、以乐感人"},
    "离":{"element":"火","polarity":"阴","virtue":"明辨洞察、擅于表达"},
    "震":{"element":"木","polarity":"阳","virtue":"发动起势、敢于突破"},
    "巽":{"element":"木","polarity":"阴","virtue":"渗透协调、善谋合众"},
    "坎":{"element":"水","polarity":"阳","virtue":"审慎探深、居安识危"},
    "艮":{"element":"土","polarity":"阳","virtue":"止定有度、守正立界"},
    "坤":{"element":"土","polarity":"阴","virtue":"厚德载物、内敛承载"}
}

# 五行相生/相克表
SHENG = {"木":"火","火":"土","土":"金","金":"水","水":"木"}
KE    = {"木":"土","土":"水","水":"火","火":"金","金":"木"}

def _rel(a: str, b: str) -> str:
    """返回 a→b 的关系词：相生/相克/同气"""
    if not a or not b: return ""
    if a == b: return "同气相求"
    if SHENG.get(a) == b: return "相生"
    if KE.get(a) == b: return "相克"
    return "相并"

def _style_by_main(h: str) -> str:
    """主卦定整体行事风格（总结句用）"""
    if h in ("乾","震"): return "整体偏进取与主导，宜把握方向、主动开局"
    if h in ("坤","艮"): return "整体偏稳守与承载，宜厚积薄发、稳中求进"
    if h in ("离",):     return "整体偏明辨与表达，宜公开复盘、以清晰促推进"
    if h in ("兑",):     return "整体偏亲和与沟通，宜以合众之力达成目标"
    if h in ("巽",):     return "整体偏协调与渗透，宜柔性推进、润物无声"
    if h in ("坎",):     return "整体偏审慎与谋略，宜先求证后判断、步步为营"
    return "行事风格平衡，宜顺势而为"

def _synthesize_combo(hexes: List[str], ta: Dict[str,Any], traits: List[str]) -> str:
    """
    易经式“主/辅/基”独立推导（不复用三分象文本）：
    - 主（姿态）定大势；辅（神情）调其用；基（面容）定其根。
    - 引入五行与阴阳，分析主-辅、基-主的关系（相生/相克/同气/相并）。
    - 输出为一段总结性占断：不拼三分象原句。
    """
    zh, sh, bh = (hexes + ["", "", ""])[:3]
    keys = [h for h in [zh, sh, bh] if h]
    if not keys: return ""

    def vw(h, key): 
        return (WUXING.get(h) or {}).get(key, "")
    def hs(h): 
        return HEX_SUMMARY.get(h, "")

    # 1) 卦德开篇：主/辅/基 + 五行/德性
    parts = []
    for role, h in (("主", zh), ("辅", sh), ("基", bh)):
        if not h: continue
        ele = vw(h,"element"); pol = vw(h,"polarity"); vir = vw(h,"virtue")
        sym = BAGUA_SYMBOLS.get(h,"")
        seg = f"{role}{h}（{sym}），属{ele}为{pol}，{vir}"
        parts.append(seg)

    lead = "；".join(parts) + "。"

    # 2) 卦间关系：主-辅、基-主
    rel1 = _rel(vw(zh,"element"), vw(sh,"element")) if zh and sh else ""
    rel2 = _rel(vw(bh,"element"), vw(zh,"element")) if bh and zh else ""

    rel_texts = []
    if rel1:
        if rel1 == "相生": rel_texts.append("主辅相生，气势顺畅，用刚得柔")
        elif rel1 == "相克": rel_texts.append("主辅相克，宜调和收放，以免用力过度")
        elif rel1 == "同气相求": rel_texts.append("主辅同气，风格纯粹但需防偏执")
        else: rel_texts.append("主辅相并，各行其势，需以节奏统摄")
    if rel2:
        if rel2 == "相生": rel_texts.append("基生主，根基供给，后劲充足")
        elif rel2 == "相克": rel_texts.append("基克主，内在拉扯，宜先稳根再行势")
        elif rel2 == "同气相求": rel_texts.append("基与主同气，所守所为一致")
        else: rel_texts.append("基与主相并，宜以规则约束以成序")

    rel_para = (" ".join(rel_texts) + "。") if rel_texts else ""

    # 3) 卦意总括：参考主卦风格收束
    style = _style_by_main(zh) if zh else "风格平衡，宜顺势而为"

    # 4) 收敛成一段独立总结
    out = f"三象相合：{lead}{rel_para}{style}。"
    out = re.sub(r"[；;]+", "；", out)
    out = _dedupe_phrase(out)
    return out

def _insight_for_domains(hexes: List[str]) -> Dict[str, str]:
    """基于卦象给“近期状态”的要点，弱模板、强卦意（更像即时表达）"""
    s = set([h for h in hexes if h])
    biz = []
    if "乾" in s or "震" in s: biz.append("推进有力、节奏向前")
    if "离" in s: biz.append("表达清楚、复盘到位")
    if "兑" in s or "巽" in s: biz.append("善谈判协同、能带动人")
    if "坤" in s or "艮" in s: biz.append("落地稳、边界明、抗干扰")
    if "坎" in s: biz.append("风险意识强、方案留后手")

    love = []
    if "兑" in s: love.append("氛围轻松、互动自然")
    if "离" in s: love.append("善表达想法、共情到位")
    if "坤" in s: love.append("重承诺与照顾")
    if "坎" in s: love.append("在意安全感、较敏感")
    if "震" in s or "乾" in s: love.append("关键时会主动")
    if "艮" in s: love.append("保持分寸与稳定")
    return {"事业": "；".join(biz), "感情": "；".join(love)}

def _merge_status_and_detail(status: str, detail: str) -> str:
    """合并“状态要点 + 模型段首句”，彻底去代词/去领域引导语/去复读"""
    detail_first = detail.split("。")[0].strip() if detail else ""
    detail_first = _neutralize(_strip_domain_lead(detail_first))
    status = _neutralize(_strip_domain_lead(status or ""))
    parts = [p for p in [status, detail_first] if p]
    text = "；".join(parts).rstrip("；")
    return _dedupe_phrase(text)

def _imperative_suggestion(detail: str, hexes: List[str], domain: str) -> str:
    """
    以卦象导向生成“可执行建议”，避免千篇一律；输出用中性客观表达。
    """
    if not detail: detail = ""
    s = set([h for h in hexes if h])
    tips = []
    if domain == "事业":
        if "乾" in s or "震" in s: tips.append("把阶段目标拉清楚，今天就推进一小步")
        if "离" in s: tips.append("把复盘公开出来，用数据说话")
        if "兑" in s or "巽" in s: tips.append("约一场关键协同，先换位再谈目标")
        if "坤" in s or "艮" in s: tips.append("定边界与节奏，不抢不拖")
        if "坎" in s: tips.append("列出前三个风险，准备B计划")
    else:
        if "兑" in s: tips.append("用轻松语气回应，及时给反馈")
        if "坤" in s: tips.append("把在意的事说清楚，并兑现承诺")
        if "离" in s: tips.append("直说真实想法，也说清界限")
        if "震" in s or "乾" in s: tips.append("重要节点别犹豫，主动一点")
        if "坎" in s: tips.append("别先入为主，多求证再判断")
        if "艮" in s: tips.append("尊重彼此节奏，保留各自空间")

    base = _neutralize(_strip_domain_lead(detail.strip())).rstrip("；")
    add = "；".join(tips[:3])
    out = (base + ("。建议：" if base else "建议：") + add + "。") if add else base
    out = re.sub(r"[；;]+", "；", out)
    return _dedupe_phrase(out)

def _collect_traits_and_merge(ta: Dict[str,Any]) -> (List[str], Dict[str,Any]):
    """收集三象里的'性格倾向'，并把每象的‘说明+解读’合并为一句"""
    traits = []
    new_ta = {}
    for key in ["姿态","神情","面容"]:
        o = (ta.get(key) or {}).copy()
        tend = (o.get("性格倾向") or "").strip().rstrip("；;。")
        if tend: traits.append(tend)
        desc = (o.get("说明") or "")
        inter = (o.get("解读") or "")
        merged = _combine_sentence(desc, inter)
        o["说明"] = desc.strip().rstrip("；;。")
        o["解读"] = merged.strip()
        o["性格倾向"] = ""
        new_ta[key] = o
    for k in ta.keys():
        if k not in new_ta:
            new_ta[k] = ta[k]
    return traits, new_ta

def _combine_sentence(desc: str, interp: str) -> str:
    """合并‘说明 + 解读’，强去代词/去复读，让句子更像人说话"""
    if not desc and not interp: return ""
    desc = _neutralize(_depronoun((desc or "").strip().rstrip("；;。")))
    interp = _neutralize(_depronoun((interp or "").strip().lstrip("——").lstrip("- ").strip().rstrip("；;。")))
    interp = re.sub(r"^(这种|此类|这类|其|这种姿态|这种神情|这种面容)[，、： ]*", "", interp)
    s = f"{desc}，{interp}。" if (desc and interp) else f"{desc or interp}。"
    s = re.sub(r"[；;]+", "，", s)
    s = re.sub(r"，，+", "，", s)
    s = _dedupe_phrase(s)
    return s

def _coerce_output(data: Dict[str,Any]) -> Dict[str,Any]:
    out = dict(data)
    meta = out.get("meta") or {}
    if not isinstance(meta, dict): meta = {}
    out["meta"] = meta

    ta = meta.get("triple_analysis") or {}
    traits, ta = _collect_traits_and_merge(ta)
    meta["triple_analysis"] = ta

    hexes = [(ta.get("姿态") or {}).get("卦象",""),
             (ta.get("神情") or {}).get("卦象",""),
             (ta.get("面容") or {}).get("卦象","")]
    combo_title = " + ".join([h for h in hexes if h])
    meta["combo_title"] = combo_title

    synthesized = _synthesize_combo(hexes, ta, traits)
    one = (ta.get("总结") or out.get("summary",""))
    overview = (synthesized or one).strip().rstrip("；;")
    meta["overview_card"] = {"title": f"🔮 卦象组合：{combo_title}" if combo_title else "🔮 卦象组合",
                             "summary": overview}

    try:
        out["confidence"] = float(out.get("confidence",0.0))
    except Exception:
        out["confidence"] = 0.0
    arch = (out.get("archetype") or "").strip()
    meta["headline"] = {"tag": arch, "confidence": out["confidence"]}

    dd = meta.get("domains_detail") or {}
    status = _insight_for_domains(hexes)
    merged_status = {
        "事业": _merge_status_and_detail(status.get("事业",""), dd.get("金钱与事业","")),
        "感情": _merge_status_and_detail(status.get("感情",""), dd.get("配偶与感情","")),
    }
    meta["domains_status"] = merged_status
    meta["domains_suggestion"] = {
        "事业": _imperative_suggestion(dd.get("金钱与事业",""), hexes, "事业"),
        "感情": _imperative_suggestion(dd.get("配偶与感情",""), hexes, "感情")
    }

    def _clean(s):
        if not isinstance(s, str): return s
        s = s.replace("——", "，")
        s = re.sub(r"[；;]+", "；", s)
        s = re.sub(r"；([。！])", r"\1", s)
        s = re.sub(r"([。！？])；", r"\1", s)
        s = _depronoun(s)
        s = _neutralize(s)
        s = _dedupe_phrase(s)
        return s

    out["summary"] = _clean(out.get("summary",""))
    out["archetype"] = _clean(out.get("archetype",""))

    def _deep_clean(x):
        if isinstance(x, dict):
            return {k:_deep_clean(v) for k,v in x.items()}
        if isinstance(x, list):
            return [_deep_clean(v) for v in x]
        return _clean(x)

    out["meta"] = _deep_clean(meta)
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
