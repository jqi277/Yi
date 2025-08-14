# fastapi_app.py  (runtime v3.9.0, analysis logic v3.9.0·易经推导版)
# 3.9.0：易经“推导版”
# - 三合象：主/辅/基 + 五行生克/同气/相并 → 只做“人物画像”，不夹带建议
# - 事业/感情：按卦象知识库推导“近期状态/近期建议”；主=风格，辅=外部，人=基础
# - 建立卦象知识库（每卦：人格关键词、事业状态/建议、感情状态/建议、五行）
# - 去模板化：所有文本从卦义+生克推导生成，而非通用口号
import os, base64, json, logging, traceback, re
from typing import Dict, Any, List, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI

RUNTIME_VERSION = "3.9.0"
ANALYSIS_VERSION = os.getenv("ANALYSIS_VERSION", "390").strip()
SCHEMA_ID = "selfy.v3"
DEBUG = str(os.getenv("DEBUG","0")).strip() in ("1","true","True","YES","yes")

logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("selfy-ai")

app = FastAPI(title="Selfy AI - YiJing Analysis API (推导版)", version=RUNTIME_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

try:
    client = OpenAI()
except Exception as e:
    logger.error("OpenAI client init failed: %s", e); client=None

# ---- 常量与工具 ----
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

def _json_hint() -> str:
    return ("只以 JSON object 返回（必须 JSON）。示例:{\"summary\":\"…\",\"archetype\":\"…\",\"confidence\":0.9,"
            "\"sections\":{\"姿态\":\"…\",\"神情\":\"…\",\"面相\":\"…\"},"
            "\"domains\":[\"金钱与事业\",\"配偶与感情\"],"
            "\"meta\":{\"triple_analysis\":{\"姿态\":{\"说明\":\"…\",\"卦象\":\"艮\",\"解读\":\"…\",\"性格倾向\":\"…\"},\"神情\":{…},\"面容\":{…},\"组合意境\":\"…\",\"总结\":\"…\"},"
            "\"face_parts\":{\"眉\":{\"特征\":\"…\",\"卦象\":\"…\",\"解读\":\"…\"},\"眼\":{…},\"鼻\":{…},\"嘴\":{…},\"颧/下巴\":{…}}}}")

def _prompt_for_image_v390():
    sys = (
      "你是 Selfy AI 的易经观相助手（v3.9.0 推导基底）。"
      "严格按“三象四段式”分析：【姿态/神情/面容】三部分。每部分必须包含："
      "1) 说明：1句，客观描绘外观/动作/气质；"
      "2) 卦象：仅写一个卦名（艮/离/兑/乾/坤/震/巽/坎）；"
      "3) 解读：1–2句，基于卦象与观察做含义阐释；"
      "4) 性格倾向：1–2句，独立成段，不要与“解读”重复措辞。"
      "然后给出简要总结、人设标签，并在 meta.face_parts 中补充五官细节（眉/眼/鼻/嘴/颧/下巴任五项）。"
      "domains 仅从 ['金钱与事业','配偶与感情'] 选择，meta.domains_detail 中可给出各 60–90 字文本（后端会再做推导融合）。"
      "将结果通过 submit_analysis_v3 工具返回，并"+_json_hint()+"。语言：中文。"
    )
    user = "请按规范分析图片，严格通过函数返回 JSON（不要输出自由文本）。"
    return [{"role":"system","content":sys},{"role":"user","content":user}]

# ---- 文本清理 ----
DOMAIN_LEADS = r"(在(金钱与事业|配偶与感情|事业|感情)(方面|中|里)?|目前|近期|当下)"
_STOPWORDS = r"(姿态|神情|面容|整体|气质|形象|给人以|一种|以及|并且|而且|更显|显得|展现出|流露出|透露出)"

def _depronoun(s: str) -> str:
    if not isinstance(s, str): return s
    s = s.strip()
    s = re.sub(r"^(他|她|TA|你|对方|其)(的)?[，、： ]*", "", s)
    s = re.sub(r"^(在(事业|感情|生活)[上中]|目前|近期)[，、： ]*", "", s)
    return s

def _neutralize(s: str) -> str:
    if not isinstance(s, str): return s
    s = s.strip()
    s = re.sub(r"(他|她|TA|对方|其)(的)?", "", s)
    s = re.sub(DOMAIN_LEADS + r"[，、： ]*", "", s)
    s = re.sub(r"(可能|或许|也许)[，、 ]*", "", s)
    s = re.sub(r"[；;]+", "；", s)
    s = re.sub(r"[，,]{2,}", "，", s)
    return s.strip("；，。 ")

def _canon_key(s: str) -> str:
    if not isinstance(s, str): return ""
    k = re.sub(_STOPWORDS, "", s)
    k = re.sub(r"[的地得]", "", k)
    k = re.sub(r"\s+", "", k)
    return k

def _dedupe_smart(s: str) -> str:
    if not isinstance(s, str): return s
    s = s.strip("。；，,; ")
    sentences = re.split(r"[。！？]", s)
    clean_sentences = []
    for sen in sentences:
        sen = sen.strip("，,;； ")
        if not sen: continue
        parts = re.split(r"[，,；;]", sen)
        seen_keys, kept = set(), []
        for p in parts:
            t = p.strip()
            if not t: continue
            ck = _canon_key(t)
            if ck and ck not in seen_keys:
                seen_keys.add(ck); kept.append(t)
        clean_sentences.append("，".join(kept))
    return "。".join(clean_sentences) + ("。" if clean_sentences else "")

def _strip_domain_lead(s: str) -> str:
    if not isinstance(s, str): return s
    s = re.sub("^" + DOMAIN_LEADS + r"[，、： ]*", "", s.strip())
    s = re.sub(r"^上[，、： ]*", "", s)  # 裁掉“上，”之类残留
    return s

# ---- 易经知识库 ----
HEX_SUMMARY = {  # 用于轻量提示
    "乾":"自信·主导·果断",
    "坤":"包容·稳定·承载",
    "震":"行动·突破·起势",
    "巽":"协调·渗透·说服",
    "坎":"谨慎·探深·求证",
    "离":"清晰·表达·洞察",
    "艮":"止定·边界·稳守",
    "兑":"亲和·交流·悦人"
}

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

SHENG = {"木":"火","火":"土","土":"金","金":"水","水":"木"}
KE    = {"木":"土","土":"水","水":"火","火":"金","金":"木"}

# 每卦在事业/感情中的“状态/建议”（基于象辞意涵，简化为可落地短句）
HEX_DOMAIN_KB: Dict[str, Dict[str, Dict[str,str]]] = {
    "乾":{
        "persona":"刚健自强、主导框架",
        "career":{"state":"行事果断，目标导向，善于定方向与标准","advice":"取象“利见大人”：向上连线、借势资源；分解目标，节奏稳健推进"},
        "love":{"state":"表达直接，重承诺与保护感","advice":"取象“天行健”：以诚相待，先定边界与节奏，再深入关系"}
    },
    "坤":{
        "persona":"厚德载物、稳定承载",
        "career":{"state":"稳扎稳打，重流程与配合，能落地执行","advice":"取象“厚德载物”：按部就班，先稳后广；把规则落成明确的步骤"},
        "love":{"state":"包容耐心，重陪伴与信任","advice":"以“地势坤”为范：多倾听少判断，承接对方需求，稳定关系底盘"}
    },
    "震":{
        "persona":"起势突破、敢为先",
        "career":{"state":"行动力强，善于开启项目并带动节奏","advice":"取象“雷动万物”：快速试错，小步快跑；以行动拉动资源聚集"},
        "love":{"state":"关键处能主动靠近，带动互动氛围","advice":"以“动”化“静”：制造正向互动，用真实行动表达在意"}
    },
    "巽":{
        "persona":"渗透协调、柔中有据",
        "career":{"state":"善协同与说服，能把人拉到同一轨道","advice":"取象“入而不争”：先融后领，厘清利害，让共识自然形成"},
        "love":{"state":"体贴分寸感强，擅化解小矛盾","advice":"以“和”为主：多确认、多复述，降低误解，稳中加深"}
    },
    "离":{
        "persona":"明晰表达、洞察分明",
        "career":{"state":"信息透明，逻辑清楚，擅总结与呈现","advice":"取象“日中见明”：先讲清缘由与标准，再进入执行与验收"},
        "love":{"state":"沟通直白，重情绪表达与仪式感","advice":"保持“明而不灼”：坦诚但不逼迫，给对方消化空间"}
    },
    "坎":{
        "persona":"居安识危、先证后行",
        "career":{"state":"审慎周密，善风险识别与预案","advice":"取象“习坎为险”：先核对关键数据与假设，留出A/B方案"},
        "love":{"state":"在意安全感，情绪起伏需被看见","advice":"以“实证”换“安心”：少猜多问，建立稳定的确认机制"}
    },
    "艮":{
        "persona":"止当其时、边界分明",
        "career":{"state":"能控节奏，守边界，推进有章法","advice":"取象“艮其背”：先定范围与优先级，再逐步扩展"},
        "love":{"state":"尊重边界，关系推进循序渐进","advice":"稳步靠近：给彼此独处与呼吸感，节奏略慢反更稳"}
    },
    "兑":{
        "persona":"以悦为和、亲和沟通",
        "career":{"state":"亲和力强，善沟通促成与客户关系","advice":"取象“说以成事”：把利益点讲清，先易后难，促成在共赢"},
        "love":{"state":"表达感受自然，互动轻松有趣","advice":"以“悦”养“深”：用日常的小确认与关怀，持续加温"}
    }
}

# ---- 生克关系 ----
def _rel(a: str, b: str) -> str:
    if not a or not b: return ""
    if a == b: return "同气相求"
    if SHENG.get(a) == b: return "相生"
    if KE.get(a) == b: return "相克"
    return "相并"

# ---- 三合象（纯人物画像） ----
def _style_by_main_plain(h: str) -> str:
    if h in ("乾","震"): return "行事节奏偏主动"
    if h in ("坤","艮"): return "行事节奏偏稳健"
    if h in ("离",):     return "风格重表达与清晰"
    if h in ("兑",):     return "风格重关系与亲和"
    if h in ("巽",):     return "风格重协调与渗透"
    if h in ("坎",):     return "风格偏谨慎与求证"
    return "风格平衡"

def _relation_plain(rel: str, pos: str) -> str:
    # pos: "mf" 主-辅；"bm" 基-主  —— 全部改为“描述性”，不下指令
    if pos == "mf":
        if rel == "相生": return "主辅同向，配合顺畅"
        if rel == "相克": return "主辅相制，推进时易有拉扯"
        if rel == "同气相求": return "主辅同频，执行干脆"
        return "主辅各守一隅，取舍权衡更显重要"
    else:
        if rel == "相生": return "根基与目标顺流，底盘给力"
        if rel == "相克": return "内在经验与目标相拧，心力有分配"
        if rel == "同气相求": return "内外一致，表达与行动不打架"
        return "资源取向与目标各有侧重，需要兼容并行"

def _synthesize_combo_portrait(hexes: List[str]) -> str:
    zh, sh, bh = (hexes + ["", "", ""])[:3]
    keys = [h for h in [zh, sh, bh] if h]
    if not keys: return ""

    def vw(h, key): 
        return (WUXING.get(h) or {}).get(key, "")

    lead_parts = []
    for role, h in (("主", zh), ("辅", sh), ("基", bh)):
        if not h: continue
        ele = vw(h,"element"); pol = vw(h,"polarity"); vir = vw(h,"virtue")
        sym = BAGUA_SYMBOLS.get(h,"")
        lead_parts.append(f"{role}{h}（{sym}），属{ele}为{pol}，{vir}")
    lead = "；".join(lead_parts) + "。"

    rel1 = _rel(vw(zh,"element"), vw(sh,"element")) if zh and sh else ""
    rel2 = _rel(vw(bh,"element"), vw(zh,"element")) if bh and zh else ""
    rel_texts = []
    if rel1: rel_texts.append(_relation_plain(rel1, "mf"))
    if rel2: rel_texts.append(_relation_plain(rel2, "bm"))
    style = _style_by_main_plain(zh) if zh else "风格平衡"

    tail = "；".join([t for t in rel_texts if t])
    tail = (tail + ("；" if tail else "") + style) if style else tail
    out = f"三象相合：{lead}{tail}。"
    return _dedupe_smart(out)

# ---- 三分象合句 & 专业提示（保持 v3.8 的优化，但去建议化） ----
def _combine_sentence(desc: str, interp: str) -> str:
    if not desc and not interp: return ""
    desc  = _neutralize(_depronoun((desc or "").strip().rstrip("；;。")))
    interp = _neutralize(_depronoun((interp or "").strip().lstrip("——").lstrip("- ").strip().rstrip("；;。")))
    interp = re.sub(r"^(这种|此类|这类|其|这种姿态|这种神情|这种面容)[，、： ]*", "", interp)
    s = f"{desc}，{interp}" if (desc and interp) else (desc or interp)
    s = re.sub(r"[；;]+", "；", s)
    s = re.sub(r"，，+", "，", s)
    return _dedupe_smart(s)

def _collect_traits_and_merge(ta: Dict[str,Any]) -> Tuple[List[str], Dict[str,Any]]:
    traits, new_ta = [], {}
    for key in ["姿态","神情","面容"]:
        o = (ta.get(key) or {}).copy()
        tend = (o.get("性格倾向") or "").strip().rstrip("；;。")
        if tend: traits.append(tend)
        desc = (o.get("说明") or "")
        inter = (o.get("解读") or "")
        merged = _combine_sentence(desc, inter)
        hexname = (o.get("卦象") or "").strip()
        pro = ""
        if hexname in HEX_SUMMARY:
            kw = HEX_SUMMARY[hexname].split("·")[1] if "·" in HEX_SUMMARY[hexname] else HEX_SUMMARY[hexname]
            pro = f"【{hexname}·{kw}】"
        if pro and merged:
            merged = f"{pro} {merged}"
        o["说明"] = desc.strip().rstrip("；;。")
        o["解读"] = merged.strip()
        o["性格倾向"] = ""  # 三合象的人格倾向只在内部合成
        new_ta[key] = o
    for k in ta.keys():
        if k not in new_ta:
            new_ta[k] = ta[k]
    return traits, new_ta

# ---- 领域推导：状态 & 建议 ----
def _rel_to_env_phrase(rel: str, domain: str, pos: str) -> str:
    # pos: "mf" 外部/人际；"bm" 资源/根基
    if domain == "事业":
        if pos == "mf":
            return {"相生":"外部助推，同事/客户更易配合",
                    "相克":"外部牵扯，协同成本上升",
                    "同气相求":"外部同频，信息传达更顺",
                    "相并":"侧重不同，需在目标与资源间取舍"}.get(rel,"")
        else:
            return {"相生":"底盘给力，资源与节奏顺流",
                    "相克":"旧经验牵制，资源与目标有拧",
                    "同气相求":"内外一致，执行阻力小",
                    "相并":"资源方向与目标各有侧重"}.get(rel,"")
    else:  # 感情
        if pos == "mf":
            return {"相生":"对方/关系易回应，互动更顺",
                    "相克":"彼此节奏不一，易误读",
                    "同气相求":"同频沟通，默契感强",
                    "相并":"关注点不同，需要更多理解"}.get(rel,"")
        else:
            return {"相生":"安全感充沛，关系底色稳定",
                    "相克":"旧情绪/旧模式牵扯当下",
                    "同气相求":"价值观一致，承诺易兑现",
                    "相并":"现实条件与期待不完全重合"}.get(rel,"")

def _domain_status_by_kb(hexes: List[str], domain: str) -> str:
    zh, sh, bh = (hexes + ["", "", ""])[:3]
    def e(h): return (WUXING.get(h) or {}).get("element","")
    s_main = HEX_DOMAIN_KB.get(zh,{}).get("career" if domain=="事业" else "love",{}).get("state","") if zh else ""
    s_fu   = HEX_DOMAIN_KB.get(sh,{}).get("career" if domain=="事业" else "love",{}).get("state","") if sh else ""
    s_base = HEX_DOMAIN_KB.get(bh,{}).get("career" if domain=="事业" else "love",{}).get("state","") if bh else ""

    r_mf = _rel(e(zh), e(sh)) if zh and sh else ""
    r_bm = _rel(e(bh), e(zh)) if bh and zh else ""

    env_phrase = _domain_status_by_kb._cache.setdefault((domain,"mf",r_mf), _rel_to_env_phrase(r_mf, domain, "mf")) if r_mf else ""
    base_phrase= _domain_status_by_kb._cache.setdefault((domain,"bm",r_bm), _rel_to_env_phrase(r_bm, domain, "bm")) if r_bm else ""

    parts = []
    if s_main: parts.append(f"主{zh}：{s_main}")
    if s_fu:   parts.append(f"辅{sh}：{s_fu}")
    if env_phrase: parts.append(env_phrase)
    if s_base: parts.append(f"基{bh}：{s_base}")
    if base_phrase: parts.append(base_phrase)

    text = "；".join([p for p in parts if p])
    return _dedupe_smart(text)
_domain_status_by_kb._cache = {}

def _domain_advice_by_kb(hexes: List[str], domain: str) -> str:
    zh, sh, bh = (hexes + ["", "", ""])[:3]
    s = []
    key = "career" if domain=="事业" else "love"
    if zh: s.append(HEX_DOMAIN_KB.get(zh,{}).get(key,{}).get("advice",""))
    if sh: s.append(HEX_DOMAIN_KB.get(sh,{}).get(key,{}).get("advice",""))
    if bh: s.append(HEX_DOMAIN_KB.get(bh,{}).get(key,{}).get("advice",""))

    # 关系修正（允许下指令，因为属于“建议”）
    def e(h): return (WUXING.get(h) or {}).get("element","")
    r_mf = _rel(e(zh), e(sh)) if zh and sh else ""
    r_bm = _rel(e(bh), e(zh)) if bh and zh else ""

    if domain=="事业":
        if r_mf=="相克": s.append("先把角色与期待讲清，再定推进节奏")
        if r_mf=="相生": s.append("识别助推点，借力发力")
        if r_bm=="相克": s.append("区分旧经验与当下目标，避免耗散")
        if r_bm=="相生": s.append("把成熟做法标准化，形成SOP")
        if r_mf=="同气相求": s.append("设置“唱反调”环节，避免盲点")
    else:
        if r_mf=="相克": s.append("明确界限与节奏，减少误会")
        if r_mf=="相生": s.append("顺势多创造正向互动")
        if r_bm=="相克": s.append("与过去和解，避免旧模式影响当下")
        if r_bm=="相生": s.append("把稳定的好习惯保留下来")
        if r_mf=="同气相求": s.append("保留空间与新鲜感")

    tips = [ _neutralize(_depronoun(t)).strip("。； ") for t in s if t ]
    tips = [t for i,t in enumerate(tips) if t and t not in tips[:i]]
    return _dedupe_smart(("；".join(tips[:3]) + "。") if tips else "")

# ---- 融合输出 ----
def _merge_status_and_detail(status: str, detail: str) -> str:
    # v3.9：保留一小句来自模型的描述作为“色彩”，但不覆盖KB推导
    detail_first = detail.split("。")[0].strip() if detail else ""
    detail_first = _neutralize(_strip_domain_lead(detail_first))
    status = _neutralize(_strip_domain_lead(status or ""))
    parts = [p for p in [status, detail_first] if p]
    text = "；".join(parts).rstrip("；")
    return _dedupe_smart(text)

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

    # 三合象：改为“人物画像”
    overview = _synthesize_combo_portrait(hexes)
    if not overview:
        overview = (ta.get("总结") or out.get("summary","")).strip().rstrip("；;")
    meta["overview_card"] = {"title": f"🔮 卦象组合：{combo_title}" if combo_title else "🔮 卦象组合",
                             "summary": overview}

    # headline
    try:
        out["confidence"] = float(out.get("confidence",0.0))
    except Exception:
        out["confidence"] = 0.0
    arch = (out.get("archetype") or "").strip()
    out["archetype"] = arch
    meta["headline"] = {"tag": arch or "人格画像", "confidence": out["confidence"]}

    # 领域：状态 & 建议（全部走KB + 生克推导）
    kb_status = {
        "事业": _domain_status_by_kb(hexes, "事业"),
        "感情": _domain_status_by_kb(hexes, "感情"),
    }
    dd = (meta.get("domains_detail") or {})  # 仅取色彩
    merged_status = {
        "事业": _merge_status_and_detail(kb_status.get("事业",""), dd.get("金钱与事业","")),
        "感情": _merge_status_and_detail(kb_status.get("感情",""), dd.get("配偶与感情","")),
    }
    meta["domains_status"] = merged_status
    meta["domains_suggestion"] = {
        "事业": _domain_advice_by_kb(hexes, "事业"),
        "感情": _domain_advice_by_kb(hexes, "感情"),
    }

    def _clean(s):
        if not isinstance(s, str): return s
        s = s.replace("——", "，")
        s = re.sub(r"[；;]+", "；", s)
        s = re.sub(r"；([。！])", r"\1", s)
        s = re.sub(r"([。！？])；", r"\1", s)
        s = _depronoun(s)
        s = _neutralize(s)
        return _dedupe_smart(s)

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

# ---- HTTP 路由 ----
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
    messages = _prompt_for_image_v390()
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
