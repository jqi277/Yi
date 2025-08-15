# fastapi_app.py  (runtime v3.8.6-ux-fix1, analysis logic v3.7.2)
# 3.8.6-ux-fix1：按你的最新要求完成的“动态易经逻辑”版本
# - 仅在 UI 显示「图片清晰度」（去掉其它可信度露出）
# - 人格画像由三象 + 五官动态生成，无固定预设标签；容错非八卦名（如“泰”）
# - 三分象解读不再插入【卦·关键词】标签；“经文提示”独立成行，且多候选去模板化
# - 面相卡片内嵌“五官细节”（含各自卦象）；面相整体卦象由“五官两强爻”综合推算
# - “三才之道”改为“八卦类比”：根据主/辅/基与五行生克推导语句（非模板）
# - 事业/感情：输出为语句（“近期状态/具体建议”），由三象“分与合”推导
# - 文本清洗修正“。， / 。；”等；修复此前 SyntaxError / KeyError / 早返回问题

import os, base64, json, logging, traceback, re
from typing import Dict, Any, List, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI

RUNTIME_VERSION = "3.8.6-ux-fix1"
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

# ---------- 八卦 / 五行基础 ----------
BAGUA_SYMBOLS = {"艮":"山","离":"火","兑":"泽","乾":"天","坤":"地","震":"雷","巽":"风","坎":"水"}
WUXING = {
    "乾":{"element":"金","polarity":"阳","virtue":"刚健自强、御领局面"},
    "兑":{"element":"金","polarity":"阴","virtue":"和悦亲和、以乐感人"},
    "离":{"element":"火","polarity":"阴","virtue":"明辨洞察、擅于表达"},
    "震":{"element":"木","polarity":"阳","virtue":"发动起势、敢于突破"},
    "巽":{"element":"木","polarity":"阴","virtue":"渗透协调、善谋合众"},
    "坎":{"element":"水","polarity":"阳","virtue":"审慎探深、居安识危"},
    "艮":{"element":"土","polarity":"阳","virtue":"止定有度、守正立界"},
    "坤":{"element":"土","polarity":"阴","virtue":"厚德载物、内敛承载"},
}
# 生克表：key 生 -> value； key 克 -> value
SHENG = {"木":"火","火":"土","土":"金","金":"水","水":"木"}
KE    = {"木":"土","土":"水","水":"火","火":"金","金":"木"}

# 经文提示（默认+候选，防模板）
JINGWEN_HINT = {
    "乾":"《乾》亢龙有悔：过强则折，宜收锋敛势",
    "坤":"《坤》含弘光大：厚载不争，慎因循不决",
    "艮":"《艮》艮其背：守界有度，忌僵硬不化",
    "兑":"《兑》和兑以说：以悦达人，忌过度迎合",
    "离":"《离》明两作：明察而易苛，宜明而不燥",
    "坎":"《坎》习坎重险：慎审求证，忌反复犹疑",
    "震":"《震》震来虩虩：初动宜稳，忌躁进冒进",
    "巽":"《巽》小亨利往：渗透有方，忌优柔寡断",
}
JINGWEN_CANDIDATES = {
    "乾":[
        "《乾》亢龙有悔：过强则折，宜收锋敛势",
        "《乾》见龙在田：位未至而志先定，慎躁进",
        "《乾》君子以自强不息：持盈保泰"
    ],
    "坤":[
        "《坤》含弘光大：厚载不争，慎因循不决",
        "《坤》直方大：守正以容，戒因小失大",
        "《坤》先迷后得：宁迟毋躁，以静制动"
    ],
    "离":[
        "《离》明两作：明察不苛，忌过度求全",
        "《离》得位守中：照见而不炫耀",
        "《离》丽日中天：光在外而本在内"
    ],
    "兑":[
        "《兑》说以先民：言贵诚恳，不饰口实",
        "《兑》和兑以说：以悦达人，忌逢迎失度",
        "《兑》泽上行：和而有界"
    ],
    "艮":[
        "《艮》艮其背：守界有度，不逐外物",
        "《艮》止于至善：能止方能行",
        "《艮》山不移：定中求通，戒刚滞"
    ],
    "坎":[
        "《坎》习坎重险：慎审求证，不陷反复",
        "《坎》有孚：以实破疑",
        "《坎》中流砥柱：难中见韧"
    ],
    "震":[
        "《震》震来虩虩：初动宜稳，勿躁进",
        "《震》雷行：起而不越",
        "《震》先庚三日：预备在前"
    ],
    "巽":[
        "《巽》小亨利往：渗透有方，忌优柔",
        "《巽》入而不争：化人于无形",
        "《巽》巽而止：度中有伸"
    ],
}
def _pick_jingwen(hexname: str, role: str, rel_mf: str = "", rel_bm: str = "") -> str:
    """按卦、位置与生克关系挑一条经文，避免千篇一律。"""
    if not hexname: 
        return ""
    cands = JINGWEN_CANDIDATES.get(hexname) or []
    if not cands:
        return JINGWEN_HINT.get(hexname, "")
    key = (role or "") + (rel_mf or "") + (rel_bm or "")
    idx = (sum(ord(ch) for ch in key) + len(role)) % len(cands)
    return cands[idx]

# ---------- OpenAI Tools Schema ----------
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
            "sections":{"type":"object","properties":{
                "姿态":{"type":"string"},
                "神情":{"type":"string"},
                "面相":{"type":"string"}
            },"required":["姿态","神情","面相"],"additionalProperties":False},
            "domains":{"type":"array","items":{"type":"string"}},
            "meta":{"type":"object","additionalProperties":True}
          },
          "required":["summary","archetype","confidence","sections","domains"],
          "additionalProperties":False
        }
      }
    }]

# ---------- Prompt ----------
def _json_hint() -> str:
    return ("只以 JSON object 返回（必须 JSON）。示例:{\"summary\":\"…\",\"archetype\":\"…\",\"confidence\":0.9,"
            "\"sections\":{\"姿态\":\"…\",\"神情\":\"…\",\"面相\":\"…\"},"
            "\"domains\":[\"金钱与事业\",\"配偶与感情\"],"
            "\"meta\":{"
            "\"triple_analysis\":{\"姿态\":{\"说明\":\"…\",\"卦象\":\"艮\",\"解读\":\"…\",\"性格倾向\":\"…\"},\"神情\":{…},\"面容\":{…}},"
            "\"face_parts\":{\"眉\":{\"特征\":\"…\",\"卦象\":\"…\",\"解读\":\"…\"}},"
            "\"domains_status\":{\"事业\":\"…\",\"感情\":\"…\"},"
            "\"domains_suggestion\":{\"事业\":\"…\",\"感情\":\"…\"}"
            "}}")

def _prompt_for_image_v372():
    sys = (
      "你是 Selfy AI 的易经观相助手（v3.7.2）。务必通过函数返回 JSON。服务端只清洗校验，不补充文案。\n"
      "【三象】姿态/神情/面容：每项含 说明(1句)、卦象(单字)、解读(1–2句, 不要【】标签)、性格倾向(1句)。\n"
      "【五官】meta.face_parts：眉/眼/鼻/嘴/颧/下巴≥5项；每项含 特征/卦象(单字)/解读(可提“过旺/受阻”风险)。\n"
      "【八卦类比】title=三卦相加(如“艮 + 坤 + 兑”)；正文需：主/辅/基一句画像 + 主与辅、基与主五行关系语句 + 综合观。必须依据当前三卦生成，避免套话。\n"
      "【经文提示】为姿态/神情/面容各给一条，简短说明为何贴合当前观察（可引用卦辞/爻辞）。\n"
      "【事业/感情】meta.domains_status：各2–3句“近期状态”；meta.domains_suggestion：各2–3句“具体建议”。两者均由三象分与合推导。\n"
      + _json_hint()
    )
    user = "请分析图片并返回 JSON（不要输出自由文本）。"
    return [{"role":"system","content":sys},{"role":"user","content":user}]

# ---------- 文本清洗 ----------
DOMAIN_LEADS = r"(在(金钱与事业|配偶与感情|事业|感情)(方面|中|里)?|目前|近期|当下)"
_STOPWORDS = r"(姿态|神情|面容|整体|气质|形象|给人以|一种|以及|并且|而且|更显|显得|展现出|流露出|透露出)"

def _strip_leading_punct(s: str) -> str:
    if not isinstance(s, str): return s
    return re.sub(r"^[\s\.,;，。；、:：\-~·【】（）()]+", "", s)

def _depronoun(s: str) -> str:
    if not isinstance(s, str): return s
    s = s.strip()
    s = re.sub(r"^(他|她|TA|你|对方|其)(的)?[，、： ]*", "", s)
    s = re.sub(r"^(在(事业|感情|生活)[上中]|目前|近期|当下)[，、： ]*", "", s)
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
    clean_sentences: List[str] = []
    for sen in sentences:
        sen = sen.strip("，,;； ")
        if not sen:
            continue
        parts = re.split(r"[，,；;]", sen)
        seen_keys, kept = set(), []
        for p in parts:
            t = p.strip()
            if not t:
                continue
            ck = _canon_key(t)
            if ck and ck not in seen_keys:
                seen_keys.add(ck)
                kept.append(t)
        if kept:
            clean_sentences.append("，".join(kept))
    if not clean_sentences:
        return ""
    return "。".join(clean_sentences) + "。"

def _strip_domain_lead(s: str) -> str:
    if not isinstance(s, str): return s
    s = re.sub("^" + DOMAIN_LEADS + r"[，、： ]*", "", s.strip())
    s = re.sub(r"^上[，、： ]*", "", s)
    return s

# ---------- 五官 → 卦（爻象推合） ----------
FACIAL_MAP = {
    "眼":  {"primary":"离", "alt":["兑"]},     # 神在目，离火主明；笑眼亦近兑
    "眉":  {"primary":"巽", "alt":["震","艮"]},# 条达生发为木；浓硬逼人近艮
    "鼻":  {"primary":"艮", "alt":["坤"]},     # 山岳立面，界与定
    "嘴":  {"primary":"兑", "alt":["离"]},     # 泽为悦，兑主言
    "颧":  {"primary":"乾", "alt":["兑","震"]},# 权骨外拓，刚健取势
    "下巴":{"primary":"坤", "alt":["艮"]},     # 承载收敛
}
FEATURE_KEYWORDS = {
    "高":1.2,"挺":1.2,"立":1.1,"饱满":1.15,"明显":1.1,"宽":1.05,"尖":1.05,"厚":1.05,
    "圆":1.05,"有神":1.25,"发光":1.25,"上扬":1.1,"分明":1.1,"立体":1.1,"对称":1.05,
    "笑":1.15,"弯":1.05,"浓":1.05,"薄":0.95,"塌":0.9,"无神":0.9
}
PART_PRIORITY = ["眼","眉","鼻","颧","嘴","下巴"]  # 并列时的器官优先级（从高到低）

def _score_face_parts(face_parts: Dict[str,Any]) -> Dict[str,float]:
    """根据‘特征’文本长度和关键词给每个五官打显著度分。"""
    scores: Dict[str,float] = {}
    for part, info in (face_parts or {}).items():
        if part not in FACIAL_MAP:
            continue
        feat = (info or {}).get("特征","") or ""
        base = 1.0 + min(len(feat), 40) / 80.0  # 文本越具体，基础分略高（1.0~1.5）
        w = 1.0
        for k, mul in FEATURE_KEYWORDS.items():
            if k in feat:
                w *= mul
        scores[part] = base * w
    return scores

def _resolve_two_hex_to_main(h1: str, h2: str) -> str:
    """根据两卦五行关系推主卦：相生取'被生者'；相克取'克者'；同/并按外部排序优先。"""
    if h1 not in WUXING and h2 not in WUXING:
        return ""
    if h1 not in WUXING:
        return h2
    if h2 not in WUXING:
        return h1
    e1, e2 = WUXING[h1]["element"], WUXING[h2]["element"]
    if SHENG.get(e1) == e2:   # e1生e2 → 取 e2
        return h2
    if SHENG.get(e2) == e1:   # e2生e1 → 取 e1
        return h1
    if KE.get(e1) == e2:      # e1克e2 → 取 e1
        return h1
    if KE.get(e2) == e1:      # e2克e1 → 取 e2
        return h2
    return h1  # 同/并，保持外部优先序

def _infer_face_hex_from_parts(face_parts: Dict[str, Any]) -> str:
    """由五官中最显著的两项爻象综合推出‘面相’的主卦。"""
    scores = _score_face_parts(face_parts)
    if not scores:
        return ""
    # 显著度降序；显著度相同按器官优先级
    ordered = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], PART_PRIORITY.index(kv[0]) if kv[0] in PART_PRIORITY else 99)
    )
    top_parts = [p for p,_ in ordered[:2]]
    if not top_parts:
        return ""
    def part_to_hex(part: str, info: Dict[str,Any]) -> str:
        # 优先使用模型给的该器官“卦象”，否则用主映射
        raw = str((info or {}).get("卦象","")).strip()
        raw = re.sub(r"(卦（[^）]*）|卦|[。\.。\s]+)$", "", raw)
        if raw in WUXING:
            return raw
        return (FACIAL_MAP.get(part) or {}).get("primary","")
    h1 = part_to_hex(top_parts[0], (face_parts or {}).get(top_parts[0], {}))
    h2 = part_to_hex(top_parts[1], (face_parts or {}).get(top_parts[1], {})) if len(top_parts)>1 else ""
    if not h1 and not h2:
        return ""
    if h1 and not h2:
        return h1
    if h1 == h2:
        return h1
    return _resolve_two_hex_to_main(h1, h2)

# ---------- 辅助描述 ----------
def _persona_line(h: str) -> str:
    if not h or h not in WUXING: return ""
    ele = WUXING[h]["element"]; vir = WUXING[h]["virtue"]; sym = BAGUA_SYMBOLS.get(h, "")
    return f"{h}（{ele}·{sym}）：{vir}"

def _pair_relation_phrase(main_ele: str, other_ele: str) -> Tuple[str,str]:
    """返回（pair文字, 关系标签）。other 对 main 的关系：other生main=相生；other克main=相克。"""
    if not main_ele or not other_ele:
        return "", "相并"
    if main_ele == other_ele:
        return f"{other_ele}同{main_ele}", "比和"
    if SHENG.get(other_ele) == main_ele:
        return f"{other_ele}生{main_ele}", "相生"
    if KE.get(other_ele) == main_ele:
        return f"{other_ele}克{main_ele}", "相克"
    return f"{other_ele}并{main_ele}", "相并"

# ---------- 组合：八卦类比 ----------
def _synthesize_combo(hexes: List[str]) -> Tuple[str,str]:
    zh, sh, bh = (hexes + ["", "", ""])[:3]
    title = " + ".join([h for h in [zh, sh, bh] if h])

    def wx(h: str) -> str: return (WUXING.get(h) or {}).get("element","")

    def rel_phrase(main: str, other: str, which: str) -> Tuple[str,str,str]:
        """返回（pair文字, 关系, 推理句）。which: 'mf' 主-辅 / 'bm' 基-主"""
        if not (main in WUXING and other in WUXING):
            return "", "", ""
        pair, rel = _pair_relation_phrase(wx(main), wx(other))
        if which == "mf":
            reason = {
                "相生": f"{other}助{main}，主势更顺，长于把握节奏",
                "相克": f"{main}制{other}，风格紧而不松，需留回旋",
                "比和": f"{main}与{other}同气，执行干脆，调性统一",
                "相并": f"{main}与{other}侧重不同，可分工互补"
            }.get(rel, "")
        else:
            reason = {
                "相生": f"{other}为基托主，底盘给力，行稳致远",
                "相克": f"{main}受基牵制，宜辨旧新，以今断事",
                "比和": f"基与主同频，内外一致，少内耗",
                "相并": f"基与主各有所长，资源配置需取舍"
            }.get(rel, "")
        return pair, rel, reason

    lines: List[str] = []
    if zh in WUXING: lines.append("主" + _persona_line(zh))
    if sh in WUXING: lines.append("辅" + _persona_line(sh))
    if bh in WUXING: lines.append("基" + _persona_line(bh))

    pair_mf, rel_mf, rsn_mf = rel_phrase(zh, sh, "mf")
    pair_bm, rel_bm, rsn_bm = rel_phrase(zh, bh, "bm")
    if pair_mf: lines.append(f"主与辅（{pair_mf}）{rel_mf}：{rsn_mf}")
    if pair_bm: lines.append(f"基与主（{pair_bm}）{rel_bm}：{rsn_bm}")

    # 综合观（由三卦德性与两条关系合成，不写固定尾句）
    if zh in WUXING:
        seg = [f"以{zh}为纲（{WUXING[zh]['virtue']}）"]
        if sh in WUXING: seg.append(f"辅以{sh}（{WUXING[sh]['virtue']}）")
        if bh in WUXING: seg.append(f"基以{bh}（{WUXING[bh]['virtue']}）")
        lines.append("综合观：" + "，".join(seg))

    card_title = f"八卦类比（{title}）" if title else "八卦类比"
    return card_title, "\n".join(lines) if lines else "—"

# ---------- 三分象合并 ----------
def _extract_jingwen(s: str) -> Tuple[str,str]:
    if not isinstance(s, str): return s, ""
    s = s.strip()
    m = re.search(r"[（(]\s*经文提示\s*[:：]\s*(.+?)[)）]\s*$", s)
    if m:
        core = s[:m.start()].strip()
        hint = m.group(1).strip()
        return core, hint
    if "经文提示" in s:
        idx = s.find("经文提示")
        core = s[:idx].rstrip("，,。；; ")
        hint = s[idx:].replace("经文提示", "").lstrip("：: ").strip("（()） 。；;")
        return core, hint
    return s, ""

def _combine_sentence(desc: str, interp: str) -> str:
    if not desc and not interp: return ""
    desc  = _neutralize(_depronoun((desc or "").strip().rstrip("；;。")))
    interp = _neutralize(_depronoun((interp or "").strip().lstrip("——").lstrip("- ").strip().rstrip("；;。")))
    interp = re.sub(r"^(这种|此类|这类|其|这种姿态|这种神情|这种面容)[，、： ]*", "", interp)
    s = f"{desc}，{interp}" if (desc and interp) else (desc or interp)
    s = _strip_leading_punct(s)
    s = re.sub(r"[；;]+", "；", s)
    s = re.sub(r"，，+", "，", s)
    s = _dedupe_smart(s)
    return s

def _collect_traits_and_merge(ta: Dict[str,Any]) -> Tuple[List[str], Dict[str,Any]]:
    traits: List[str] = []
    new_ta: Dict[str,Any] = {}
    for key in ["姿态","神情","面容"]:
        o = (ta.get(key) or {}).copy()
        tend = (o.get("性格倾向") or "").strip().rstrip("；;。")
        if tend: traits.append(tend)
        merged = _combine_sentence(o.get("说明",""), o.get("解读",""))
        hexname = (o.get("卦象") or "").strip()
        hexname = re.sub(r"(卦（[^）]*）|卦|[。\.。\s]+)$", "", hexname)
        o["卦象"] = hexname

        # 经文提示：从“解读”尾部拆出；若无，用 _pick_jingwen（已在外层写入 _rel_mf/_rel_bm）
        pure, hint = _extract_jingwen(merged)
        if not hint and hexname:
            hint = _pick_jingwen(hexname, key, o.get("_rel_mf",""), o.get("_rel_bm",""))
        o["说明"] = ""
        o["解读"] = pure.strip()
        if hint:
            o["经文提示"] = hint
        o["性格倾向"] = ""
        new_ta[key] = o
    # 透传其它键
    for k in ta.keys():
        if k not in new_ta:
            new_ta[k] = ta[k]
    return traits, new_ta

# ---------- 事业/感情：句子化 ----------
def _human_status_sentence(s: set, domain: str) -> str:
    lines: List[str] = []
    if domain == "事业":
        if "乾" in s or "震" in s: lines.append("有计划也肯动手，遇事不拖")
        if "离" in s: lines.append("说清楚想法，能把原因讲明白")
        if "兑" in s or "巽" in s: lines.append("会把人拉进来一起做，气氛不紧张")
        if "坤" in s or "艮" in s: lines.append("先稳住，再决定，事情能落到结果上")
        if "坎" in s: lines.append("会先查清信息，留个备选方案")
    else:
        if "兑" in s: lines.append("聊天自然，愿意表达感受")
        if "离" in s: lines.append("讲道理也讲分寸")
        if "坤" in s: lines.append("重承诺，愿意花时间陪伴")
        if "坎" in s: lines.append("在意安全感，容易多想")
        if "震" in s or "乾" in s: lines.append("关键时能主动靠近")
        if "艮" in s: lines.append("尊重彼此边界")
    return "；".join(lines)

def _insight_for_domains(hexes: List[str]) -> Dict[str, str]:
    s = set([h for h in hexes if h])
    return {"事业": _human_status_sentence(s, "事业"),
            "感情": _human_status_sentence(s, "感情")}

def _imperative_suggestion_points(hexes: List[str], domain: str) -> List[str]:
    s = set([h for h in hexes if h])
    tips: List[str] = []
    if domain == "事业":
        if "乾" in s or "震" in s: tips.append("先把最重要的一件事定下来，今天推进一小步")
        if "离" in s: tips.append("当面讲清理由，再落到具体做法")
        if "兑" in s or "巽" in s: tips.append("找关键人聊一聊，先听对方的，再说自己的")
        if "坤" in s or "艮" in s: tips.append("把范围和时间说清楚，别一口吃成胖子")
        if "坎" in s: tips.append("做事前先核对信息，准备一个备选方案")
    else:
        if "兑" in s: tips.append("用平常语气聊心里的事，不用绕弯子")
        if "坤" in s: tips.append("答应的事尽量按时做到，让对方有底")
        if "离" in s: tips.append("把界限说清楚，让对方知道你的想法")
        if "震" in s or "乾" in s: tips.append("在重要时刻主动一点")
        if "坎" in s: tips.append("少靠猜，多确认")
        if "艮" in s: tips.append("给彼此一些独处时间")
    return tips[:3]

# ---------- 人格画像（动态生成，容错） ----------
def _gen_archetype(hexes: List[str], ta: Dict[str,Any], face_parts: Dict[str,Any]) -> str:
    main = hexes[0] if hexes and hexes[0] in WUXING else ""
    aux  = hexes[1] if len(hexes)>1 and hexes[1] in WUXING else ""
    base = hexes[2] if len(hexes)>2 and hexes[2] in WUXING else ""

    kw = {
        "乾":["主导","果断","坚毅"], "坤":["包容","稳重","承载"], "离":["明晰","表达","洞察"],
        "兑":["亲和","悦人","沟通"], "震":["起势","果敢","突破"], "巽":["协调","渗透","合众"],
        "艮":["定力","守度","边界"], "坎":["审慎","求证","韧性"]
    }
    def pick(h: str) -> List[str]:
        return kw.get(h, [])

    pool: List[str] = []
    pool += pick(main)[:2] + pick(aux)[:1] + pick(base)[:1]
    # 融合五官暗示（例：鼻→艮，眼→离/兑；仅轻权重堆叠）
    if isinstance(face_parts, dict):
        for info in face_parts.values():
            hh = str((info or {}).get("卦象","")).strip()
            hh = re.sub(r"(卦（[^）]*）|卦|[。\.。\s]+)$", "", hh)
            if hh in WUXING:
                pool += pick(hh)[:1]

    seen, words = set(), []
    for p in pool:
        if p and p not in seen:
            seen.add(p); words.append(p)
        if len(words) >= 3: break

    if not words:
        return "气象平衡"
    # 常见并置优先命名
    combos = [
        ({"主导","亲和"}, "刚柔相济"),
        ({"主导","明晰"}, "明断果决"),
        ({"稳重","亲和"}, "厚载和悦"),
        ({"定力","表达"}, "守度能言"),
    ]
    wset = set([w for w in words if len(w)<=2])
    for ks, lab in combos:
        if ks.issubset(wset):
            return lab
    return "、".join(words)

# ---------- 主输出组装 ----------
def _to_points(s: str, max_items: int = 4) -> List[str]:
    if not s: return []
    s = _neutralize(s)
    s = re.sub(r"[；;]+", "；", s.strip("；。 \n\t"))
    parts = [p.strip("；，。 \n\t") for p in s.split("；") if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip("；，。 \n\t") for p in re.split(r"[，,]", s) if p.strip()]
    seen, uniq = set(), []
    for p in parts:
        if p in seen: continue
        seen.add(p); uniq.append(p)
        if len(uniq) >= max_items: break
    return uniq

def _merge_status_and_detail(status: str, detail: str) -> str:
    detail_first = detail.split("。")[0].strip() if detail else ""
    detail_first = _neutralize(_strip_domain_lead(detail_first))
    status = _neutralize(_strip_domain_lead(status or ""))
    parts = [p for p in [status, detail_first] if p]
    text = "；".join(parts).rstrip("；")
    return _dedupe_smart(text)

def _clean_text(s: str) -> str:
    if not isinstance(s, str): return s
    s = s.replace("——", "，")
    s = re.sub(r"[；;]+", "；", s)
    s = re.sub(r"；([。！])", r"\1", s)
    s = re.sub(r"([。！？])；", r"\1", s)
    s = _depronoun(s)
    s = _neutralize(s)
    s = _strip_leading_punct(s)
    return _dedupe_smart(s)

def _coerce_output(data: Dict[str,Any]) -> Dict[str,Any]:
    out = dict(data)
    meta = out.get("meta") or {}
    if not isinstance(meta, dict): meta = {}
    out["meta"] = meta

    # ---------- 先从原始 ta 读取卦象，计算关系（供经文挑选用） ----------
    ta_raw = (meta.get("triple_analysis") or {}).copy()
    def get_hex_from(obj: Dict[str,Any], key: str) -> str:
        hx = (obj.get(key) or {}).get("卦象","") if isinstance(obj.get(key), dict) else ""
        hx = str(hx).strip()
        hx = re.sub(r"(卦（[^）]*）|卦|[。\.。\s]+)$", "", hx)
        return hx
    zh0, sh0, bh0 = get_hex_from(ta_raw,"姿态"), get_hex_from(ta_raw,"神情"), get_hex_from(ta_raw,"面容")
    def wx(h: str) -> str: return (WUXING.get(h) or {}).get("element","")
    rel_mf = rel_bm = ""
    if zh0 in WUXING and sh0 in WUXING:
        _, rel_mf = _pair_relation_phrase(wx(zh0), wx(sh0))
    if zh0 in WUXING and bh0 in WUXING:
        _, rel_bm = _pair_relation_phrase(wx(zh0), wx(bh0))
    # 把关系提示写回 ta_raw，供 _collect_traits_and_merge 时选经文
    for k in ["姿态","神情","面容"]:
        if k in ta_raw and isinstance(ta_raw[k], dict):
            ta_raw[k]["_rel_mf"] = rel_mf
            ta_raw[k]["_rel_bm"] = rel_bm

    # ---------- 三分象：合并&清洗（含经文提示动态挑选） ----------
    traits, ta = _collect_traits_and_merge(ta_raw)
    meta["triple_analysis"] = ta

    # ---------- 面相：若缺卦象→由“五官两强爻”推断 ----------
    fps = meta.get("face_parts") or {}
    if isinstance(ta.get("面容"), dict):
        if not (ta["面容"].get("卦象") or "").strip():
            infer_hex = _infer_face_hex_from_parts(fps)
            if infer_hex:
                ta["面容"]["卦象"] = infer_hex

    # ---------- 顶层 sections 用清洗后的文本 ----------
    out["sections"] = {
        "姿态": (ta.get("姿态") or {}).get("解读",""),
        "神情": (ta.get("神情") or {}).get("解读",""),
        "面相": (ta.get("面容") or {}).get("解读",""),
    }

    # ---------- 八卦类比卡片 ----------
    hexes = [
        (ta.get("姿态") or {}).get("卦象",""),
        (ta.get("神情") or {}).get("卦象",""),
        (ta.get("面容") or {}).get("卦象",""),
    ]
    card_title, card_content = _synthesize_combo(hexes)
    meta["combo_title"] = card_title.replace("八卦类比（", "").rstrip("）")
    meta["overview_card"] = {"title": card_title, "summary": card_content}

    # ---------- 人格画像：动态生成 ----------
    out["archetype"] = _gen_archetype(hexes, ta, fps)

    # ---------- 仅保留“图片清晰度” ----------
    cb = meta.get("confidence_breakdown") or {"图像清晰度": 0.30}
    meta["confidence_breakdown"] = {"图像清晰度": cb.get("图像清晰度", 0.30)}

    # ---------- 事业/感情：文本（状态/建议） ----------
    # 优先使用模型填的 meta.domains_status/suggestion；若缺，用规则生成
    status_text = meta.get("domains_status") or {}
    suggest_text = meta.get("domains_suggestion") or {}

    def _status_sentences(hexes: List[str], domain: str) -> str:
        base = _human_status_sentence(set([h for h in hexes if h]), domain)
        return _dedupe_smart(base)

    def _suggestion_sentences(hexes: List[str], domain: str) -> str:
        tips = _imperative_suggestion_points(hexes, domain)
        return _dedupe_smart("；".join(tips))

    meta["domains_status"] = {
        "事业": _dedupe_smart(_strip_domain_lead(status_text.get("事业",""))) or _status_sentences(hexes,"事业"),
        "感情": _dedupe_smart(_strip_domain_lead(status_text.get("感情",""))) or _status_sentences(hexes,"感情"),
    }
    meta["domains_suggestion"] = {
        "事业": _dedupe_smart(suggest_text.get("事业","")) or _suggestion_sentences(hexes,"事业"),
        "感情": _dedupe_smart(suggest_text.get("感情","")) or _suggestion_sentences(hexes,"感情"),
    }
    # 如果前端需要“词条化”要点，也保留列表
    meta["domains_status_list"] = {k:_to_points(v) for k,v in meta["domains_status"].items()}
    meta["domains_suggestion_list"] = {k:_to_points(v) for k,v in meta["domains_suggestion"].items()}

    # ---------- 五官文本统一标点 ----------
    if isinstance(fps, dict):
        for k, v in list(fps.items()):
            if not isinstance(v, dict): continue
            feat = (v.get("特征") or "").strip().strip("。；;，, ")
            expl = (v.get("解读") or "").strip()
            if feat and expl and feat in expl:
                expl = re.sub(re.escape(feat)+r"[，,；;]?", "", expl)
            v["特征"] = feat
            v["解读"] = re.sub(r"[；;]+", "；", expl).strip("；。 ")
    meta["face_parts"] = fps

    # ---------- 顶层清洗 ----------
    try:
        out["confidence"] = float(out.get("confidence", 0.0))
    except Exception:
        out["confidence"] = 0.0
    out["summary"] = _clean_text(out.get("summary",""))

    # ---------- 最终整体清洗 ----------
    out = json.loads(json.dumps(out, ensure_ascii=False))
    return out

# ---------- FastAPI ----------
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

def _to_data_url(content: bytes, content_type: str) -> str:
    return f"data:{content_type};base64,{base64.b64encode(content).decode('utf-8')}"

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
