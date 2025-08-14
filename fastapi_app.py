# fastapi_app.py  (runtime v3.8.3, analysis logic v3.7.2)
# 3.8.3: “易经可解释版”
# - 卦象组合：主/辅/基专业开头 + 关系与主风格的“意象→白话解释”两步落地
# - 三分象：开头轻量专业提示（如【乾·主导】），后接白话解释；强去重复
# - 事业/感情：状态与建议避免口号/堆词，改为“可感知场景 + 明确动作”的句子
# - 文本后处理：_dedupe_smart 保句读，清理“在…方面/…上”残影与代词
import os, base64, json, logging, traceback, re, math
from typing import Dict, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI

RUNTIME_VERSION = "3.8.5-ux"
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
      "5) 卦象组合：标题=三象卦名相加（如“艮 + 离 + 兑”），正文 90–150 字。"
      "6) 总结性格印象：20–40字，避免模板化；"
      "7) 人格标签 archetype：2–5字中文，如“外冷内热/主导型/谨慎型”。"
      "面相需拆成五官：在 meta.face_parts 中，给【眉/眼/鼻/嘴/颧/下巴】（任选5项覆盖）各写“特征（外观）”与“解读（基于易经）”。"
      "domains 仅从 ['金钱与事业','配偶与感情'] 选择；在 meta.domains_detail 中分别写 60–90 字建议文本。"
      "将结果通过 submit_analysis_v3 工具返回，并"+_json_hint()+"。语言：中文。"
    )
    user = "请按 3.7.2 风格分析图片，严格通过函数返回 JSON（不要输出自由文本）。"
    return [{"role":"system","content":sys},{"role":"user","content":user}]

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
    return "。".join(clean_sentences) + "。"

def _strip_domain_lead(s: str) -> str:
    if not isinstance(s, str): return s
    s = re.sub("^" + DOMAIN_LEADS + r"[，、： ]*", "", s.strip())
    s = re.sub(r"^上[，、： ]*", "", s)  # 裁掉“上，”之类残留
    return s

# --- 易经语义表 ---
HEX_SUMMARY = {
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

# —— 经文提示（简洁可读）——
CLASSIC_TIPS = {
    "乾": "《乾卦》亢龙有悔：过强则折，宜收锋敛势",
    "坤": "《坤卦》含弘光大：厚载不争，忌因循不决",
    "离": "《离卦》明两作：洞察高也易苛求，宜以明驭明",
    "坎": "《坎卦》履霜坚冰至：居安思危，先证后断",
    "震": "《震卦》震来虩虩：起势莫乱，定神而行",
    "巽": "《巽卦》小亨：入微渗透，忌反复不决",
    "艮": "《艮卦》艮其背：守界有度，忌僵硬不化",
    "兑": "《兑卦》和而不媚：悦人不失节，忌逢迎失真",
}

# —— 元素 → 推荐调和卦 & 说明（当“相克”出现时给出补法）——
ELEM_TO_HEX = {"金":"乾/兑", "火":"离", "水":"坎", "木":"震/巽", "土":"艮/坤"}
REMEDY_PHRASE = {
    # 用 X 制 Y 的直觉解释：只给一句“怎么做”
    ("火","金"): "以水制火（增配坎）：先收信息降温节奏，再定夺",
    ("木","土"): "以金伐木（增配乾/兑）：定规则、明边界，减少粘连",
    ("土","水"): "以木破土（增配震/巽）：先动起来、打通阻滞",
    ("水","火"): "以土泄水（增配艮/坤）：落到流程，稳住节律",
    ("金","木"): "以火炼金（增配离）：先讲清理由，让执行有说服力",
}

# —— 五官默认卦象（用于更具体的面相语言）——
FACE_HEX_DEFAULT = {
    "眉": "巽",      # 风，条理/判断
    "眼": "离",      # 火，明察/表达
    "鼻": "艮",      # 山，定力/边界
    "嘴": "兑",      # 泽，沟通/情绪
    "颧/下巴": "坤", # 地，承载/稳重
}

def _rel(a: str, b: str) -> str:
    if not a or not b: return ""
    if a == b: return "同气相求"
    if SHENG.get(a) == b: return "相生"
    if KE.get(a) == b: return "相克"
    return "相并"

def _harmony_suggestion(main_hex: str, other_hex: str) -> str:
    """当 other 克 main 时，给一个“补第三元素”的调和建议。"""
    if not (main_hex and other_hex): return ""
    A = (WUXING.get(main_hex) or {}).get("element","")
    B = (WUXING.get(other_hex) or {}).get("element","")
    if not A or not B: return ""
    # 若 B 克 A，则找一条 “X 制 B” 的建议
    if KE.get(B) == A:
        # 找对 REMEDY_PHRASE 的键： (B, A) 是“B克A”，我们需要“X制B”，按映射写死常见搭配
        for (x, y), phrase in REMEDY_PHRASE.items():
            if x == B:  # 这个映射是“用 ELEM(x) 去节制 x（=B）”
                return f"调和建议：{phrase}"
    return ""

def _derive_archetype(main: str, mf_rel: str, bm_rel: str) -> str:
    """从主卦与关系给个不生硬的人格标签（2~5字）。"""
    if main in ("乾","震"):
        if "相克" in (mf_rel or "") or "相克" in (bm_rel or ""): return "稳中带锋"
        if "相生" in (mf_rel or "") or "比和" in (mf_rel or ""): return "刚柔相济"
        return "张弛并进"
    if main in ("坤","艮"):
        if "相克" in (mf_rel or "") or "相克" in (bm_rel or ""): return "厚重而警"
        if "相生" in (mf_rel or "") or "比和" in (mf_rel or ""): return "厚实开朗"
        return "沉着中行"
    if main == "离":
        return "明断果决" if (mf_rel in ("相生","比和")) else "明慎并举"
    if main == "兑":
        return "和而不媚"
    if main == "坎":
        return "慎思笃行"
    if main == "巽":
        return "润物无声"
    return "中和之姿"

def _style_by_main_plain(h: str) -> str:
    # 主风格 → 白话解释
    if h in ("乾","震"): return "整体偏主动：看准就先做第一步"
    if h in ("坤","艮"): return "整体偏稳妥：先把基础打牢，再慢慢放大"
    if h in ("离",):     return "整体偏清楚表达：先把理由讲明白"
    if h in ("兑",):     return "整体偏亲和：先把关系处好，事就好办"
    if h in ("巽",):     return "整体偏协调：先融进去，再带着往前走"
    if h in ("坎",):     return "整体偏谨慎：先确认信息，再决定"
    return "整体风格平衡"

def _relation_plain(rel: str, pos: str) -> str:
    # rel1 主-辅；rel2 基-主
    if pos == "mf":  # main-fu
        if rel == "相生": return "主与辅能对上号：想法容易被理解与支持"
        if rel == "相克": return "主与辅有摩擦：先把期待讲清再推进"
        if rel == "同气相求": return "主与辅风格一致：效率高，但要留意不同意见"
        return "主与辅各有侧重：需要你来定次序和分工"
    else:            # base-main
        if rel == "相生": return "内在力量在支持主线：遇到变化也不容易乱"
        if rel == "相克": return "内心与目标有拉扯：先想清原则再出手"
        if rel == "同气相求": return "内外一致：想法和做法不打架"
        return "内在与目标各走各的：用简单规则把它们拢在一起"


def _pair_label(main_hex: str, other_hex: str, relation: str, which: str) -> str:
    """which: '辅' or '基'；输出“主X（金）×辅/基Y（土）：土生金 → 助力/制衡/比和/并行（白话）”"""
    if not (main_hex and other_hex and relation): return ""
    A = (WUXING.get(main_hex) or {}).get("element","")
    B = (WUXING.get(other_hex) or {}).get("element","")
    if not (A and B): return ""
    # 方向：other → main
    if SHENG.get(B) == A: arrow, tag = "生", "助力"
    elif KE.get(B) == A: arrow, tag = "克", "制衡"
    elif A == B:         arrow, tag = "同", "比和"
    else:                arrow, tag = "并", "并行"
    zh = f"主{main_hex}（{A}）×{which}{other_hex}（{B}）：{B}{arrow}{A} → {tag}"
    note = {"助力":"配合顺畅，优势互补", "制衡":"风格有张力，推进需更多协调", "比和":"同频协同，执行干脆", "并行":"关注点不同，各擅其长"}[tag]
    return zh + f"（{note}）"

def _persona_line(h: str) -> str:
    if not h: return ""
    ele = (WUXING.get(h) or {}).get("element","")
    vir = (WUXING.get(h) or {}).get("virtue","")
    return f"{h}（{ele}）：{vir}"

def _synthesize_combo(hexes: List[str], ta: Dict[str,Any], traits: List[str]) -> str:
    zh, sh, bh = (hexes + ["", "", ""])[:3]
    if not any([zh, sh, bh]): return ""

    wx   = lambda h: (WUXING.get(h) or {}).get("element","")
    sym  = lambda h: BAGUA_SYMBOLS.get(h, "")
    virt = lambda h: (WUXING.get(h) or {}).get("virtue","")

    def rel_from_to(a: str, b: str):
        A, B = wx(a), wx(b)
        if not A or not B: return "", ""
        if SHENG.get(A) == B: return f"{A}生{B}", "相生"
        if KE.get(A)   == B: return f"{A}克{B}", "相克"
        if A == B:           return f"{A}同{B}", "比和"
        return f"{A}并{B}", "相并"

    # 关系（按对主方向：辅→主 / 基→主）
    mf_pair, mf_rel = rel_from_to(sh, zh)
    bm_pair, bm_rel = rel_from_to(bh, zh)

    mf_note = {
        "相生": "同频协同，执行干脆",
        "相克": "风格有张力，推进需更多协调",
        "比和": "同频协同，执行干脆",
        "相并": "关注点不同，各擅其长",
    }.get(mf_rel, "")
    bm_note = {
        "相生": "根基助推，底盘给力",
        "相克": "旧经验牵扯，取舍要稳",
        "比和": "内外一致，表达与行动不打架",
        "相并": "资源与目标各有侧重",
    }.get(bm_rel, "")

    lines = []
    # 三行主辅基
    if zh: lines.append(f"主{zh}（{wx(zh)}·{sym(zh)}）：{virt(zh)}")
    if sh: lines.append(f"辅{sh}（{wx(sh)}·{sym(sh)}）：{virt(sh)}")
    if bh: lines.append(f"基{bh}（{wx(bh)}·{sym(bh)}）：{virt(bh)}")
    # 两条关系
    if mf_rel: lines.append(f"主与辅（{mf_pair}）{mf_rel}：{mf_note}")
    if bm_rel: lines.append(f"基与主（{bm_pair}）{bm_rel}：{bm_note}")
    # 若出现相克，给调和建议
    if mf_rel == "相克":
        hs = _harmony_suggestion(zh, sh)
        if hs: lines.append(hs)
    if bm_rel == "相克":
        hs = _harmony_suggestion(zh, bh)
        if hs: lines.append(hs)

    # —— 三段式收束：外在气象 / 内在基质 / 运势格局 —— 
    def kw(h: str):
        s = HEX_SUMMARY.get(h, "")
        return (s.split("·")[0] if s else "", s.split("·")[1] if ("·" in s) else "")
    main_kw, main_trait = kw(zh)
    fu_kw,   fu_trait   = kw(sh)
    # 外在气象（主卦）
    if zh:
        lines.append(f"外在气象：以{zh}为纲（{main_kw}），先立标准再带节奏。{CLASSIC_TIPS.get(zh,'').split('：')[0]}可作镜鉴。")
    # 内在基质（辅/基作用）
    inner_frag = []
    if sh: inner_frag.append(f"辅{sh}助{zh}（{mf_rel or '并行'}）")
    if bh: inner_frag.append(f"基{bh}托底（{bm_rel or '并行'}）")
    if inner_frag: lines.append("内在基质：" + "；".join(inner_frag) + "。")
    # 运势格局（整体定性——不写“看准就先做第一步”这种口号）
    left  = main_trait or "主导力"
    right = fu_trait   or "亲和力"
    soft  = "外刚内柔" if (mf_rel in ("相生","比和") and bm_rel in ("相生","比和")) else "张弛有度"
    lines.append(f"运势格局：{soft}，既有{left}，又具{right}。以稳推进、分层决断为宜。")

    return "\n".join(lines)


def _human_status_sentence(s: set, domain: str) -> str:
    lines = []
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
    return {
        "事业": _human_status_sentence(s, "事业"),
        "感情": _human_status_sentence(s, "感情"),
    }

def _imperative_suggestion(detail: str, hexes: List[str], domain: str) -> str:
    s = set([h for h in hexes if h])
    tips = []
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
    add = "；".join(tips[:3])
    return (add + "。") if add else ""


def _imperative_suggestion_points(hexes: List[str], domain: str) -> List[str]:
    s = set([h for h in hexes if h])
    tips = []
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

# ---- 三分象合句 & 专业提示 ----
def _combine_sentence(desc: str, interp: str) -> str:
    if not desc and not interp:
        return ""
    # 基础清洗
    desc  = _neutralize(_depronoun((desc or "").strip()))
    interp = _neutralize(_depronoun((interp or "").strip()))
    # 去掉破折号/前导语
    interp = re.sub(r"^(——|-+)\s*", "", interp)
    interp = re.sub(r"^(这种|此类|这类|其|这种姿态|这种神情|这种面容|这种面相)[，、： ]*", "", interp)
    # 合并
    s = f"{desc}，{interp}" if (desc and interp) else (desc or interp)
    # 统一标点 & 去掉句首孤立标点
    s = re.sub(r"[；;]+", "；", s)
    s = re.sub(r"^[，,。；;：:]+", "", s)      # 关键：去掉“。，”之类
    s = re.sub(r"，，+", "，", s)
    return _dedupe_smart(s)

def _collect_traits_and_merge(ta: Dict[str,Any]) -> (List[str], Dict[str,Any]):
    traits, new_ta = [], {}
    for key in ["姿态","神情","面容"]:
        o = (ta.get(key) or {}).copy()
        tend = (o.get("性格倾向") or "").strip().rstrip("；;。")
        if tend: traits.append(tend)

        desc   = (o.get("说明") or "")
        inter  = (o.get("解读") or "")
        merged = _combine_sentence(desc, inter)

        hexname = re.sub(r"(卦（[^）]*）|卦|[。\.。\s]+)$", "", (o.get("卦象") or "").strip())
        o["卦象"] = hexname

        pro = ""
        if hexname in HEX_SUMMARY:
            kw = HEX_SUMMARY[hexname].split("·")[1] if "·" in HEX_SUMMARY[hexname] else HEX_SUMMARY[hexname]
            pro = f""

        tip = CLASSIC_TIPS.get(hexname, "")
        if pro: merged = f"{pro} {merged}".strip()
        if tip: merged = f"{merged}（经文提示：{tip}）".strip()

        o["说明"] = ""
        o["解读"] = merged
        o["性格倾向"] = ""
        new_ta[key] = o

    for k in ta.keys():
        if k not in new_ta: new_ta[k] = ta[k]
    return traits, new_ta

def _to_points(s: str, max_items: int = 4) -> List[str]:
    """Split a sentence by Chinese semicolons/commas into 2-4 concise bullet points."""
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

def _confidence_breakdown(out: Dict[str,Any]) -> Dict[str,Any]:
    meta = out.get("meta") or {}
    ta   = meta.get("triple_analysis") or {}

    # 1) 图像清晰度：无法读像素，这里用“五官条目齐全度” & 文本噪声占比当 proxy
    face = meta.get("face_parts") or {}
    face_count = sum(1 for k,v in (face or {}).items() if isinstance(v,dict) and (v.get("特征") or v.get("解读")))
    clarity = min(1.0, 0.5 + 0.1 * face_count)  # 0.5~1.0 之间

    # 2) 卦象一致性：三象是否齐、是否有比和/相生
    hexes = [(ta.get("姿态") or {}).get("卦象",""), (ta.get("神情") or {}).get("卦象",""), (ta.get("面容") or {}).get("卦象","")]
    present = [h for h in hexes if h]
    uniq = len(set(present))
    if len(present) < 2:
        hx_cons = 0.5
    elif uniq == 1:
        hx_cons = 1.0  # 完全比和
    else:
        hx_cons = 0.8  # 有分工但不冲突，给 0.8

    # 3) 特征显著性：三分象“解读”长度 & 去重后密度
    texts = []
    for k in ("姿态","神情","面容"):
        t = (ta.get(k) or {}).get("解读","")
        if t: texts.append(t)
    avg_len = sum(len(t) for t in texts)/max(1,len(texts))
    salience = 0.6 if avg_len < 30 else (0.8 if avg_len < 80 else 1.0)

    # 配置权重（可微调）
    w1, w2, w3 = 0.30, 0.40, 0.30
    score = w1*clarity + w2*hx_cons + w3*salience
    # 不覆盖模型自带 confidence，只给“解释”
    return {
        "weights":{"图像清晰度":w1,"卦象一致性":w2,"特征显著性":w3},
        "scores":{"图像清晰度":round(clarity,2),"卦象一致性":round(hx_cons,2),"特征显著性":round(salience,2)},
        "explain":"分项权重为经验值，用五官覆盖度、三象齐整度/比和度、与文本密度作为近似指标，仅用于帮助理解“90%把握”的来源，不代表统计学置信区间。"
    }

def _coerce_output(data: Dict[str,Any]) -> Dict[str,Any]:
    out = dict(data)
    meta = out.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    out["meta"] = meta

    # 1) 合并三分象（会做去重/去“卦/。”等清洗）
    ta = meta.get("triple_analysis") or {}
    traits, ta = _collect_traits_and_merge(ta)
    meta["triple_analysis"] = ta

    # 2) 用合并后的三分象回填顶层 sections，修掉“。，/。；”等
    out["sections"] = {
        "姿态": (ta.get("姿态") or {}).get("解读", ""),
        "神情": (ta.get("神情") or {}).get("解读", ""),
        "面相": (ta.get("面容") or {}).get("解读", ""),
    }

    # 3) 组合标题与总览（并裁掉首行标题）
    hexes = [
        (ta.get("姿态") or {}).get("卦象", ""),
        (ta.get("神情") or {}).get("卦象", ""),
        (ta.get("面容") or {}).get("卦象", "")
    ]
    combo_title = " + ".join([h for h in hexes if h])
    meta["combo_title"] = combo_title

    synthesized = _synthesize_combo(hexes, ta, traits)
    one = (ta.get("总结") or out.get("summary", ""))
    overview = (synthesized or one).strip().rstrip("；;")
    if overview.startswith("🔮 卦象组合"):
        lines = overview.splitlines()
        if len(lines) > 1:
            overview = "\n".join(lines[1:]).strip()
    meta["overview_card"] = {
        "title": f"🔮 卦象组合：{combo_title}" if combo_title else "🔮 卦象组合",
        "summary": overview
    }

    # 4) 可信度与人物标签抬头
    try:
        out["confidence"] = float(out.get("confidence", 0.0))
    except Exception:
        out["confidence"] = 0.0
    arch = (out.get("archetype") or "").strip()
    meta["headline"] = {"tag": arch, "confidence": out["confidence"]}

    # 5) 事业/感情：状态与建议（合并为更人话的要点 + 列表）
    dd = meta.get("domains_detail") or {}
    status = _insight_for_domains(hexes)
    merged_status = {
        "事业": _merge_status_and_detail(status.get("事业", ""), dd.get("金钱与事业", "")),
        "感情": _merge_status_and_detail(status.get("感情", ""), dd.get("配偶与感情", "")),
    }
    meta["domains_status"] = merged_status
    meta["domains_status_list"] = {k: _to_points(v) for k, v in merged_status.items()}
    meta["domains_suggestion"] = {
        "事业": _imperative_suggestion(dd.get("金钱与事业", ""), hexes, "事业"),
        "感情": _imperative_suggestion(dd.get("配偶与感情", ""), hexes, "感情"),
    }
    meta["domains_suggestion_list"] = {
        "事业": _imperative_suggestion_points(hexes, "事业"),
        "感情": _imperative_suggestion_points(hexes, "感情"),
    }

    # 6) 文本清洗器
    def _clean(s):
        if not isinstance(s, str):
            return s
        s = s.replace("——", "，")
        s = re.sub(r"[；;]+", "；", s)
        s = re.sub(r"；([。！])", r"\1", s)
        s = re.sub(r"([。！？])；", r"\1", s)
        s = _depronoun(s)
        s = _neutralize(s)
        return _dedupe_smart(s)

    out["summary"] = _clean(out.get("summary", ""))
    out["archetype"] = _clean(out.get("archetype", ""))

    def _deep_clean(x):
        if isinstance(x, dict):
            return {k: _deep_clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_deep_clean(v) for v in x]
        return _clean(x)

    # 7) 五官细节：若“解读”里重复“特征”，则去重，统一标点
    fps = meta.get("face_parts") or {}
    if isinstance(fps, dict):
        for k, v in list(fps.items()):
            if not isinstance(v, dict):
                continue
            feat = (v.get("特征") or "").strip().strip("。；;，, ")
            expl = (v.get("解读") or "").strip()
            if feat and expl and feat in expl:
                import re as _re
                expl = _re.sub(_re.escape(feat) + r"[，,；;]?", "", expl)
            v["特征"] = feat
            v["解读"] = re.sub(r"[；;]+", "；", expl).strip("；。 ")
    meta["face_parts"] = fps

    # 8) 全量深度清洗
    out["meta"] = _deep_clean(meta)
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
