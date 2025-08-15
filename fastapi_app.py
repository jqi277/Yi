# fastapi_app.py  (runtime v3.8.6-ux, analysis logic v3.7.2)
# 仅按用户要求改动：
# - 可信度整体移除，仅保留 meta.image_quality（图片清晰度）
# - 人格画像动态生成，不用固定预设词
# - 三分象解读去掉【】专业标签；经文提示独立行
# - “三才之道”改为“八卦类比”
# - 输出 domains_long（事业/感情：状态/建议 为段落文本）
import os, base64, json, logging, traceback, re
from typing import Dict, Any, List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

RUNTIME_VERSION = "3.8.6-ux"
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

JINGWEN_HINT = {
    "乾":"《乾》亢龙有悔：过强则折，宜收锋敛势",
    "坤":"《坤》含弘光大：厚载不争，忌因循不决",
    "艮":"《艮》艮其背：守界有度，忌僵硬不化",
    "兑":"《兑》和兑以说：以悦达人，忌过度迎合",
    "离":"《离》明两作：明察而易苛，宜明而不燥",
    "坎":"《坎》习坎重险：慎审求证，忌反复犹疑",
    "震":"《震》震来虩虩：初动宜稳，忌躁进冒进",
    "巽":"《巽》小亨利往：渗透有方，忌优柔寡断",
}

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
            "\"meta\":{\"triple_analysis\":{\"姿态\":{\"说明\":\"…\",\"卦象\":\"艮\",\"解读\":\"…\",\"性格倾向\":\"…\"},\"神情\":{…},\"面容\":{…}},"
            "\"face_parts\":{\"眉\":{\"特征\":\"…\",\"卦象\":\"…\",\"解读\":\"…\"}},"
            "\"domains_detail\":{\"金钱与事业\":\"…\",\"配偶与感情\":\"…\"}}}")

def _prompt_for_image_v372():
    sys = (
      "你是 Selfy AI 的易经观相助手（v3.7.2 风格）。"
      "严格按“三象四段式”分析：【姿态/神情/面容】三部分。每部分必须包含："
      "1) 说明：1句；2) 卦象：仅写一个卦名；3) 解读：1–2句；4) 性格倾向：1–2句。"
      "再给出：卦象组合（90–150字）、总结印象（20–40字）、人格标签（2–5字）。"
      "面相需拆成五官：在 meta.face_parts 写【眉/眼/鼻/嘴/颧或下巴】的“特征/卦象/解读”。"
      "domains 选择 ['金钱与事业','配偶与感情'] 并在 meta.domains_detail 写 60–90字建议。"
      "通过 submit_analysis_v3 工具返回，并"+_json_hint()+"。语言：中文。"
    )
    user = "请按 3.7.2 风格分析图片，严格通过函数返回 JSON（不要输出自由文本）。"
    return [{"role":"system","content":sys},{"role":"user","content":user}]

DOMAIN_LEADS = r"(在(金钱与事业|配偶与感情|事业|感情)(方面|中|里)?|目前|近期|当下)"
_STOPWORDS = r"(姿态|神情|面容|整体|气质|形象|给人以|一种|以及|并且|而且|更显|显得|展现出|流露出|透露出)"

def _strip_leading_punct(s: str) -> str:
    if not isinstance(s, str): return s
    return re.sub(r"^[\s\.,;，。；、:：\-~·【】（）()]+", "", s)

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
    k = re.sub(_STOPWORDS, "", s); k = re.sub(r"[的地得]", "", k); k = re.sub(r"\s+", "", k)
    return k

def _dedupe_smart(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.strip("。；，,; ")
    sentences = re.split(r"[。！？]", s)
    clean = []
    for sen in sentences:
        sen = sen.strip("，,;； ")
        if not sen:
            continue
        parts = re.split(r"[，,；;]", sen)
        seen, kept = set(), []
        for p in parts:
            t = p.strip()
            if not t:
                continue
            ck = _canon_key(t)
            if ck and ck not in seen:
                seen.add(ck)
                kept.append(t)
        if kept:
            clean.append("，".join(kept))
    return ("。".join(clean) + "。") if clean else ""

def _strip_domain_lead(s: str) -> str:
    if not isinstance(s, str): return s
    s = re.sub("^" + DOMAIN_LEADS + r"[，、： ]*", "", s.strip())
    s = re.sub(r"^上[，、： ]*", "", s)
    return s

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

EIGHT_TRIGRAMS = set(list("乾坤震巽坎离艮兑"))

def _normalize_hex_name(raw: str) -> str:
    """把模型可能输出的“泰/否/需/讼/乾卦/坤（地）/艮卦（山）/离火”等，规范成 8 卦之一；否则返回空串。"""
    if not isinstance(raw, str): 
        return ""
    s = raw.strip()
    # 快速命中：如果本身就是八卦之一
    if s in EIGHT_TRIGRAMS:
        return s
    # 去掉“卦”“卦（…）”以及括注
    s = re.sub(r"(卦（[^）]*）|卦|[()（）\s])", "", s)
    # 在字符串里找第一个八卦字
    for ch in s:
        if ch in EIGHT_TRIGRAMS:
            return ch
    # 六十四卦名里含“天地雷风水火山泽”，取其上下卦常见字，也做一次兜底
    for ch in "天地雷风水火山泽":
        # 映射到八卦（天→乾 地→坤 雷→震 风→巽 水→坎 火→离 山→艮 泽→兑）
        m = {"天":"乾","地":"坤","雷":"震","风":"巽","水":"坎","火":"离","山":"艮","泽":"兑"}.get(ch)
        if m and ch in raw:
            return m
    return ""

def _pair_relation_phrase(a_ele: str, b_ele: str) -> (str, str):
    if not a_ele or not b_ele: return "", "相并"
    if a_ele == b_ele: return f"{b_ele}同{a_ele}", "比和"
    if SHENG.get(b_ele) == a_ele: return f"{b_ele}生{a_ele}", "相生"
    if KE.get(b_ele) == a_ele:    return f"{b_ele}克{a_ele}", "相克"
    return f"{b_ele}并{a_ele}", "相并"

def _relation_note(rel: str, which: str) -> str:
    if which == "mf":
        return {"相生":"配合顺畅，优势互补","相克":"风格有张力，推进需更多协调","比和":"同频协同，执行干脆","相并":"关注点不同，各擅其长"}.get(rel,"各擅其长")
    return {"相生":"根基助推，底盘给力","相克":"旧经验牵扯，当下取舍要稳","比和":"内外一致，表达与行动不打架","相并":"资源与目标各有侧重"}.get(rel,"内外各有侧重")

def _persona_line(h: str) -> str:
    if not h: return ""
    ele = (WUXING.get(h) or {}).get("element","")
    vir = (WUXING.get(h) or {}).get("virtue","")
    sym = BAGUA_SYMBOLS.get(h, "")
    return f"{h}（{ele}·{sym}）：{vir}"

def _synthesize_combo(hexes: List[str]) -> (str, str):
    zh, sh, bh = (hexes + ["", "", ""])[:3]
    title = " + ".join([h for h in [zh, sh, bh] if h])
    lines: List[str] = []
    if zh: lines.append("主" + _persona_line(zh))
    if sh: lines.append("辅" + _persona_line(sh))
    if bh: lines.append("基" + _persona_line(bh))
    def wx(h: str) -> str: return (WUXING.get(h) or {}).get("element","")
    if zh and sh:
        pair, rel = _pair_relation_phrase(wx(zh), wx(sh))
        lines.append(f"主与辅（{pair}）{rel}：" + _relation_note(rel, "mf"))
    if bh and zh:
        pair, rel = _pair_relation_phrase(wx(zh), wx(bh))
        lines.append(f"基与主（{pair}）{rel}：" + _relation_note(rel, "bm"))
    if zh:
        zl = (WUXING.get(zh) or {}).get("virtue","")
        sl = (WUXING.get(sh) or {}).get("virtue","") if sh else ""
        bl = (WUXING.get(bh) or {}).get("virtue","") if bh else ""
        desc = [f"以{zh}为纲（{zl}）"]
        if sh: desc.append(f"辅以{sh}（{sl}）")
        if bh: desc.append(f"基以{bh}（{bl}）")
        lines.append("易理综合：" + "，".join(desc) + "。三象相成：外在取势，内里守度，整体气象稳健而不滞、开张而不散。")
    content = "\n".join(lines)
    card_title = f"八卦类比（{title}）" if title else "八卦类比"
    return card_title, content

def _extract_jingwen(s: str) -> (str, str):
    if not isinstance(s, str): return s, ""
    s = s.strip()
    m = re.search(r"[（(]\s*经文提示\s*[:：]\s*(.+?)[)）]\s*$", s)
    if m:
        core = s[:m.start()].strip(); hint = m.group(1).strip(); return core, hint
    if "经文提示" in s:
        idx = s.find("经文提示")
        core = s[:idx].rstrip("，,。；; "); hint = s[idx:].replace("经文提示","").lstrip("：: ").strip("（()） 。；;")
        return core, hint
    return s, ""

def _combine_sentence(desc: str, interp: str) -> str:
    if not desc and not interp: return ""
    desc  = _neutralize(_depronoun((desc or "").strip().rstrip("；;。")))
    interp = _neutralize(_depronoun((interp or "").strip().lstrip("——").lstrip("- ").strip().rstrip("；;。")))
    interp = re.sub(r"^(这种|此类|这类|其|这种姿态|这种神情|这种面容)[，、： ]*", "", interp)
    s = f"{desc}，{interp}" if (desc and interp) else (desc or interp)
    s = _strip_leading_punct(s); s = re.sub(r"[；;]+", "；", s); s = re.sub(r"，，+", "，", s)
    return _dedupe_smart(s)

def _collect_traits_and_merge(ta: Dict[str,Any]) -> (List[str], Dict[str,Any]):
    traits: List[str] = []; new_ta: Dict[str,Any] = {}
    for key in ["姿态","神情","面容"]:
        o = (ta.get(key) or {}).copy()
        tend = (o.get("性格倾向") or "").strip().rstrip("；;。")
        if tend: traits.append(tend)
        merged = _combine_sentence(o.get("说明") or "", o.get("解读") or "")
        hexname_raw = (o.get("卦象") or "").strip()
        hexname = _normalize_hex_name(hexname_raw)
        o["卦象"] = hexname  # 现在一定是 乾坤震巽坎离艮兑 或者 ""
        pure, hint = _extract_jingwen(merged)
        if not hint and hexname: hint = JINGWEN_HINT.get(hexname, "")
        o["说明"] = ""
        o["解读"] = pure.strip()                  # 不再加【卦·关键词】
        if hint: o["经文提示"] = hint
        o["性格倾向"] = ""
        new_ta[key] = o
    for k in ta.keys():
        if k not in new_ta: new_ta[k] = ta[k]
    return traits, new_ta

def _clean_text(s: str) -> str:
    if not isinstance(s, str): return s
    s = s.replace("——", "，")
    s = re.sub(r"[；;]+", "；", s)
    s = re.sub(r"；([。！])", r"\1", s)
    s = re.sub(r"([。！？])；", r"\1", s)
    s = _depronoun(s); s = _neutralize(s); s = _strip_leading_punct(s)
    return _dedupe_smart(s)

def _gen_archetype(hexes: List[str], ta: Dict[str,Any], face_parts: Dict[str,Any]) -> str:
    """
    依据主/辅/基三卦（已尽量规范为八卦），结合五官细节，生成 2–4 个词的人格标签；
    若命中常见并置，压缩为 4 字短语。全程容错、零 KeyError。
    """
    # 八卦→关键词池
    kw = {
        "乾":"主导/果断/坚毅",
        "坤":"包容/稳重/承载",
        "离":"明晰/表达/洞察",
        "兑":"亲和/悦人/沟通",
        "震":"起势/果敢/突破",
        "巽":"协调/渗透/合众",
        "艮":"定力/守度/边界",
        "坎":"审慎/求证/韧性",
    }

    # 规范化（若外部已做 _normalize_hex_name，这里仍二次兜底）
    def _norm(h: str) -> str:
        try:
            return _normalize_hex_name(h)  # 外部已有工具则用之
        except NameError:
            return h if isinstance(h, str) and h in kw else ""

    main = _norm(hexes[0]) if hexes and hexes[0] else ""
    aux  = _norm(hexes[1]) if len(hexes) > 1 else ""
    base = _norm(hexes[2]) if len(hexes) > 2 else ""

    # 安全取词
    def pick(h: str) -> List[str]:
        if not h or h not in kw:
            return []
        return kw[h].split("/")

    pool: List[str] = []
    pool += pick(main)[:2]  # 主卦优先拿 2 个
    pool += pick(aux)[:1]   # 辅卦 1 个
    pool += pick(base)[:1]  # 基卦 1 个

    # 从五官细节里提取加分词（可选）
    # 轻量关键词映射：命中即加，避免重复
    face_bonus_map = {
        "眉": [("平直","坚定"), ("浓","果敢"), ("弯曲","亲和")],
        "眼": [("明亮","洞察"), ("坚定","坚毅"), ("柔和","包容")],
        "鼻": [("高挺","目标感"), ("端正","稳重")],
        "嘴": [("唇厚","表达"), ("上扬","乐观")],
        "颧": [("分明","主导"), ("高","进取")],
        "下巴": [("圆润","承载"), ("有角度","边界")],
    }
    try:
        for part, confs in face_bonus_map.items():
            info = face_parts.get(part) if isinstance(face_parts, dict) else None
            txt = ""
            if isinstance(info, dict):
                txt = f"{info.get('特征','')}{info.get('解读','')}"
            elif isinstance(info, str):
                txt = info
            txt = (txt or "").strip()
            for needle, tag in confs:
                if needle in txt and tag not in pool:
                    pool.append(tag)
    except Exception:
        pass  # 面部信息缺失或结构异常时忽略

    # 去重，最多取 3~4 个
    seen, words = set(), []
    for p in pool:
        p = (p or "").strip()
        if not p:
            continue
        if p not in seen:
            seen.add(p)
            words.append(p)
        if len(words) >= 4:
            break

    # 若仍为空，用主卦风格兜底
    if not words:
        style = _style_by_main_plain(main) if main else "整体风格平衡"
        # 去掉“整体偏”/“整体”，保留核心词
        style = style.replace("整体偏", "").replace("整体", "")
        words = [style or "平衡"]

    # 常见并置 → 4 字短语（优先命中两词组合）
    mapping = {
        frozenset(("主导","亲和")): "刚柔相济",
        frozenset(("主导","明晰")): "明断果决",
        frozenset(("稳重","亲和")): "厚载和悦",
        frozenset(("定力","表达")): "守度能言",
        frozenset(("果敢","明晰")): "明勇并济",
        frozenset(("审慎","表达")): "谨言有度",
    }
    # 只用长度<=2 的精炼词参与 4 字匹配
    short_key = frozenset([w for w in words if len(w) <= 2])
    label = mapping.get(short_key)
    if label:
        return label

    # 否则返回“、”连接的词组（2~4 个）
    return "、".join(words[:4])

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

def _merge_status_and_detail(status: str, detail: str) -> str:
    detail_first = detail.split("。")[0].strip() if detail else ""
    detail_first = _neutralize(_strip_domain_lead(detail_first))
    status = _neutralize(_strip_domain_lead(status or ""))
    parts = [p for p in [status, detail_first] if p]
    text = "；".join(parts).rstrip("；")
    return _dedupe_smart(text)

def _compose_long_paragraphs(hexes: List[str], dd: Dict[str,str]) -> Dict[str, Dict[str,str]]:
    # 用象结果合成段落（状态/建议）
    st = _insight_for_domains(hexes)
    long_ = {}
    long_["事业"] = {
        "状态": _merge_status_and_detail(st.get("事业",""), dd.get("金钱与事业","")),
        "建议": _clean_text(dd.get("金钱与事业",""))
    }
    long_["感情"] = {
        "状态": _merge_status_and_detail(st.get("感情",""), dd.get("配偶与感情","")),
        "建议": _clean_text(dd.get("配偶与感情",""))
    }
    return long_

def _to_data_url(content: bytes, content_type: str) -> str:
    return f"data:{content_type};base64,{base64.b64encode(content).decode('utf-8')}"

def _call_openai(messages):
    if client is None: raise RuntimeError("OpenAI client not initialized")
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

# ---------------- 主输出组装 ----------------
def _coerce_output(data: Dict[str,Any]) -> Dict[str,Any]:
    out = dict(data)
    meta = out.get("meta") or {}
    if not isinstance(meta, dict): meta = {}
    out["meta"] = meta

    # 仅保留图片清晰度（若上游未给则按0.90示意）
    meta["image_quality"] = float(meta.get("image_quality", 0.90))

    # 三分象清洗与合并
    ta = meta.get("triple_analysis") or {}
    traits, ta = _collect_traits_and_merge(ta)
    meta["triple_analysis"] = ta

    # 顶层 sections 用清洗后的文本
    out["sections"] = {
        "姿态": (ta.get("姿态") or {}).get("解读",""),
        "神情": (ta.get("神情") or {}).get("解读",""),
        "面相": (ta.get("面容") or {}).get("解读",""),
    }

    # 八卦类比卡片
    hexes = [
        (ta.get("姿态") or {}).get("卦象",""),
        (ta.get("神情") or {}).get("卦象",""),
        (ta.get("面容") or {}).get("卦象",""),
    ]
    card_title, card_content = _synthesize_combo(hexes)
    meta["combo_title"] = card_title.replace("八卦类比（", "").rstrip("）")
    meta["overview_card"] = {"title": card_title, "summary": card_content}

    # 动态人格画像
    face_parts = meta.get("face_parts") or {}
    out["archetype"] = _gen_archetype(hexes, ta, face_parts)

    # 顶层 summary/清洗（若模型给了就清理；否则用 overview 第一段）
    overview_first = card_content.split("\n")[-1] if card_content else ""
    out["summary"] = _clean_text(out.get("summary", overview_first))

    # 事业/感情：生成段落文本
    dd = meta.get("domains_detail") or {}
    meta["domains_long"] = _compose_long_paragraphs(hexes, dd)

    # face_parts 统一标点（保留并“弄回来”）
    fps = meta.get("face_parts") or {}
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

    # 抹掉上游 confidence（避免前端显示）
    out["confidence"] = 0.0

    return json.loads(json.dumps(out, ensure_ascii=False))

# ---------------- FastAPI ----------------
@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/")
def root():
    return HTMLResponse("<h3>Selfy AI</h3><a href='/docs'>/docs</a> · <a href='/mobile'>/mobile</a>")

@app.head("/")
def root_head(): return Response(status_code=200)

@app.get("/version")
def version(): return {"runtime":RUNTIME_VERSION,"analysis":ANALYSIS_VERSION,"schema":SCHEMA_ID,"debug":DEBUG}

@app.get("/mobile")
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
        data_url = f"data:{ct};base64,{base64.b64encode(raw).decode('utf-8')}"
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
