# fastapi_app.py  (runtime v3.8.0, analysis logic v3.7.2, humanized phrasing + 主/辅/基 synthesis)
import os, base64, json, logging, traceback, re, math
from typing import Dict, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI

RUNTIME_VERSION = "3.8.0"
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

def _depronoun(s: str) -> str:
    """去掉“他/她/TA/你/其/对方/在…上/中/目前/近期”等口头起句，使语句客观中性"""
    if not isinstance(s, str): return s
    s = s.strip()
    s = re.sub(r"^(他|她|TA|你|对方|其)(的)?[，、： ]*", "", s)
    s = re.sub(r"^(在(事业|感情|生活)[上中]|目前|近期)[，、： ]*", "", s)
    return s

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
# 注：以下词义用于“组合推导”的提示，不是模板；可视为“术语词库”。
HEX_SUMMARY = {
    "乾":"自信·领导·果断",     # 易经术语：乾为天，健行，主刚健与主导
    "坤":"包容·稳定·承载",     # 坤为地，厚德载物，主柔顺与承载
    "震":"行动·突破·起势",     # 震为雷，动而行，主启动与开拓
    "巽":"协调·渗透·说服",     # 巽为风，入而不争，主渗透与调和
    "坎":"谨慎·探深·智谋",     # 坎为水，险而智，主风险意识与谋略
    "离":"明晰·表达·洞察",     # 离为火，附丽明，主洞察与表达
    "艮":"止定·边界·稳重",     # 艮为山，止于所当止，主定力与边界
    "兑":"亲和·交流·悦人"      # 兑为泽，说也，主欣悦与沟通
}

def _combine_sentence(desc: str, interp: str) -> str:
    """合并‘说明 + 解读’，去代词、去重复，让句子更像人说话"""
    if not desc and not interp: return ""
    desc = _depronoun((desc or "").strip().rstrip("；;。"))
    interp = _depronoun((interp or "").strip().lstrip("——").lstrip("- ").strip().rstrip("；;。"))
    # 去口头化起句
    interp = re.sub(r"^(这种|此类|这类|其|这种姿态|这种神情|这种面容)[，、： ]*", "", interp)
    s = f"{desc}，{interp}。" if (desc and interp) else f"{desc or interp}。"
    s = re.sub(r"[；;]+", "，", s)
    s = re.sub(r"，，+", "，", s)
    s = _dedupe_phrase(s)
    return s

def _synthesize_combo(hexes: List[str], ta: Dict[str,Any], traits: List[str]) -> str:
    """
    主/辅/基推导规则：
    - 主（姿态）定大势（相当于外在“动象/外卦”）；
    - 辅（神情）看运用与对人（承上启下，调性与交互方式）；
    - 基（面容）看底色与长期（地基/下卦，稳定倾向）。
    输出风格：先给“合象总括”，再以“主/辅/基”三分阐明；允许少量专业术语，并保持人话化。
    """
    zh, sh, bh = (hexes + ["", "", ""])[:3]  # 姿态=主，神情=辅，面容=基
    keys = [h for h in [zh, sh, bh] if h]
    if not keys:
        base = (ta.get("总结") or "") + ("；" + "；".join(traits) if traits else "")
        return base.strip("；")

    def w(h): return HEX_SUMMARY.get(h, "")

    words = "、".join([w(h) for h in keys if w(h)])
    lead = f"三象相合，取其象意为「{words}」。" if words else "三象相合。"

    # 主 / 辅 / 基 结构；术语注释见 HEX_SUMMARY
    p_main = f"主{zh}（{w(zh)}）" if zh else ""
    p_sub  = f"辅{sh}（{w(sh)}）" if sh else ""
    p_base = f"基{bh}（{w(bh)}）" if bh else ""

    seq = "，".join([p for p in [p_main, p_sub, p_base] if p])

    # 抽取每象解读的一小段，用于“落地感”
    snippets = []
    for k in ["姿态","神情","面容"]:
        inter = (ta.get(k) or {}).get("解读","")
        if inter: snippets.append(inter[:14])
    snippet_text = "；".join(snippets[:2])

    out = lead + seq
    if snippet_text:
        out += f"。{snippet_text}。"

    # 性格倾向（traits）适度融入，避免复读
    if traits:
        trait_text = "；".join([t[:12] for t in traits[:2]])
        if trait_text:
            out += trait_text + "。"

    out = re.sub(r"[；;]+", "；", out)
    out = _dedupe_phrase(out)
    return out

def _insight_for_domains(hexes: List[str]) -> Dict[str, str]:
    """基于卦象给“近期状态”的要点，弱模板、强卦意（更像即时表达）"""
    s = set([h for h in hexes if h])
    biz = []
    # 事业：从“势-法-守-险-合”五个角度勾勒
    if "乾" in s or "震" in s: biz.append("推进有力、节奏向前")
    if "离" in s: biz.append("表达清楚、复盘到位")
    if "兑" in s or "巽" in s: biz.append("善谈判协同、能带动人")
    if "坤" in s or "艮" in s: biz.append("落地稳、边界明、抗干扰")
    if "坎" in s: biz.append("风险意识强、方案留后手")

    love = []
    # 感情：从“亲密-表达-安全-主动-边界”五个角度勾勒
    if "兑" in s: love.append("氛围轻松、互动自然")
    if "离" in s: love.append("善表达想法、共情到位")
    if "坤" in s: love.append("重承诺与照顾")
    if "坎" in s: love.append("在意安全感、较敏感")
    if "震" in s or "乾" in s: love.append("关键时会主动")
    if "艮" in s: love.append("保持分寸与稳定")
    return {"事业": "；".join(biz), "感情": "；".join(love)}

def _merge_status_and_detail(status: str, detail: str) -> str:
    """把 GPT 的领域文案与我们基于卦象的状态要点合并，去代词、去复读"""
    detail_first = detail.split("。")[0].strip() if detail else ""
    detail_first = _depronoun(detail_first)
    parts = [p for p in [_depronoun(status), detail_first] if p]
    text = "；".join(parts).rstrip("；")
    return _dedupe_phrase(text)

def _imperative_suggestion(detail: str, hexes: List[str], domain: str) -> str:
    """
    以卦象导向生成“可执行建议”，避免千篇一律；
    - 事业：结合 乾/震（取势）、离（明晰表达）、兑/巽（协同）、坤/艮（稳守）、坎（风控）
    - 感情：结合 兑（亲和）、坤（承载）、离（表达）、乾/震（主动）、坎（止疑）、艮（边界）
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

    base = _depronoun(detail.strip()).rstrip("；")
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
        # 合并文本
        desc = (o.get("说明") or "")
        inter = (o.get("解读") or "")
        merged = _combine_sentence(desc, inter)
        o["说明"] = desc.strip().rstrip("；;。")
        o["解读"] = merged.strip()
        o["性格倾向"] = ""  # 倾向统一融入组合卡，避免复读
        new_ta[key] = o
    for k in ta.keys():
        if k not in new_ta:
            new_ta[k] = ta[k]
    return traits, new_ta

def _coerce_output(data: Dict[str,Any]) -> Dict[str,Any]:
    # 基本整理
    out = dict(data)
    meta = out.get("meta") or {}
    if not isinstance(meta, dict): meta = {}
    out["meta"] = meta

    ta = meta.get("triple_analysis") or {}
    traits, ta = _collect_traits_and_merge(ta)
    meta["triple_analysis"] = ta

    # 组合卦（标题）
    hexes = [(ta.get("姿态") or {}).get("卦象",""),
             (ta.get("神情") or {}).get("卦象",""),
             (ta.get("面容") or {}).get("卦象","")]
    combo_title = " + ".join([h for h in hexes if h])
    meta["combo_title"] = combo_title

    # 组合总结（主/辅/基推导）
    synthesized = _synthesize_combo(hexes, ta, traits)
    one = (ta.get("总结") or out.get("summary",""))
    overview = (synthesized or one).strip().rstrip("；;")
    meta["overview_card"] = {"title": f"🔮 卦象组合：{combo_title}" if combo_title else "🔮 卦象组合",
                             "summary": overview}

    # headline
    try:
        out["confidence"] = float(out.get("confidence",0.0))
    except Exception:
        out["confidence"] = 0.0
    arch = (out.get("archetype") or "").strip()
    meta["headline"] = {"tag": arch, "confidence": out["confidence"]}

    # 事业 / 感情：状态 + 建议
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

    # 全局文本轻清理：统一标点 + 去代词 + 去复读
    def _clean(s):
        if not isinstance(s, str): return s
        s = s.replace("——", "，")
        s = re.sub(r"[；;]+", "；", s)
        s = re.sub(r"；([。！])", r"\1", s)
        s = re.sub(r"([。！？])；", r"\1", s)
        s = _depronoun(s)
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
