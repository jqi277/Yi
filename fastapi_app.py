# fastapi_app.py  (runtime v3.9.0, AI-led YiJing)
# 变更要点：
# - 将“分析与语言生成”尽量交由 AI；后端仅提供结构 Schema 与最小清洗/兜底。
# - 五官细节：指令中强调“先以爻判读五官→再综合为整体卦象与三象(姿态/神情/面容)”。
# - 三象组合与事业/感情建议均由 AI 直接生成文案；Python 不再拼接句子或添加经文模板。
# - 保留前端所需字段：summary、archetype、confidence、sections、domains、meta.*。
# - 仅做轻度规范化与容错（如把 meta.triple_analysis 回填到 sections、组合标题）。

import os, base64, json, logging, traceback, re
from typing import Dict, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI

RUNTIME_VERSION = "3.9.0-ai-led"
ANALYSIS_VERSION = os.getenv("ANALYSIS_VERSION", "390").strip()
SCHEMA_ID = "selfy.v3"
DEBUG = str(os.getenv("DEBUG","0")).strip() in ("1","true","True","YES","yes")

logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("selfy-ai")

app = FastAPI(title="Selfy AI - YiJing Analysis API", version=RUNTIME_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"] if DEBUG else ["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

try:
    client = OpenAI()
except Exception as e:
    logger.error("OpenAI client init failed: %s", e); client=None

BAGUA_SYMBOLS = {"艮":"山","离":"火","兑":"泽","乾":"天","坤":"地","震":"雷","巽":"风","坎":"水"}

# ----------------- 基础工具 -----------------

def _to_data_url(content: bytes, content_type: str) -> str:
    return f"data:{content_type};base64,{base64.b64encode(content).decode('utf-8')}"


def _build_tools_schema() -> List[Dict[str, Any]]:
    """定义唯一工具：submit_analysis_v3 —— 输出必须匹配前端结构。"""
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
    return (
        "只以 JSON object 返回（必须 JSON）。示例:{"
        "\"summary\":\"整体形象沉稳，兼具表达与洞察，适合稳中求进的节奏。\","
        "\"archetype\":\"稳中带锋\","
        "\"confidence\":0.92,"
        "\"sections\":{\"姿态\":\"姿态端正，气场稳重…\",\"神情\":\"神情自若，目光专注…\",\"面相\":\"面部轮廓分明，五官协调…\"},"
        "\"domains\":[\"金钱与事业\",\"配偶与感情\"],"
        "\"meta\":{"
            "\"triple_analysis\":{"
                "\"姿态\":{\"说明\":\"肩背挺直，重心稳定\",\"卦象\":\"艮\",\"解读\":\"止而有度，稳中求进\",\"性格倾向\":\"重视原则，行事有节制\"},"
                "\"神情\":{\"说明\":\"眼神专注，表情淡定\",\"卦象\":\"离\",\"解读\":\"洞察细致，表达清晰\",\"性格倾向\":\"逻辑性强\"},"
                "\"面容\":{\"说明\":\"下颌线条清晰，颧骨适中\",\"卦象\":\"兑\",\"解读\":\"沟通顺畅，情感外露\",\"性格倾向\":\"平易近人\"}"
            "},"
            "\"face_parts\":{"
                "\"眉\":{\"特征\":\"眉形平直\",\"卦象\":\"巽\",\"解读\":\"条理分明，判断力强\"},"
                "\"眼\":{\"特征\":\"双眼有神\",\"卦象\":\"离\",\"解读\":\"洞察力高，善于表达\"},"
                "\"鼻\":{\"特征\":\"鼻梁挺直\",\"卦象\":\"艮\",\"解读\":\"稳守本分，重边界\"},"
                "\"嘴\":{\"特征\":\"唇形对称\",\"卦象\":\"兑\",\"解读\":\"乐于沟通，情感真挚\"},"
                "\"颧/下巴\":{\"特征\":\"颧骨适中，下巴圆润\",\"卦象\":\"坤\",\"解读\":\"承载力强，内敛稳重\"}"
            "},"
            "\"overview\":\"结合姿态、神情与面容的卦象分析，此人外在稳健，内在兼具洞察与沟通能力，行事有分寸且能在需要时果断推进，适合在多变环境中保持稳定并创造机遇。\","
            "\"domains_detail\":{"
                "\"金钱与事业\":\"近期事业发展稳中有升，宜巩固现有成果，同时逐步开拓新领域，保持信息通畅以应对变化。\","
                "\"配偶与感情\":\"感情关系稳定且有深度交流，适合安排共同活动增加互动频率，避免因忙碌而忽视彼此感受。\""
            "},"
            "\"confidence_breakdown\":{"
                "\"图像清晰度\":0.85,"
                "\"卦象一致性\":0.9,"
                "\"特征显著性\":0.88"
            "}"
        "}"
        "}"
    )

# ----------------- Prompt（AI主导） -----------------

def _prompt_for_image_ai_led():
    sys = (
      "你是 Selfy AI 的易经观相助手。\n"
      "目标：由你(模型)产出全部分析与语言，我们(后端)只做结构约束。\n"
      "所有输出必须符合 Schema，并且包含以下必需字段，否则视为不合格：\n"
      "1) summary：20–50字综合性格印象。\n"
      "2) archetype：2–5字中文人格标签，禁止是卦名或卦象组合。\n"
      "3) sections：姿态、神情、面相，各有说明、卦象、解读、性格倾向。\n"
      "4) meta.face_parts：至少包含眉、眼、鼻、嘴、颧/下巴 5 项，每项含特征、卦象、解读，卦象需结合爻分析。\n"
      "5) meta.overview：90–150字，综合三象与五官分析成完整性格与风格说明。\n"
      "6) domains：['金钱与事业','配偶与感情']。\n"
      "7) meta.domains_detail：每项 60–90字，包含当前状态与可执行建议。\n"
      "8) confidence：0–1浮点。\n"
      "9) meta.confidence_breakdown：{图像清晰度, 卦象一致性, 特征显著性} 各0–1值，并附简短解释。\n"
      "10) meta.overview_card：标题=卦象组合，summary=简述，可与meta.overview一致。\n"
      "文风：融合易经术语与白话解释，避免模板化，禁止与图像无关的推测。\n"
      + _json_hint()
    )
    user = "请对输入人像进行AI主导的易经观相分析，并严格以JSON通过工具函数submit_analysis_v3返回，缺任何必需字段视为不合格输出。"
    return [{"role":"system","content":sys},{"role":"user","content":user}]

# ----------------- 轻度清洗 / 兜底 -----------------

_DOMAIN_LEADS = r"(在(金钱与事业|配偶与感情|事业|感情)(方面|中|里)?|目前|近期|当下)"
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
    s = re.sub(_DOMAIN_LEADS + r"[，、： ]*", "", s)
    s = re.sub(r"(可能|或许|也许)[，、 ]*", "", s)
    s = re.sub(r"[；;]+", "；", s)
    s = re.sub(r"[，,]{2,}", "，", s)
    return s.strip("；，。 ")


def _deep_clean(x):
    if isinstance(x, dict):
        return {k: _deep_clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_deep_clean(v) for v in x]
    if isinstance(x, str):
        return _neutralize(_depronoun(x))
    return x


def _ensure_sections(out: Dict[str,Any]) -> None:
    """若 sections 为空，则尝试从 meta.triple_analysis 回填。"""
    sec = out.get("sections") or {}
    meta = out.get("meta") or {}
    ta = (meta.get("triple_analysis") or {}) if isinstance(meta, dict) else {}
    if not sec or not all(k in sec and isinstance(sec[k], str) and sec[k].strip() for k in ("姿态","神情","面相")):
        out["sections"] = {
            "姿态": (ta.get("姿态") or {}).get("解读", ""),
            "神情": (ta.get("神情") or {}).get("解读", ""),
            "面相": (ta.get("面容") or {}).get("解读", ""),
        }


def _set_combo_title(out: Dict[str,Any]) -> None:
    """根据三象卦象生成组合标题，若 AI 已在 meta.overview/title 给出，则不覆盖。"""
    meta = out.setdefault("meta", {})
    ta = meta.get("triple_analysis") or {}
    hexes = [ (ta.get("姿态") or {}).get("卦象",""), (ta.get("神情") or {}).get("卦象",""), (ta.get("面容") or {}).get("卦象","") ]
    hexes = [h for h in hexes if h]
    if hexes:
        meta.setdefault("combo_title", " + ".join(hexes))
        meta.setdefault("overview_card", {"title": f"🔮 卦象组合：{' + '.join(hexes)}", "summary": (meta.get("overview") or "").strip()})


def _coerce_output(data: Dict[str,Any]) -> Dict[str,Any]:
    out = dict(data or {})
    out.setdefault("summary", "")
    out.setdefault("archetype", "")
    try:
        out["confidence"] = float(out.get("confidence", 0.0))
    except Exception:
        out["confidence"] = 0.0
    out.setdefault("sections", {"姿态":"","神情":"","面相":""})
    out.setdefault("domains", [])
    out.setdefault("meta", {})

    # 轻度清洗
    out = _deep_clean(out)

    # 兜底：从 triple_analysis 回填 sections、生成组合标题卡片
    _ensure_sections(out)
    _set_combo_title(out)

    return out


# ----------------- 路由 -----------------

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


# ----------------- OpenAI 调用 -----------------

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
    messages = _prompt_for_image_ai_led()
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


# ----------------- 上传接口 -----------------

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
