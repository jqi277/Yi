
# fastapi_app.py  (runtime v3.9.0-min, analysis logic v3.9.0)
# 改动要点（相对 v3.8.6-ux）
# - 大幅减少“预设解读/模板化加工”，把“分析与文字输出”交给 AI 模型；
# - 后端仅做：数据校验 + 六爻→上下卦/本卦/之卦 的“规则型计算”（不涉主观解读）；
# - 保持接口/外壳不变（submit_analysis_v3 工具 + 顶层 summary/sections/domains/meta 等）。
# - meta 下新增 yi 字段，统一承载易经计算结果：six_yao/trigrams/hexagram/derived_hexagram。
#
# 运行：uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload

import os, base64, json, logging, traceback, re
from typing import Dict, Any, List, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI

RUNTIME_VERSION = "3.9.0-min"
ANALYSIS_VERSION = os.getenv("ANALYSIS_VERSION", "390").strip()
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

# ====== YiJing core tables (rule-only, no interpretation) ======

# King Wen grid: key=(upper<<3)|lower (3-bit codes where 1=阳,0=阴; bits bottom->top)
KW_GRID = {
    # Upper: 乾 111
    (0b111<<3)|0b111:(1,"乾为天"), (0b111<<3)|0b011:(43,"泽天夬"), (0b111<<3)|0b101:(14,"火天大有"), (0b111<<3)|0b001:(34,"雷天大壮"),
    (0b111<<3)|0b110:(9,"风天小畜"), (0b111<<3)|0b010:(5,"水天需"), (0b111<<3)|0b100:(26,"山天大畜"), (0b111<<3)|0b000:(11,"地天泰"),
    # Upper: 兑 011
    (0b011<<3)|0b111:(10,"天泽履"), (0b011<<3)|0b011:(58,"兑为泽"), (0b011<<3)|0b101:(38,"火泽睽"), (0b011<<3)|0b001:(54,"雷泽归妹"),
    (0b011<<3)|0b110:(61,"风泽中孚"), (0b011<<3)|0b010:(60,"水泽节"), (0b011<<3)|0b100:(41,"山泽损"), (0b011<<3)|0b000:(19,"地泽临"),
    # Upper: 离 101
    (0b101<<3)|0b111:(13,"天火同人"), (0b101<<3)|0b011:(49,"泽火革"), (0b101<<3)|0b101:(30,"离为火"), (0b101<<3)|0b001:(55,"雷火丰"),
    (0b101<<3)|0b110:(37,"风火家人"), (0b101<<3)|0b010:(63,"水火既济"), (0b101<<3)|0b100:(22,"山火贲"), (0b101<<3)|0b000:(36,"地火明夷"),
    # Upper: 震 001
    (0b001<<3)|0b111:(25,"天雷无妄"), (0b001<<3)|0b011:(17,"泽雷随"), (0b001<<3)|0b101:(21,"火雷噬嗑"), (0b001<<3)|0b001:(51,"震为雷"),
    (0b001<<3)|0b110:(42,"风雷益"), (0b001<<3)|0b010:(3,"水雷屯"), (0b001<<3)|0b100:(27,"山雷颐"), (0b001<<3)|0b000:(24,"地雷复"),
    # Upper: 巽 110
    (0b110<<3)|0b111:(44,"天风姤"), (0b110<<3)|0b011:(28,"泽风大过"), (0b110<<3)|0b101:(50,"火风鼎"), (0b110<<3)|0b001:(32,"雷风恒"),
    (0b110<<3)|0b110:(57,"巽为风"), (0b110<<3)|0b010:(48,"水风井"), (0b110<<3)|0b100:(18,"山风蛊"), (0b110<<3)|0b000:(46,"地风升"),
    # Upper: 坎 010
    (0b010<<3)|0b111:(6,"天水讼"), (0b010<<3)|0b011:(47,"泽水困"), (0b010<<3)|0b101:(64,"火水未济"), (0b010<<3)|0b001:(40,"雷水解"),
    (0b010<<3)|0b110:(59,"风水涣"), (0b010<<3)|0b010:(29,"坎为水"), (0b010<<3)|0b100:(4,"山水蒙"), (0b010<<3)|0b000:(7,"地水师"),
    # Upper: 艮 100
    (0b100<<3)|0b111:(33,"天山遯"), (0b100<<3)|0b011:(31,"泽山咸"), (0b100<<3)|0b101:(56,"火山旅"), (0b100<<3)|0b001:(62,"雷山小过"),
    (0b100<<3)|0b110:(53,"风山渐"), (0b100<<3)|0b010:(39,"水山蹇"), (0b100<<3)|0b100:(52,"艮为山"), (0b100<<3)|0b000:(15,"地山谦"),
    # Upper: 坤 000
    (0b000<<3)|0b111:(12,"天地否"), (0b000<<3)|0b011:(45,"泽地萃"), (0b000<<3)|0b101:(35,"火地晋"), (0b000<<3)|0b001:(16,"雷地豫"),
    (0b000<<3)|0b110:(20,"风地观"), (0b000<<3)|0b010:(8,"水地比"), (0b000<<3)|0b100:(23,"山地剥"), (0b000<<3)|0b000:(2,"坤为地"),
}

TRI_MAP = {
    0b111:"乾", 0b011:"兑", 0b101:"离", 0b001:"震",
    0b110:"巽", 0b010:"坎", 0b100:"艮", 0b000:"坤"
}

FACIAL_ORDER = ["下巴","嘴","鼻","颧","眼","眉"]  # 初爻→上爻

def _tri_code(bits: List[int]) -> int:
    # bits: bottom->top 3 entries of {0,1}
    return (bits[0] << 0) | (bits[1] << 1) | (bits[2] << 2)

def _hex_bin(bits6: List[int]) -> int:
    val=0
    for i,b in enumerate(bits6):
        val |= (b&1) << i
    return val

def _king_wen(upper_code:int, lower_code:int) -> Tuple[int,str]:
    num, name = KW_GRID[(upper_code<<3)|lower_code]
    return num, name

def _derive_from_sixyao(lines: List[Dict[str,Any]]) -> Dict[str,Any]:
    """lines: list of 6 dicts with keys: feature, yin_yang('阴'|'阳'), moving(bool). position:1..6 bottom->top"""
    # normalize and order
    if not isinstance(lines, list) or len(lines)!=6:
        raise ValueError("six_yao must contain exactly 6 lines")
    # sort by position just in case
    lines_sorted = sorted(lines, key=lambda x: int(x.get("position",0)))
    bits = [1 if (ln.get("yin_yang")=="阳") else 0 for ln in lines_sorted]
    lower_bits = bits[0:3]
    upper_bits = bits[3:6]
    lower_code = _tri_code(lower_bits)
    upper_code = _tri_code(upper_bits)

    # derived by flipping moving
    moved_bits = []
    for ln in lines_sorted:
        b = 1 if (ln.get("yin_yang")=="阳") else 0
        if bool(ln.get("moving", False)):
            b = 1 - b
        moved_bits.append(b)
    lower2_code = _tri_code(moved_bits[0:3])
    upper2_code = _tri_code(moved_bits[3:6])

    kw_num, kw_name = _king_wen(upper_code, lower_code)
    kw2_num, kw2_name = _king_wen(upper2_code, lower2_code)

    return {
        "six_yao_bits": bits,
        "trigrams": {
            "lower": {"code": lower_code, "name": TRI_MAP[lower_code]},
            "upper": {"code": upper_code, "name": TRI_MAP[upper_code]},
        },
        "hexagram": {
            "king_wen_no": kw_num,
            "name": kw_name,
            "binary": _hex_bin(bits),
            "moving_lines": [int(ln.get("position",0)) for ln in lines_sorted if bool(ln.get("moving", False))]
        },
        "derived_hexagram": {
            "king_wen_no": kw2_num,
            "name": kw2_name,
            "binary": _hex_bin(moved_bits)
        }
    }

# ====== OpenAI tool schema (unchanged function name) ======
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

# ====== Prompt (shift analysis to AI; backend only computes YiJing math) ======
def _json_hint() -> str:
    return ("只以 JSON object 返回（必须 JSON）。顶层键包含 summary/archetype/confidence/sections/domains/meta。"
            "meta 中请包含：signals(可选)、face_parts、six_yao (按【下巴,嘴,鼻,颧,眼,眉】映射初至上爻)、"
            "domains_detail（含『金钱与事业』『配偶与感情』两段 60–120 字描述与建议）。")

def _prompt_minimal_v390():
    sys = (
      "你是 Selfy AI 的易经观相助手，负责‘分析与撰写’，而非模板复读。"
      "现在请：\n"
      "A) 观察图片并用自然语言完成解读：summary（20–40字），sections.姿态/神情/面相（各 1–3 句，无口号）。"
      "B) 产出人格标签 archetype（2–5字中文，避免热词）。\n"
      "C) 生成六爻：将【下巴,嘴,鼻,颧,眼,眉】依次映射为 1..6 爻（初→上），"
      "   对每一爻给出：feature, position(1..6), score(0..100), yin_yang('阳'或'阴'), moving(true/false)。"
      "   判定建议（可被后端验证，不必一致）：score>=66 判为阳，<=34 判为阴；其余区间按观感决定；"
      "   moving 的参考：极强或极弱（如>=88 或<=12）可判为动爻。\n"
      "D) 在 meta.domains_detail 中分别写『金钱与事业』『配偶与感情』两段，状态+建议一体化、可落地，避免自我重复。\n"
      "E) 可在 meta.signals 给出 Pose/Expression/Face 等简要标签（可选）。\n"
      "F) 严禁输出固定模板、禁止口水话；语言简洁、具体、可执行。\n"
      "仅通过 submit_analysis_v3 工具返回 JSON。"
    )
    user = "请按上述要求分析图片，所有结果通过函数返回。"+_json_hint()
    return [{"role":"system","content":sys},{"role":"user","content":user}]

# ====== helpers ======
def _to_data_url(content: bytes, content_type: str) -> str:
    return f"data:{content_type};base64,{base64.b64encode(content).decode('utf-8')}"

def _call_openai(messages):
    if client is None:
        raise RuntimeError("OpenAI client not initialized")
    return client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL","gpt-4o"),
        temperature=0.6,
        tools=_build_tools_schema(),
        tool_choice={"type":"function","function":{"name":"submit_analysis_v3"}},
        response_format={"type":"json_object"},
        messages=messages
    )

def _extract_tool_args(resp) -> Dict[str,Any]:
    choice = resp.choices[0]
    tc = getattr(choice.message, "tool_calls", None)
    if tc:
        return json.loads(tc[0].function.arguments)
    content = getattr(choice.message, "content", None)
    if isinstance(content, str) and content.strip().startswith("{"):
        return json.loads(content)
    raise RuntimeError("Model did not return tool_calls.")

def _minimal_clean(obj: Any) -> Any:
    # 不做风格化改写，只做结构/类型/空格的微清洗
    if isinstance(obj, dict):
        return {k: _minimal_clean(v) for k,v in obj.items()}
    if isinstance(obj, list):
        return [_minimal_clean(v) for v in obj]
    if isinstance(obj, str):
        s = obj.strip()
        s = re.sub(r"\s+", " ", s)
        return s
    return obj

def _compute_yi_append(meta: Dict[str,Any]) -> Dict[str,Any]:
    # 从 meta.six_yao 计算 Yi 结果，附加到 meta.yi
    lines = (meta or {}).get("six_yao") or []
    try:
        yi = _derive_from_sixyao(lines)
    except Exception as e:
        yi = {"error": f"yi_compute_failed: {e.__class__.__name__}: {e}"}
    meta = dict(meta or {})
    meta["yi"] = yi
    return meta

# ====== endpoints ======
@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/", include_in_schema=False)
def root():
    return HTMLResponse("<h3>Selfy AI</h3><a href='/docs'>/docs</a> · <a href='/version'>/version</a>")

@app.head("/", include_in_schema=False)
def root_head():
    return Response(status_code=200)

@app.get("/version")

@app.get("/mobile", include_in_schema=False)
def mobile():
    import os
    from fastapi.responses import HTMLResponse
    path = os.path.join(os.path.dirname(__file__), "index_mobile.html")
    try:
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
    except Exception:
        html = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Selfy AI Mobile</title>
<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;padding:16px;line-height:1.5}code{background:#f5f5f5;padding:2px 4px;border-radius:4px}</style>
</head><body>
<h2>Selfy AI Mobile</h2>
<p>这是默认的移动端占位页。你可以在项目根目录放置 <code>index_mobile.html</code> 来覆盖本页。</p>
<p>API:</p>
<ul>
  <li><code>GET /health</code></li>
  <li><code>GET /version</code></li>
  <li><code>POST /upload</code> (multipart/form-data, <code>file</code>=image)</li>
</ul>
</body></html>"""
    return HTMLResponse(html)

def version(): return {"runtime":RUNTIME_VERSION,"analysis":ANALYSIS_VERSION,"schema":SCHEMA_ID,"debug":DEBUG}

def _call_gpt_tool_with_image(data_url: str) -> Dict[str,Any]:
    messages = _prompt_minimal_v390()
    messages[-1]["content"] = [
        {"type":"text","text":messages[-1]["content"]},
        {"type":"image_url","image_url":{"url":data_url}}
    ]
    resp = _call_openai(messages)
    args = _extract_tool_args(resp)
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
        tool_args = _minimal_clean(result["tool_args"])

        # 仅计算 Yi（规则），并附加到 meta.yi；不改写任何文本
        meta = tool_args.get("meta") or {}
        meta = _compute_yi_append(meta)
        tool_args["meta"] = meta

        # 顶层壳体保持原状
        final_out = tool_args

        if DEBUG:
            meta_dbg = final_out.setdefault("meta",{}).setdefault("debug",{})
            meta_dbg["file_info"]={"filename":file.filename,"content_type":ct,"size":len(raw)}
            try:
                meta_dbg["oai_choice_finish_reason"]=result["oai_raw"].choices[0].finish_reason
            except Exception:
                meta_dbg["oai_choice_finish_reason"]="n/a"

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
