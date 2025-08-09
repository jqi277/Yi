# fastapi_app.py  (v3.5-len.1)
import os, base64, json, logging, traceback, re
from typing import Dict, Any, List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

VERSION = "3.5-len.1"
SCHEMA_ID = "selfy.v3"
DEBUG = str(os.getenv("DEBUG", "0")).strip() in ("1","true","True","YES","yes")

logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("selfy-ai")

app = FastAPI(title="Selfy AI - YiJing Analysis API", version=VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

try:
    client = OpenAI()
except Exception as e:
    logger.error("OpenAI client init failed: %s", e); client = None

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/", include_in_schema=False)
def root():
    return HTMLResponse('''<h3>Selfy AI - YiJing Analysis API</h3>
    <ul><li><a href="/docs">/docs (Swagger)</a></li><li><a href="/health">/health</a></li><li><a href="/version">/version</a></li></ul>''')

@app.head("/", include_in_schema=False)
def root_head(): return Response(status_code=200)

@app.get("/version")
def version(): return {"version": VERSION, "debug": DEBUG, "schema": SCHEMA_ID}

def _to_data_url(content: bytes, content_type: str) -> str:
    return f"data:{content_type};base64," + base64.b64encode(content).decode("utf-8")

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
                    "domains":{"type":"array","items":{"type":"string"},"description":"Only from ['金钱与事业','配偶与感情']"},
                    "meta":{"type":"object","additionalProperties":True}
                },
                "required":["summary","archetype","confidence","sections","domains"],
                "additionalProperties":False
            }
        }
    }]

def _prompt_for_image() -> List[Dict[str, Any]]:
    sys = (
        "你是 Selfy AI 的易经观相助手。必须先用“三象四段式”分析，并通过工具函数 submit_analysis_v3 返回结果。"
        "\n\n【三象固定】仅使用：姿态、神情、面容（不要写环境）。每一象包含以下四点："
        "1) 说明：1句，60–80字；"
        "2) 卦象：仅1个卦名（如 艮/离/兑/乾/坤/坎/震/巽）；"
        "3) 解读：1–2句，60–80字；"
        "4) 性格倾向：1–2句，60–80字（把“特征”合并成倾向，避免重复词）。"
        "\n\n【卦象参考锚点】（偏好而非唯一）："
        "\n- 姿态：挺拔/克制/边界→艮；昂首自信/领导感→乾；放松亲和→兑；温厚承载→坤。"
        "\n- 神情：目光直接/明亮→离；沉静谨慎→坎；果断开启→震；圆融入微→巽。"
        "\n- 面容：立体外放→离/乾；柔和亲和→坤/兑；端正持重→艮/坤。"
        "\n\n【卦象组合】标题=三象卦名按“姿态 + 神情 + 面容”相加（例：艮 + 离 + 兑）。正文 90–150字，覆盖整体气质、内外关系、行为风格、人际取向。"
        "\n\n【总结性格印象】20–40字。"
        "\n\n【领域建议】domains 仅从 ['金钱与事业','配偶与感情'] 选择；在 meta.domains_detail 中为每个选中的领域写 60–90字建议（包含优势+建议）。"
        "\n\n【输出契约】仅用 submit_analysis_v3：summary=总结；archetype=4–6字意境词；sections(三象各1句，用“说明；卦象：X；解读：…；性格倾向：…”拼接)；"
        "domains(数组)；meta.triple_analysis(含'姿态','神情','面容','组合意境','总结'，且每象含'说明/卦象/解读/性格倾向')；meta.domains_detail。"
        "语言：中文。禁止自由文本输出。"
    )
    user = "请分析这张图片，专注姿态/神情/面容三象，忽略服饰与背景。严格使用工具函数返回 JSON，并包含 meta.triple_analysis / meta.domains_detail。"
    return [{"role":"system","content":sys},{"role":"user","content":user}]

def _call_gpt_tool_with_image(data_url: str) -> Dict[str, Any]:
    if client is None: raise RuntimeError("OpenAI client is not initialized. Check OPENAI_API_KEY.")
    messages = _prompt_for_image()
    messages[-1]["content"] = [{"type":"text","text":messages[-1]["content"]},{"type":"image_url","image_url":{"url":data_url}}]
    resp = client.chat.completions.create(
        model="gpt-4o", temperature=0.3, tools=_build_tools_schema(),
        tool_choice={"type":"function","function":{"name":"submit_analysis_v3"}},
        response_format={"type":"json_object"}, messages=messages,
    )
    if DEBUG:
        try: logger.debug("[OAI] raw response: %s", resp)
        except Exception: pass
    choice = resp.choices[0]
    tool_calls = getattr(choice.message, "tool_calls", None)
    if not tool_calls:
        content = getattr(choice.message, "content", None)
        if isinstance(content, str) and content.strip().startswith("{"):
            try: return {"tool_args": json.loads(content), "oai_raw": resp if DEBUG else None}
            except Exception: pass
        raise RuntimeError("Model did not return tool_calls. Inspect raw response in DEBUG logs.")
    tool = tool_calls[0]
    if tool.function.name != "submit_analysis_v3": raise RuntimeError(f"Unexpected tool called: {tool.function.name}")
    try: args = json.loads(tool.function.arguments)
    except json.JSONDecodeError as e: raise RuntimeError(f"Tool arguments JSON decode failed: {e}")
    return {"tool_args": args, "oai_raw": resp if DEBUG else None}

# --------------------------- robust normalizer ---------------------------

_DEF_SUMMARY = "外在沉稳，内里有光，选择性社交。"

def _parse_seg_from_line(line: str) -> Dict[str, str]:
    """
    反向从“一句合成文本”里解析三象四段：
    形如：说明；卦象：X；解读：...；性格倾向：...
    """
    if not isinstance(line, str): line = ""
    line = line.strip("；").strip()
    seg = {"说明":"", "卦象":"", "解读":"", "性格倾向":""}
    if not line: return seg
    # 卦象
    m = re.search(r"卦象[:：]\s*([^\s；。,\u3002]+)", line)
    if m: seg["卦象"] = m.group(1).strip()
    # 解读
    m = re.search(r"解读[:：]\s*(.+?)(?:；|$)", line)
    if m: seg["解读"] = m.group(1).strip()
    # 性格倾向
    m = re.search(r"(性格倾向|倾向)[:：]\s*(.+?)(?:；|$)", line)
    if m: seg["性格倾向"] = m.group(2).strip()
    # 说明 = 去掉已知字段之前的部分
    parts = re.split(r"[；;]", line)
    if parts: seg["说明"] = parts[0].strip()
    return seg

def _parse_triple_from_free_text(txt: str) -> Dict[str, Any]:
    """
    模型偶尔把 meta.triple_analysis 写成一整段字符串。
    这里尝试从中抽取三象 + 组合意境 + 总结。
    """
    triple = {"姿态":{}, "神情":{}, "面容":{}, "组合意境":"", "总结":""}
    if not isinstance(txt, str): return triple
    # 尝试切块：按“姿态/神情/面容/组合意境/总结”关键词
    # 也兼容不带小标题、只是顺序拼接的情况（取三次“卦象：”分段）
    # 1) 直接命名块
    for key in ["姿态","神情","面容"]:
        m = re.search(rf"{key}.*?(卦象[:：].+?)(?=(姿态|神情|面容|组合意境|总结|$))", txt, flags=re.S)
        if m:
            triple[key] = _parse_seg_from_line(m.group(1))
    # 2) 组合意境 / 总结
    m = re.search(r"(组合意境)[:：]\s*(.+?)(?=(姿态|神情|面容|总结|$))", txt, flags=re.S)
    if m: triple["组合意境"] = m.group(2).strip()
    m = re.search(r"(总结)[:：]\s*(.+)$", txt, flags=re.S)
    if m: triple["总结"] = m.group(2).strip()
    return triple

def _coerce_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    目标：
    - 任何字段即便是 str（甚至是 JSON 字符串）也不崩；
    - sections.* 始终输出一句话；
    - domains 支持对象/数组；对象转数组，长文进 meta.domains_detail；
    - 永远构造出 meta.triple_analysis 的标准结构；补 meta.combo_title。
    """
    allowed = {"金钱与事业", "配偶与感情"}

    out = dict(data) if isinstance(data, dict) else {}
    meta = out.get("meta")
    if isinstance(meta, str):
        try: meta = json.loads(meta)
        except Exception: meta = {}
    if not isinstance(meta, dict): meta = {}
    out["meta"] = meta

    # ---------- 读原 triple（可能是 dict 或 字符串） ----------
    raw_ta = meta.get("triple_analysis")
    triple: Dict[str, Any]
    if isinstance(raw_ta, dict):
        triple = raw_ta
    elif isinstance(raw_ta, str):
        # 先尝试当 JSON
        t = None
        try:
            t = json.loads(raw_ta)
            if not isinstance(t, dict): t = None
        except Exception:
            t = None
        triple = t if t is not None else _parse_triple_from_free_text(raw_ta)
    else:
        triple = {}

    # ---------- sections 读取 ----------
    sections_raw = out.get("sections")
    if isinstance(sections_raw, str):
        try:
            tmp = json.loads(sections_raw); sections = tmp if isinstance(tmp, dict) else {}
        except Exception:
            sections = {}
    elif isinstance(sections_raw, dict):
        sections = sections_raw
    else:
        sections = {}

    # 如果 triple 中三象为空，尝试从 sections 反向解析
    def _ensure_seg(o): return {"说明":o.get("说明",""),"卦象":o.get("卦象",""),"解读":o.get("解读",""),"性格倾向":o.get("性格倾向","")}
    def _from_sections(name_cn, fallback_key):
        line = sections.get(fallback_key) if isinstance(sections.get(fallback_key), str) else ""
        return _ensure_seg(_parse_seg_from_line(line))

    t姿 = triple.get("姿态") if isinstance(triple.get("姿态"), dict) else {}
    t神 = triple.get("神情") if isinstance(triple.get("神情"), dict) else {}
    t面 = (triple.get("面容") if isinstance(triple.get("面容"), dict) else (triple.get("面相") if isinstance(triple.get("面相"), dict) else {}))

    if not any(t姿.values()): t姿 = _from_sections("姿态", "姿态")
    if not any(t神.values()): t神 = _from_sections("神情", "神情")
    if not any(t面.values()): t面 = _from_sections("面容", "面相")

    triple = {
        "姿态": _ensure_seg(t姿),
        "神情": _ensure_seg(t神),
        "面容": _ensure_seg(t面),
    }

    # 组合意境 / 总结
    combo = meta.get("combo_title") or meta.get("组合意境") or ""
    if not combo:
        hx = [triple["姿态"]["卦象"], triple["神情"]["卦象"], triple["面容"]["卦象"]]
        combo = " + ".join([h for h in hx if h])
    summary = out.get("summary") or ""
    if not summary:
        # 尝试从原始字符串里面抓“总结：”
        raw_ta_str = meta.get("triple_analysis")
        if isinstance(raw_ta_str, str):
            m = re.search(r"(总结)[:：]\s*(.+)$", raw_ta_str, flags=re.S)
            if m: summary = m.group(2).strip()
    if not summary: summary = _DEF_SUMMARY

    meta["triple_analysis"] = {
        "姿态": triple["姿态"],
        "神情": triple["神情"],
        "面容": triple["面容"],
        "组合意境": combo if combo else "整体气质克制而有火光，外在稳、内在明，处事理性中带热度。",
        "总结": summary
    }

    # ---------- 输出三象一句话到 sections ----------
    def _mk_line(seg):
        desc, hexg, mean, tend = seg["说明"], seg["卦象"], seg["解读"], seg["性格倾向"]
        parts = [p for p in [desc, f"卦象：{hexg}" if hexg else "", mean, tend] if p]
        return "；".join(parts).strip("；")

    out["sections"] = {
        "姿态": _mk_line(meta["triple_analysis"]["姿态"]),
        "神情": _mk_line(meta["triple_analysis"]["神情"]),
        "面相": _mk_line(meta["triple_analysis"]["面容"]),
    }

    # ---------- domains ----------
    domains = out.get("domains")
    if isinstance(domains, str):
        try: domains = json.loads(domains)
        except Exception: domains = []
    if isinstance(domains, dict):
        keys = [k for k in domains.keys() if k in allowed]
        out["domains"] = keys
        meta["domains_detail"] = {k: domains[k] for k in keys}
    elif isinstance(domains, list):
        out["domains"] = [d for d in domains if isinstance(d, str) and d in allowed]
    else:
        out["domains"] = []

    dd = meta.get("domains_detail")
    if isinstance(dd, str):
        try: dd = json.loads(dd)
        except Exception: dd = {}
    if not isinstance(dd, dict): dd = {}
    # 兜底建议
    def _ensure_advice(key, fallback):
        if key not in dd or not isinstance(dd.get(key), str) or not dd.get(key).strip():
            dd[key] = fallback
            if key not in out["domains"]: out["domains"].append(key)

    arche = out.get("archetype") or "外冷内热"
    _ensure_advice("金钱与事业", f"{arche}：擅长独立推进与质量把控，短期重稳健现金流；建议设立节点复盘与对外协作位，避免闭环过严错失窗口。")
    _ensure_advice("配偶与感情", f"{arche}：表冷里热，重边界与真诚；建议放缓观察周期，适度表达需求，匹配价值观与生活节奏。")
    meta["domains_detail"] = dd

    # ---------- 其它必填 ----------
    out["summary"] = summary
    out["archetype"] = out.get("archetype") or "自信圆融"
    try: out["confidence"] = float(out.get("confidence", 0.0))
    except Exception: out["confidence"] = 0.0

    # combo_title 给前端用
    meta["combo_title"] = " + ".join([x for x in [triple["姿态"]["卦象"], triple["神情"]["卦象"], triple["面容"]["卦象"]] if x])

    return out

# --------------------------- /upload ---------------------------

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        if not file: raise HTTPException(status_code=400, detail="No file uploaded.")
        ct = file.content_type or ""
        if not ct.startswith("image/"): raise HTTPException(status_code=415, detail=f"Unsupported content type: {ct}")
        raw = await file.read()
        if not raw: raise HTTPException(status_code=400, detail="Empty file.")
        if len(raw) > 15*1024*1024: raise HTTPException(status_code=413, detail="File too large (>15MB).")
        data_url = _to_data_url(raw, ct); logger.info("[UPLOAD] file=%s size=%d type=%s", file.filename, len(raw), ct)
        result = _call_gpt_tool_with_image(data_url); tool_args = result["tool_args"]; final_out = _coerce_output(tool_args)
        if DEBUG:
            meta = final_out.setdefault("meta", {}); dbg = meta.setdefault("debug", {}); dbg["debug_mode"]=True
            dbg["file_info"]={"filename":file.filename,"content_type":ct,"size":len(raw)}
            if result.get("oai_raw") is not None:
                try: dbg["oai_choice_finish_reason"]=result["oai_raw"].choices[0].finish_reason; dbg["oai_has_tool_calls"]=bool(result["oai_raw"].choices[0].message.tool_calls)
                except Exception: dbg["oai_choice_finish_reason"]="n/a"; dbg["oai_has_tool_calls"]="n/a"
        return JSONResponse(content=final_out, status_code=200)
    except HTTPException as he:
        if DEBUG: return JSONResponse(status_code=he.status_code, content={"error": he.detail, "debug": {"trace": traceback.format_exc()}})
        raise
    except Exception as e:
        logger.exception("[ERROR] /upload failed: %s", e)
        body={"error":"Internal Server Error"}
        if DEBUG: body["debug"]={"message":str(e),"trace":traceback.format_exc()}
        return JSONResponse(status_code=500, content=body)
