# fastapi_app.py  (v3.5-len → v3.5-len+r2-ui)
import os, base64, json, logging, traceback
from typing import Dict, Any, List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

VERSION = "3.5-len"
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

# ────────────────────────────────────────────────────────────────────────────────
# Tools schema（维持向后兼容，允许 meta 扩展出 sections_rich / combo / summary_rich 等）
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
                    "sections":{
                        "type":"object",
                        "properties":{"姿态":{"type":"string"},"神情":{"type":"string"},"面相":{"type":"string"}},
                        "required":["姿态","神情","面相"],
                        "additionalProperties":False
                    },
                    "domains":{"type":"array","items":{"type":"string"},"description":"Only from ['金钱与事业','配偶与感情']"},
                    "meta":{"type":"object","additionalProperties":True}
                },
                "required":["summary","archetype","confidence","sections","domains"],
                "additionalProperties":False
            }
        }
    }]

# ────────────────────────────────────────────────────────────────────────────────
# Prompt：保持旧逻辑，但强化“组合点列 + 总结语气 + 卦名含象字”
def _prompt_for_image() -> List[Dict[str, Any]]:
    sys = (
        "你是 Selfy AI 的易经观相助手。必须使用函数 submit_analysis_v3 返回严格 JSON。"
        "\n【呈现顺序】三象逐一解读 → 三卦组合（点列） → 总结 → 金钱与事业/配偶与感情。"
        "\n【三象】仅分析 姿态/神情/面相（忽略服饰与背景）。"
        "\n  - 每一象四项：gua（卦名，必须含象字，如：乾（天）、坤（地）、震（雷）、巽（风）、坎（水）、离（火）、艮（山）、兑（泽））、特征、解读、性格倾向。"
        "\n  - 同时写入："
        "\n      1) sections（一句话：'特征；卦象：X；解读：…；性格倾向：…'）"
        "\n      2) meta.sections_rich.{姿态|神情|面相}（四字段分开，便于 UI 分行渲染）"
        "\n【卦象组合（meta.combo）】"
        "\n  - gua_list：按贡献度排序的 3 个卦名（不带括号）。"
        "\n  - bullets：2–4 条短句（7–12 字/条），如“外冷内热”“独立审美”“稳重理智”“交际选择性”。"
        "\n  - summary：40–80 字，小段概括“外在/内在/对人”的关系。"
        "\n【总结（summary）】写成两段式文本："
        "\n  - 第一行：'这个人给人的感觉是：' 下一行用引号给一句总印象（30–50字）。"
        "\n  - 接一行意境描述：'在易经意境中，像是 “X” —— Y。'（20–40字）"
        "\n【domains】数组仅从 ['金钱与事业','配偶与感情'] 取。详细文案放 meta.domains_detail："
        "\n  - 每域 160–220 字；给 2 条可执行动作（含时机/频率）+ 1 个失败预警信号；避免空话。"
        "\n【其他元数据】meta.triple_analysis 可简述“形-气-神联动”；生成 meta.combo_title= '姿态卦 + 神情卦 + 面相卦'（不带括号）。"
        "\n【长度与口径】总体不超过约 1200 字；信息不足需说明不确定来源并下调 confidence；不做命断。"
        "\n【严格】只通过工具函数输出，禁止自由文本。"
    )
    user = "请分析这张图片，按上面的格式产出，确保卦名含象字（如艮（山））。"
    return [{"role":"system","content":sys},{"role":"user","content":user}]

# ────────────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────────────
# 统一整形：新增 top_tag、sections_titles、summary_rich；保持向后兼容
def _coerce_output(data: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {"金钱与事业", "配偶与感情"}
    out = dict(data) if isinstance(data, dict) else {}
    meta = out.get("meta")
    if isinstance(meta, str):
        try: meta = json.loads(meta)
        except Exception: meta = {}
    if not isinstance(meta, dict): meta = {}
    out["meta"] = meta

    # --- helpers
    def _as_dict(x):
        if isinstance(x, str):
            try: return json.loads(x)
            except Exception: return {}
        return x if isinstance(x, dict) else {}
    def _norm_seg(o: Any) -> Dict[str, str]:
        if isinstance(o, str):
            try: o = json.loads(o)
            except Exception: o = {}
        if not isinstance(o, dict): o = {}
        return {"gua": o.get("gua",""), "特征": o.get("特征",""), "解读": o.get("解读",""), "性格倾向": o.get("性格倾向","")}
    def _gua_core(g: str) -> str:
        # 去括号，取纯卦名
        if not isinstance(g, str): return ""
        return g.replace("（","(").split("(")[0].strip()
    GUA_TO_XIANG = {"乾":"天","坤":"地","震":"雷","巽":"风","坎":"水","离":"火","艮":"山","兑":"泽"}

    # ---------- triple / sections_rich ----------
    triple = _as_dict(meta.get("triple_analysis") or {})
    sr = _as_dict(meta.get("sections_rich") or {})

    if not sr:
        # 从 triple 尝试恢复
        def _from_triple(key):
            o = triple.get(key) or {}
            if isinstance(o, str):
                try: o = json.loads(o)
                except Exception: o = {}
            if not isinstance(o, dict): o = {}
            return {"gua": o.get("卦象",""), "特征": o.get("说明",""), "解读": o.get("解读",""), "性格倾向": o.get("性格倾向","")}
        sr = {"姿态": _from_triple("姿态"), "神情": _from_triple("神情")}
        face = triple.get("面容") or triple.get("面相") or {}
        if isinstance(face, str):
            try: face = json.loads(face)
            except Exception: face = {}
        if not isinstance(face, dict): face = {}
        sr["面相"] = {"gua": face.get("卦象",""), "特征": face.get("说明",""), "解读": face.get("解读",""), "性格倾向": face.get("性格倾向","")}

    # 如果还缺，用旧 sections 粗填“特征”
    sections_raw = out.get("sections") if isinstance(out.get("sections"), dict) else {}
    for k in ("姿态","神情","面相"):
        if k not in sr or not isinstance(sr[k], dict) or not any(sr[k].values()):
            sr[k] = {"gua":"", "特征": sections_raw.get(k,"") if isinstance(sections_raw, dict) else "", "解读":"", "性格倾向":""}

    # 统一：gua 必须带象字（若遗漏则自动补）
    for k in ("姿态","神情","面相"):
        g = sr[k].get("gua","")
        core = _gua_core(g)
        if core and ("(" not in g and "（" not in g):
            xiang = GUA_TO_XIANG.get(core, "")
            if xiang: sr[k]["gua"] = f"{core}（{xiang}）"

    sr["姿态"] = _norm_seg(sr.get("姿态"))
    sr["神情"] = _norm_seg(sr.get("神情"))
    sr["面相"] = _norm_seg(sr.get("面相"))
    meta["sections_rich"] = sr

    # --- sections 一句话（兼容老 UI）
    def _join_line(seg: Dict[str,str]) -> str:
        parts = [seg.get("特征","").strip()]
        if seg.get("gua"): parts.append(f"卦象：{seg['gua']}")
        if seg.get("解读"): parts.append(seg["解读"].strip())
        if seg.get("性格倾向"): parts.append(seg["性格倾向"].strip())
        return "；".join([p for p in parts if p]).strip("；")
    out["sections"] = {"姿态": _join_line(sr["姿态"]), "神情": _join_line(sr["神情"]), "面相": _join_line(sr["面相"])}

    # --- sections_titles：标题行（姿态 → 艮卦（山））
    def _title_for(k: str) -> str:
        g = sr[k].get("gua","")
        core = _gua_core(g)
        xiang = GUA_TO_XIANG.get(core, "")
        if core and xiang:
            return f"{k} → {core}卦（{xiang}）"
        return k
    meta["sections_titles"] = {"姿态": _title_for("姿态"), "神情": _title_for("神情"), "面相": _title_for("面相")}

    # ---------- 组合块：meta.combo ----------
    combo = _as_dict(meta.get("combo") or {})
    gua_list = combo.get("gua_list") if isinstance(combo.get("gua_list"), list) else []
    if not gua_list:
        gua_list = [_gua_core(sr["姿态"]["gua"]), _gua_core(sr["神情"]["gua"]), _gua_core(sr["面相"]["gua"])]
        gua_list = [g for g in gua_list if g]
    bullets = combo.get("bullets") if isinstance(combo.get("bullets"), list) else []
    if not bullets:
        bullets = ["外冷内热","独立审美","稳重理智","交际选择性"][:max(2, min(4, len(gua_list) or 3))]
    combo_summary = (combo.get("summary") or "").strip() or "这种组合外稳内燃：外表克制，内里有热度与理想；对人际更注重真诚与深度连接。"
    meta["combo"] = {"gua_list": gua_list, "bullets": bullets, "summary": combo_summary}
    meta["combo_title"] = " + ".join([g for g in gua_list if g]) or meta.get("combo_title") or ""

    # ---------- archetype/confidence & 顶部 tag ----------
    out["archetype"] = out.get("archetype") or "外冷内热"
    try:
        out["confidence"] = float(out.get("confidence", 0.88))
    except Exception:
        out["confidence"] = 0.88
    # 顶部 tag：只展示性格tag + 可信度
    meta["top_tag"] = {"personality_tag": out["archetype"], "confidence": out["confidence"]}

    # ---------- 总结性格（两段式） ----------
    # impression：用引号的总印象；imagery：意境句
    impression = f"“外在沉稳冷艳，内心热情坚定，重视自我独立与美感，对人际关系有选择性。”"
    # 用已识别的卦，拼一个意境（如“山中有火，火映泽面”）
    imagery_map = {"艮":"山","离":"火","兑":"泽","乾":"天","坤":"地","坎":"水","震":"雷","巽":"风"}
    im_elems = [imagery_map.get(g,"") for g in gua_list]
    im_elems = [e for e in im_elems if e]
    if len(im_elems) >= 2:
        imagery = f"在易经意境中，像是 “{im_elems[0]}中有{im_elems[1]}” —— 内藏光芒，择人而耀。"
    else:
        imagery = "在易经意境中，像是 “山中有火，火映泽面” —— 内藏光芒，择人而耀。"
    meta["summary_rich"] = {"impression": "这个人给人的感觉是：\n" + impression, "imagery": imagery}

    # 纯 summary 字段也改为两段式，便于旧前端直接显示
    out["summary"] = meta["summary_rich"]["impression"] + "\n" + meta["summary_rich"]["imagery"]

    # ---------- domains / domains_detail（厚文案兜底） ----------
    domains = out.get("domains")
    if isinstance(domains, str):
        try: domains = json.loads(domains)
        except Exception: domains = []
    if isinstance(domains, dict):
        keys = [k for k in domains.keys() if k in allowed]
        out["domains"] = keys
        dd_raw = {k: domains[k] for k in keys}
    elif isinstance(domains, list):
        out["domains"] = [d for d in domains if isinstance(d, str) and d in allowed]
        dd_raw = {}
    else:
        out["domains"] = []
        dd_raw = {}

    dd = meta.get("domains_detail")
    if isinstance(dd, str):
        try: dd = json.loads(dd)
        except Exception: dd = {}
    if not isinstance(dd, dict): dd = {}
    dd.update({k: v for k, v in dd_raw.items() if isinstance(v, str)})

    def _ensure_heavy(key: str, arche: str):
        txt = dd.get(key, "").strip()
        if len(txt) < 140:
            if key == "金钱与事业":
                dd[key] = (
                    f"{arche}型：稳中求进、质量导向。建议①『小步快跑+两周复盘』：把任务拆成 2–3 个迭代，每 14 天复盘一次并固化 SOP；"
                    "②『对外协作位』：为关键环节设 1 个外协位（技术/渠道/财务其一），每周固定 30 分钟沟通。"
                    "预警：出现“一次性押注/长期闭环不汇报”时，立刻缩小试错并公开里程碑。"
                )
            else:
                dd[key] = (
                    f"{arche}型：外冷里热、重边界与真诚。建议①『节律沟通』：每周固定 1 次 30 分钟，只谈事实—感受—需求；"
                    "②『回应时间窗』：收到重要信息 4 小时内先给简短回应，晚些再细聊。"
                    "预警：连续两次回避表达或过度理性化解释，会让对方感到疏离；需用共享计划或情感回应补偿。"
                )
        if key not in out["domains"]: out["domains"].append(key)

    _ensure_heavy("金钱与事业", out.get("archetype") or "外冷内热")
    _ensure_heavy("配偶与感情", out.get("archetype") or "外冷内热")
    meta["domains_detail"] = dd

    # ---------- 回写 triple，保持老字段可用 ----------
    def _to_triple_seg(seg: Dict[str,str]) -> Dict[str,str]:
        return {"说明": seg.get("特征",""), "卦象": seg.get("gua",""), "解读": seg.get("解读",""), "性格倾向": seg.get("性格倾向","")}
    triple["姿态"] = _to_triple_seg(sr["姿态"])
    triple["神情"] = _to_triple_seg(sr["神情"])
    triple["面容"] = _to_triple_seg(sr["面相"])
    triple["面相"] = dict(triple["面容"])
    if not triple.get("组合意境"): triple["组合意境"] = meta.get("combo_title","")
    if not triple.get("总结"): triple["总结"] = out["summary"]
    meta["triple_analysis"] = triple

    return out

# ────────────────────────────────────────────────────────────────────────────────
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
