# fastapi_app.py  (v3.5-len → v3.5-len+r1)
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
# Tools schema（维持向后兼容，允许 meta 扩展出 sections_rich / combo）
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
# Prompt：先“三象逐一解读（含卦）”→ “三卦组合” → “总结” → “两域建议（更厚）”
def _prompt_for_image() -> List[Dict[str, Any]]:
    sys = (
        "你是 Selfy AI 的易经观相助手。必须使用函数 submit_analysis_v3 返回严格 JSON。"
        "\n【呈现顺序】三象逐一解读 → 三卦组合 → 总结 → 金钱与事业/配偶与感情。"
        "\n【三象定义】仅分析 姿态/神情/面相（忽略服饰与背景）。"
        "\n  - 为每一象产出四项：gua（卦名，如“艮（山）/离（火）/兑（泽）…”）、特征（客观描述）、解读（卦意启示）、性格倾向（1句稳定倾向）。"
        "\n  - 这些内容需同时写入："
        "\n      1) sections（合成一句：'特征；卦象：X；解读：…；性格倾向：…'）"
        "\n      2) meta.sections_rich.{姿态|神情|面相}（四字段分开，便于 UI 分行渲染）"
        "\n【卦象参考锚点】偏好而非唯一：姿态→艮/乾/兑/坤；神情→离/坎/震/巽；面相→离/乾/坤/兑/艮。"
        "\n【三卦组合（meta.combo）】"
        "\n  - gua_list：按贡献度排序的 3 个卦名（不含括号）。"
        "\n  - bullets：2–4 条短句（7–10 字/条），如“外冷内热”“独立审美”。"
        "\n  - summary：40–70 字，小段概括“外在/内在/对人”的关系。"
        "\n【summary】80–120字：一句印象 + 两个最强依据 + 1句意境。"
        "\n【domains】只能是数组：['金钱与事业','配偶与感情'] 中的若干。详细文案放 meta.domains_detail："
        "\n  - 每个领域 160–220 字；写 2 条可执行动作（含时机/频率）+ 1 个失败预警信号；避免空话。"
        "\n【其他元数据】meta.triple_analysis 可简述“形-气-神联动”。生成 meta.combo_title= '姿态卦 + 神情卦 + 面相卦'。"
        "\n【长度与口径】总体不超过约 1200 字；信息不足要说明不确定来源并下调 confidence；不做命断。"
        "\n【仅用工具输出】禁止在消息体自由文本输出。"
    )
    user = "请分析这张图片，先三象逐一解读（含卦/特征/解读/性格倾向），再给三卦组合与总结，最后写更厚的事业/感情建议。只用工具函数返回 JSON。"
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
# 统一整形：补齐 meta.sections_rich / meta.combo；保证四区块内容完整且顺序可渲染
def _coerce_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    - 任何字段即便是 str（甚至是 JSON 字符串）也不崩；
    - 先用 meta.sections_rich 驱动三象卡片；缺失则从 meta.triple_analysis 或 sections 拼装；
    - 生成/兜底 meta.combo（gua_list/bullets/summary）与 meta.combo_title；
    - domains 仅保留 ['金钱与事业','配偶与感情']，长文进 meta.domains_detail（加厚兜底）。
    """
    allowed = {"金钱与事业", "配偶与感情"}

    out = dict(data) if isinstance(data, dict) else {}
    meta = out.get("meta")
    if isinstance(meta, str):
        try: meta = json.loads(meta)
        except Exception: meta = {}
    if not isinstance(meta, dict): meta = {}
    out["meta"] = meta

    # ---------- 解析 triple_analysis ----------
    def _as_dict(x):
        if isinstance(x, str):
            try: return json.loads(x)
            except Exception: return {}
        return x if isinstance(x, dict) else {}
    triple = _as_dict(meta.get("triple_analysis") or {})
    if not isinstance(triple, dict): triple = {}

    # ---------- sections_rich：优先来源 ----------
    sr = _as_dict(meta.get("sections_rich") or {})
    def _norm_seg(o: Any) -> Dict[str, str]:
        if isinstance(o, str):
            try: o = json.loads(o)
            except Exception: o = {}
        if not isinstance(o, dict): o = {}
        return {
            "gua": o.get("gua",""),
            "特征": o.get("特征",""),
            "解读": o.get("解读",""),
            "性格倾向": o.get("性格倾向",""),
        }

    # 若无 sections_rich，则尽量从 triple 或原 sections 拆解
    if not sr:
        # 从 triple 的“姿态/神情/面容(或面相)”尝试恢复
        def _from_triple(key, fallback):
            o = triple.get(key) or {}
            if isinstance(o, str):
                try: o = json.loads(o)
                except Exception: o = {}
            if not isinstance(o, dict): o = {}
            desc = o.get("说明") or ""
            hexg = o.get("卦象") or ""
            mean = o.get("解读") or ""
            tend = o.get("性格倾向") or ""
            return {"gua": hexg, "特征": desc, "解读": mean, "性格倾向": tend} if any([desc,hexg,mean,tend]) else {}
        sr = {
            "姿态": _from_triple("姿态", "姿态"),
            "神情": _from_triple("神情", "神情"),
            "面相": _from_triple("面容", "面相"),
        }

    # 如果还缺，尽量用老的 sections 文本粗分解（无法拆分则仅填“特征”）
    sections_raw = out.get("sections") if isinstance(out.get("sections"), dict) else {}
    for k_old, k in (("姿态","姿态"),("神情","神情"),("面相","面相")):
        if k not in sr or not isinstance(sr[k], dict) or not any(sr[k].values()):
            text = sections_raw.get(k_old, "") if isinstance(sections_raw, dict) else ""
            sr[k] = {"gua":"", "特征": text, "解读":"", "性格倾向":""}

    # 归一化
    sr["姿态"] = _norm_seg(sr.get("姿态"))
    sr["神情"] = _norm_seg(sr.get("神情"))
    sr["面相"] = _norm_seg(sr.get("面相"))

    meta["sections_rich"] = sr

    # 用 sections_rich 合成老的 sections 一句话版本（兼容前端现有渲染）
    def _join_line(seg: Dict[str,str]) -> str:
        parts = [seg.get("特征","").strip()]
        if seg.get("gua"): parts.append(f"卦象：{seg['gua']}")
        if seg.get("解读"): parts.append(seg["解读"].strip())
        if seg.get("性格倾向"): parts.append(seg["性格倾向"].strip())
        return "；".join([p for p in parts if p]).strip("；")

    out["sections"] = {
        "姿态": _join_line(sr["姿态"]),
        "神情": _join_line(sr["神情"]),
        "面相": _join_line(sr["面相"]),
    }

    # ---------- 组合块：meta.combo ----------
    combo = _as_dict(meta.get("combo") or {})
    def _non_empty_list(x):
        return x if isinstance(x, list) else []

    # 计算组合标题与默认 gua_list
    gua_list = _non_empty_list(combo.get("gua_list"))
    if not gua_list:
        gl = [sr["姿态"]["gua"], sr["神情"]["gua"], sr["面相"]["gua"]]
        gua_list = [g.replace("（","(").split("(")[0] for g in gl if g]
    bullets = _non_empty_list(combo.get("bullets"))
    if not bullets:
        bullets = ["外冷内热","独立审美","交际选择性强"][:max(2, min(4, len(gua_list) or 3))]
    summary_short = combo.get("summary","").strip()
    if not summary_short:
        summary_short = "此组合外稳内燃：外在克制、内心自持且重审美取舍，对人际更注重深度与真诚。"
    combo = {"gua_list": gua_list, "bullets": bullets, "summary": summary_short}
    meta["combo"] = combo

    # 组合标题（如 “艮 + 离 + 兑”）
    meta["combo_title"] = " + ".join([g for g in gua_list if g]) or meta.get("combo_title") or ""

    # ---------- 总结与 archetype/confidence 兜底 ----------
    out["summary"] = out.get("summary") or "外在沉稳冷静，内里有热度与坚持，重自我边界与审美。"
    out["archetype"] = out.get("archetype") or "外冷内热"
    try:
        out["confidence"] = float(out.get("confidence", 0.85))
    except Exception:
        out["confidence"] = 0.85

    # ---------- domains / domains_detail ----------
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
        if len(txt) < 120:  # 加厚到 160–220 目标，至少给一版可靠文本
            if key == "金钱与事业":
                dd[key] = (
                    f"{arche}型：稳中求进、质量导向。建议①『小步快跑+两周复盘』：把任务拆成 2–3 个迭代，每 14 天复盘一次，记录复盘结论并固化 SOP；"
                    "②『对外协作位』：为关键环节设 1 个外协位（技术/渠道/财务其一），每周固定 30 分钟沟通。预警：当你出现“一次性押注/长时间闭环不汇报”时，风险增大，应立刻缩小试错尺度并公开里程碑。"
                )
            else:  # 配偶与感情
                dd[key] = (
                    f"{arche}型：外冷里热、重边界与真诚。建议①『节律沟通』：每周固定 1 次 30 分钟对话，仅谈事实—感受—需求，避免追责；"
                    "②『回应时间窗』：收到重要信息 4 小时内给出简短回应，晚些再细聊，降低对方不确定感。预警：当你连续两次回避表达或过度理性化解释，对方会感到疏离，需要补偿式表达（夸张亲密举动或共享计划）。"
                )
        # domains 列表确保包含 key
        if key not in out["domains"]: out["domains"].append(key)

    _ensure_heavy("金钱与事业", out.get("archetype") or "外冷内热")
    _ensure_heavy("配偶与感情", out.get("archetype") or "外冷内热")
    meta["domains_detail"] = dd

    # ---------- 回写 triple_analysis（保持兼容：包含四键） ----------
    def _to_triple_seg(seg: Dict[str,str]) -> Dict[str,str]:
        return {"说明": seg.get("特征",""), "卦象": seg.get("gua",""), "解读": seg.get("解读",""), "性格倾向": seg.get("性格倾向","")}
    triple["姿态"] = _to_triple_seg(sr["姿态"])
    triple["神情"] = _to_triple_seg(sr["神情"])
    triple["面容"] = _to_triple_seg(sr["面相"])
    triple["面相"] = dict(triple["面容"])
    if not triple.get("组合意境"):
        triple["组合意境"] = meta.get("combo_title","")
    if not triple.get("总结"):
        triple["总结"] = out["summary"]
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
