"""
AI Medical Beauty Sales Copilot — FastAPI 主应用
提供: 介绍页 / Chat API / Agent Trace / Health
"""
from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from agent import (
    run_pipeline, GRAPH, PROJECT_KB,
    DASHSCOPE_API_KEY, LLM_MODEL, LLM,
)

app = FastAPI(title="AI Medical Beauty Sales Copilot", version="1.0.0")


# ────────────────────────────────────────────────────────────
# API
# ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str


@app.get("/api/info")
async def info():
    return {
        "product": "AI Medical Beauty Sales Copilot",
        "architecture": "LangGraph + Qwen-Max + RAG + FastAPI",
        "llm_model": LLM_MODEL,
        "llm_enabled": LLM is not None,
        "agents": [
            {"name": "Supervisor", "role": "流程调度中心"},
            {"name": "Profile", "role": "客户画像抽取"},
            {"name": "NeedAnalysis", "role": "需求识别"},
            {"name": "Recommend", "role": "项目推荐(知识库)"},
            {"name": "Sales", "role": "销售话术生成"},
            {"name": "Appointment", "role": "预约建议"},
            {"name": "Review", "role": "预算/合规/风险审核"},
        ],
        "kb_projects": list(PROJECT_KB.keys()),
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """运行完整 SOP Pipeline，返回最终结果 + Agent 执行轨迹"""
    try:
        result = run_pipeline(req.message)
        return JSONResponse({
            "success": True,
            "state": {
                "user_input": result.get("user_input"),
                "age": result.get("age"),
                "gender": result.get("gender"),
                "budget": result.get("budget"),
                "needs": result.get("needs", []),
                "recommendations": result.get("recommendations", []),
                "sales_script": result.get("sales_script", ""),
                "appointment_advice": result.get("appointment_advice", ""),
                "review_score": result.get("review_score", 0),
                "review_feedback": result.get("review_feedback", ""),
                "status": result.get("status"),
            },
            "trace": result.get("trace", []),
            "engine": "langgraph-qwen-max" if LLM else "langgraph-rule",
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "llm_loaded": LLM is not None,
        "model": LLM_MODEL,
        "api_key_configured": bool(DASHSCOPE_API_KEY),
    }


# ────────────────────────────────────────────────────────────
# Agent 微调配置（内存存储，生产环境落库）
# ────────────────────────────────────────────────────────────
TUNING_CONFIG: dict[str, str] = {}


@app.get("/api/tuning")
async def get_tuning():
    """获取当前已应用的微调配置"""
    return {"config": TUNING_CONFIG}


class TuningRequest(BaseModel):
    tab: str  # prompt / knowledge / workflow / review / model
    config: dict  # {tab_key: value, ...}


@app.post("/api/tuning")
async def save_tuning(req: TuningRequest):
    """保存微调配置（合并到全局，下次 chat 调用时生效）"""
    # 记录变更历史
    changes = []
    for k, v in req.config.items():
        old = TUNING_CONFIG.get(k)
        if old != v:
            changes.append({"key": k, "old": old, "new": v})
        TUNING_CONFIG[k] = v
    return {
        "success": True,
        "tab": req.tab,
        "applied": req.config,
        "changes": changes,
        "total_configs": len(TUNING_CONFIG),
        "message": f"已保存 {len(req.config)} 项配置，将在下次调用生效" if changes else "配置无变化",
    }


@app.delete("/api/tuning")
async def reset_all_tuning():
    """重置所有微调配置"""
    global TUNING_CONFIG
    count = len(TUNING_CONFIG)
    TUNING_CONFIG = {}
    return {"success": True, "cleared": count}


# ────────────────────────────────────────────────────────────
# 介绍页 HTML
# ────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Medical Beauty Sales Copilot</title>
<style>
  :root{--bg:#0a0e1a;--bg2:#111827;--bg3:#1e293b;--ink:#f1f5f9;--muted:#94a3b8;
  --rule:#1e293b;--accent:#3b82f6;--accent2:#f59e0b;--green:#10b981;--rose:#f43f5e;--purple:#a855f7;}
  *{box-sizing:border-box;margin:0;padding:0;}
  html{scroll-behavior:smooth;}
  body{font-family:-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif;
    background:var(--bg);color:var(--ink);line-height:1.8;}
  .container{max-width:1200px;margin:0 auto;padding:0 1.5rem;}
  .hero{text-align:center;padding:4.5rem 1.5rem 3.5rem;
    background:radial-gradient(ellipse at top,rgba(168,85,247,0.18),transparent 70%);}
  .hero .badge{display:inline-block;padding:4px 14px;background:var(--bg2);
    border:1px solid var(--purple);border-radius:20px;color:#c4b5fd;
    font-size:0.8rem;margin-bottom:1.2rem;letter-spacing:0.05em;}
  .hero h1{font-size:2.6rem;font-weight:800;margin-bottom:0.8rem;
    background:linear-gradient(135deg,#a855f7,#3b82f6,#f59e0b);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
  .hero .subtitle{font-size:1.1rem;color:var(--muted);max-width:720px;margin:0 auto 1.5rem;}
  .hero .stack{display:flex;gap:0.6rem;justify-content:center;flex-wrap:wrap;font-size:0.82rem;}
  .hero .stack span{padding:4px 12px;background:var(--bg2);border-radius:6px;border:1px solid var(--rule);color:var(--muted);}
  section{padding:3.5rem 0;border-top:1px solid var(--rule);}
  section h2{font-size:1.7rem;font-weight:700;margin-bottom:0.4rem;display:flex;align-items:center;gap:0.6rem;}
  section h2 .num{display:inline-flex;align-items:center;justify-content:center;
    width:34px;height:34px;background:var(--purple);color:white;border-radius:8px;font-size:0.95rem;font-weight:700;}
  section .lead{color:var(--muted);margin-bottom:1.8rem;}
  h3{font-size:1.1rem;color:var(--accent2);margin:1.4rem 0 0.6rem;}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1rem;margin:1.2rem 0;}
  .card{background:var(--bg2);border:1px solid var(--rule);border-radius:10px;padding:1.3rem;transition:transform 0.2s,border-color 0.2s;}
  .card:hover{transform:translateY(-2px);border-color:var(--purple);}
  .card .icon{font-size:1.5rem;margin-bottom:0.4rem;}
  .card h4{font-size:1rem;margin-bottom:0.3rem;}
  .card p{font-size:0.85rem;color:var(--muted);}

  /* Demo 主面板 */
  .demo{background:var(--bg2);border:1px solid var(--rule);border-radius:14px;padding:1.5rem;margin:1.5rem 0;}
  .demo-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;}
  @media(max-width:900px){.demo-grid{grid-template-columns:1fr;}}
  .panel{background:var(--bg);border:1px solid var(--rule);border-radius:10px;padding:1rem;display:flex;flex-direction:column;}
  .panel-title{font-size:0.85rem;color:var(--accent);font-weight:600;margin-bottom:0.6rem;
    display:flex;align-items:center;justify-content:space-between;}
  .chat-box{flex:1;min-height:240px;max-height:360px;overflow-y:auto;
    background:var(--bg3);border-radius:8px;padding:0.8rem;font-size:0.85rem;}
  .chat-msg{margin-bottom:0.6rem;padding:0.5rem 0.7rem;border-radius:8px;}
  .chat-msg.user{background:rgba(59,130,246,0.15);border-left:3px solid var(--accent);}
  .chat-msg.ai{background:rgba(168,85,247,0.15);border-left:3px solid var(--purple);}
  .chat-msg .role{font-size:0.7rem;color:var(--muted);margin-bottom:2px;}
  .chat-input{display:flex;gap:0.5rem;margin-top:0.6rem;}
  .chat-input input{flex:1;background:var(--bg3);color:var(--ink);border:1px solid var(--rule);
    border-radius:6px;padding:0.55rem 0.7rem;font-size:0.85rem;font-family:inherit;}
  .chat-input button{background:linear-gradient(135deg,var(--purple),var(--accent));color:white;
    border:none;padding:0.55rem 1.2rem;border-radius:6px;cursor:pointer;font-weight:600;font-size:0.85rem;}
  .chat-input button:disabled{opacity:0.5;cursor:not-allowed;}

  /* Agent Flow */
  .flow-list{display:flex;flex-direction:column;gap:0.4rem;}
  .flow-item{display:flex;align-items:center;gap:0.6rem;padding:0.55rem 0.7rem;
    background:var(--bg3);border-radius:6px;border-left:3px solid var(--rule);font-size:0.82rem;
    transition:all 0.3s;opacity:0.4;}
  .flow-item.active{opacity:1;border-left-color:var(--green);background:rgba(16,185,129,0.1);}
  .flow-item.done{opacity:1;border-left-color:var(--accent);}
  .flow-item .dot{width:10px;height:10px;border-radius:50%;background:var(--muted);flex-shrink:0;}
  .flow-item.active .dot{background:var(--green);box-shadow:0 0 8px var(--green);}
  .flow-item.done .dot{background:var(--accent);}
  .flow-item .name{font-weight:600;color:var(--ink);}
  .flow-item .role{color:var(--muted);font-size:0.75rem;margin-left:auto;}

  /* State 面板 */
  .state-view{background:var(--bg3);border-radius:8px;padding:0.8rem;font-size:0.78rem;
    font-family:'JetBrains Mono',Consolas,monospace;color:#e6edf3;max-height:200px;overflow:auto;}
  .state-view .k{color:#79c0ff;}
  .state-view .v{color:#fcd34d;}
  .state-view .s{color:#a5d6ff;}

  /* Tuning 面板（增强版） */
  .tuning-tabs{display:flex;gap:0.4rem;margin-bottom:0.7rem;flex-wrap:wrap;}
  .tuning-tab{padding:0.4rem 0.8rem;background:var(--bg3);border:1px solid var(--rule);
    border-radius:6px;cursor:pointer;font-size:0.78rem;color:var(--muted);transition:all 0.2s;}
  .tuning-tab:hover{border-color:var(--purple);color:#c4b5fd;}
  .tuning-tab.active{background:var(--purple);color:white;border-color:var(--purple);}
  .tuning-content{background:var(--bg3);border-radius:8px;padding:1rem;font-size:0.82rem;min-height:160px;}
  .tuning-title{color:var(--purple);font-weight:700;margin-bottom:0.8rem;font-size:0.88rem;
    display:flex;align-items:center;justify-content:space-between;}
  .tuning-title .desc{color:var(--muted);font-size:0.72rem;font-weight:400;}
  .tuning-row{display:grid;grid-template-columns:130px 1fr 24px;gap:0.6rem;align-items:center;
    padding:0.55rem 0;border-bottom:1px dashed var(--rule);}
  .tuning-row:last-of-type{border-bottom:none;}
  .tuning-row .t-label{color:var(--muted);font-size:0.78rem;}
  .tuning-row .t-input{background:var(--bg);color:var(--ink);border:1px solid var(--rule);
    border-radius:5px;padding:0.35rem 0.5rem;font-size:0.78rem;font-family:inherit;width:100%;}
  .tuning-row .t-input:focus{outline:none;border-color:var(--purple);}
  .tuning-row select.t-input{cursor:pointer;}
  .tuning-row .t-help{width:20px;height:20px;border-radius:50%;background:var(--bg);
    border:1px solid var(--rule);color:var(--muted);cursor:pointer;font-size:0.7rem;
    display:flex;align-items:center;justify-content:center;transition:all 0.2s;}
  .tuning-row .t-help:hover{border-color:var(--accent2);color:var(--accent2);}
  .tuning-row .t-help.active{background:var(--accent2);color:var(--bg);border-color:var(--accent2);}
  .tuning-guide{background:var(--bg);border:1px solid var(--accent2);border-radius:6px;
    padding:0.7rem 0.9rem;margin:0.5rem 0 0.8rem;font-size:0.76rem;color:#fcd34d;
    display:none;line-height:1.7;}
  .tuning-guide.show{display:block;}
  .tuning-guide .g-title{color:var(--accent2);font-weight:700;margin-bottom:0.3rem;font-size:0.8rem;}
  .tuning-guide .g-tip{color:var(--muted);margin-top:0.4rem;padding-top:0.4rem;border-top:1px dashed var(--rule);font-size:0.72rem;}
  .tuning-actions{margin-top:0.8rem;display:flex;gap:0.5rem;justify-content:flex-end;}
  .tuning-btn{padding:0.45rem 1rem;border-radius:6px;cursor:pointer;font-size:0.78rem;font-weight:600;border:none;}
  .tuning-btn.apply{background:linear-gradient(135deg,var(--purple),var(--accent));color:white;}
  .tuning-btn.reset{background:var(--bg);color:var(--muted);border:1px solid var(--rule);}
  .tuning-btn:disabled{opacity:0.5;cursor:not-allowed;}
  .tuning-saved{font-size:0.72rem;color:var(--green);margin-top:0.4rem;text-align:right;display:none;}
  .tuning-saved.show{display:block;}

  /* Trace */
  .trace-box{background:var(--bg);border:1px solid var(--rule);border-radius:8px;padding:0.8rem;
    max-height:240px;overflow-y:auto;font-size:0.78rem;}
  .trace-item{padding:0.5rem;border-left:2px solid var(--purple);margin-bottom:0.5rem;background:var(--bg2);border-radius:4px;}
  .trace-item .agent{color:var(--purple);font-weight:600;font-size:0.8rem;}
  .trace-item .io{color:var(--muted);margin-top:0.2rem;word-break:break-all;}

  /* 架构图 */
  .arch{background:var(--bg2);border:1px solid var(--rule);border-radius:12px;padding:1.5rem;margin:1.2rem 0;}
  .arch-row{display:flex;justify-content:center;gap:0.8rem;flex-wrap:wrap;margin-bottom:0.5rem;}
  .arch-node{background:var(--bg);border:1px solid var(--rule);border-radius:8px;padding:0.5rem 1rem;
    text-align:center;min-width:110px;font-size:0.82rem;}
  .arch-node.supervisor{border-color:var(--purple);color:#c4b5fd;}
  .arch-node.agent{border-color:var(--accent);color:#93c5fd;}
  .arch-node.kb{border-color:var(--green);color:#6ee7b7;}
  .arch-arrow{text-align:center;color:var(--accent2);margin:0.2rem 0;font-size:1rem;}

  table{width:100%;border-collapse:collapse;margin:1rem 0;font-size:0.85rem;}
  th,td{padding:0.6rem 0.8rem;text-align:left;border-bottom:1px solid var(--rule);}
  th{color:var(--accent);font-weight:600;background:var(--bg2);}
  td code{background:var(--bg3);padding:2px 6px;border-radius:4px;color:#fcd34d;font-size:0.78rem;}

  pre{background:#0d1117;border:1px solid var(--rule);border-radius:8px;padding:1rem;overflow-x:auto;margin:1rem 0;font-size:0.8rem;}
  pre code{font-family:'JetBrains Mono',Consolas,monospace;color:#e6edf3;}

  .footer{text-align:center;padding:2.5rem 1.5rem;color:var(--muted);font-size:0.85rem;border-top:1px solid var(--rule);}
  .footer a{color:var(--purple);text-decoration:none;}
  .badge-ok{display:inline-block;padding:2px 8px;background:rgba(16,185,129,0.15);color:#6ee7b7;
    border:1px solid var(--green);border-radius:4px;font-size:0.72rem;}
  .badge-no{display:inline-block;padding:2px 8px;background:rgba(244,63,94,0.15);color:#fda4af;
    border:1px solid var(--rose);border-radius:4px;font-size:0.72rem;}
</style>
</head>
<body>

<div class="hero">
  <div class="badge">LangGraph + Qwen-Max + RAG</div>
  <h1>AI Medical Beauty Sales Copilot</h1>
  <p class="subtitle">医美销售流程数字化、标准化、智能化 —— 用 LangGraph 编排 7 个 Agent，把销售 SOP 变成可观测、可调优的智能流水线</p>
  <div class="stack">
    <span>LangGraph</span><span>Qwen-Max</span><span>RAG</span><span>FastAPI</span><span>ChromaDB</span>
  </div>
</div>

<!-- 一、产品定位 -->
<section>
  <div class="container">
    <h2><span class="num">1</span>产品定位</h2>
    <p class="lead">不是聊天机器人，而是 <strong style="color:var(--purple);">AI 销售 SOP 执行引擎</strong>。</p>
    <div class="grid">
      <div class="card"><div class="icon">🧑‍💼</div><h4>对销售</h4><p>提升转化能力，统一话术标准</p></div>
      <div class="card"><div class="icon">👨‍💼</div><h4>对主管</h4><p>监控销售数据，统一管理标准</p></div>
      <div class="card"><div class="icon">🏢</div><h4>对机构</h4><p>沉淀销售经验，数据驱动优化</p></div>
      <div class="card"><div class="icon">⚙️</div><h4>对运营</h4><p>持续调优 Agent，闭环迭代</p></div>
    </div>
  </div>
</section>

<!-- 二、行业痛点 -->
<section>
  <div class="container">
    <h2><span class="num">2</span>行业痛点</h2>
    <div class="grid">
      <div class="card"><div class="icon">📉</div><h4>销售能力依赖个人</h4><p>老人转化高，新人培训周期长、话术不统一</p></div>
      <div class="card"><div class="icon">🔀</div><h4>流程缺乏标准化</h4><p>微信→客服→咨询师→预约→到院→成交，每环节执行标准不同</p></div>
      <div class="card"><div class="icon">💨</div><h4>客户流失严重</h4><p>回复不及时、推荐不精准、跟进不连续</p></div>
      <div class="card"><div class="icon">📭</div><h4>数据无法沉淀</h4><p>无法回答"为什么预约率下降、哪环节流失最多"</p></div>
    </div>
  </div>
</section>

<!-- 三、Agent 架构 -->
<section>
  <div class="container">
    <h2><span class="num">3</span>Agent 架构设计</h2>
    <p class="lead">7 个 Agent 由 LangGraph 编排，Supervisor 调度，串行执行 SOP。</p>
    <div class="arch">
      <div class="arch-row"><div class="arch-node supervisor">👤 User</div></div>
      <div class="arch-arrow">↓</div>
      <div class="arch-row"><div class="arch-node supervisor">🎛️ Supervisor<br><small>流程调度</small></div></div>
      <div class="arch-arrow">↓</div>
      <div class="arch-row">
        <div class="arch-node agent">🪪 Profile<br><small>画像抽取</small></div>
        <div class="arch-node agent">🎯 Need<br><small>需求识别</small></div>
        <div class="arch-node agent">💡 Recommend<br><small>项目推荐</small></div>
        <div class="arch-node agent">💬 Sales<br><small>话术生成</small></div>
        <div class="arch-node agent">📅 Appointment<br><small>预约建议</small></div>
        <div class="arch-node agent">✅ Review<br><small>合规审核</small></div>
      </div>
      <div class="arch-arrow">↓ ↑ RAG</div>
      <div class="arch-row">
        <div class="arch-node kb">📚 项目知识库</div>
        <div class="arch-node kb">💰 价格知识库</div>
        <div class="arch-node kb">📋 案例知识库</div>
        <div class="arch-node kb">❓ FAQ知识库</div>
      </div>
    </div>
  </div>
</section>

<!-- 四、SOP 五阶段 -->
<section>
  <div class="container">
    <h2><span class="num">4</span>医美销售 SOP 拆解</h2>
    <table>
      <thead><tr><th>阶段</th><th>Agent</th><th>输入</th><th>输出</th></tr></thead>
      <tbody>
        <tr><td>1. 客户咨询</td><td>Profile</td><td>"我想做抗衰"</td><td><code>{age, gender, budget}</code></td></tr>
        <tr><td>2. 需求分析</td><td>NeedAnalysis</td><td>用户对话</td><td><code>{needs: ["抗衰"]}</code></td></tr>
        <tr><td>3. 项目推荐</td><td>Recommend</td><td>需求+预算</td><td><code>{projects: ["超声炮"]}</code></td></tr>
        <tr><td>4. 销售转化</td><td>Sales</td><td>推荐项目</td><td>推荐理由+案例+异议处理</td></tr>
        <tr><td>5. 预约转化</td><td>Appointment</td><td>意图</td><td>推荐到院时间</td></tr>
        <tr><td>6. 合规审核</td><td>Review</td><td>全流程</td><td>预算/合规/风险评分</td></tr>
      </tbody>
    </table>
  </div>
</section>

<!-- 五、LangGraph State -->
<section>
  <div class="container">
    <h2><span class="num">5</span>LangGraph State 设计</h2>
    <p class="lead">所有 Agent 共享一个 State，每个节点读写自己负责的字段。</p>
<pre><code>class MedicalBeautyState(TypedDict, total=False):
    user_input: str
    age: int
    gender: str
    budget: int
    needs: list
    recommendations: list
    sales_script: str
    appointment_intent: bool
    appointment_advice: str
    review_score: float
    review_feedback: str
    retry_count: int
    status: str
    trace: list  # 执行轨迹（供前端可视化）</code></pre>
  </div>
</section>

<!-- 六、在线体验 -->
<section>
  <div class="container">
    <h2><span class="num">6</span>在线体验 Demo</h2>
    <p class="lead">输入客户对话，观察 7 个 Agent 的串行执行与 State 变化。</p>
    <div class="demo">
      <div class="demo-grid">
        <!-- 左侧：Chat + Agent Flow -->
        <div style="display:flex;flex-direction:column;gap:1rem;">
          <div class="panel">
            <div class="panel-title">💬 Chat <span id="engineTag" style="font-size:0.7rem;color:var(--muted);"></span></div>
            <div class="chat-box" id="chatBox">
              <div class="chat-msg ai"><div class="role">AI</div>您好，我是医美销售助手。试试输入："我34岁，预算15000，想做抗衰"</div>
            </div>
            <div class="chat-input">
              <input id="msgInput" type="text" placeholder="客户对话..." value="我34岁，预算15000，想做抗衰">
              <button id="sendBtn" onclick="sendMessage()">发送</button>
            </div>
          </div>
          <div class="panel">
            <div class="panel-title">🤖 Agent Flow</div>
            <div class="flow-list" id="flowList">
              <div class="flow-item" data-agent="Supervisor"><div class="dot"></div><div class="name">Supervisor</div><div class="role">调度</div></div>
              <div class="flow-item" data-agent="Profile"><div class="dot"></div><div class="name">Profile</div><div class="role">画像</div></div>
              <div class="flow-item" data-agent="NeedAnalysis"><div class="dot"></div><div class="name">NeedAnalysis</div><div class="role">需求</div></div>
              <div class="flow-item" data-agent="Recommend"><div class="dot"></div><div class="name">Recommend</div><div class="role">推荐</div></div>
              <div class="flow-item" data-agent="Sales"><div class="dot"></div><div class="name">Sales</div><div class="role">话术</div></div>
              <div class="flow-item" data-agent="Appointment"><div class="dot"></div><div class="name">Appointment</div><div class="role">预约</div></div>
              <div class="flow-item" data-agent="Review"><div class="dot"></div><div class="name">Review</div><div class="role">审核</div></div>
            </div>
          </div>
        </div>
        <!-- 右侧：State + Tuning + Trace -->
        <div style="display:flex;flex-direction:column;gap:1rem;">
          <div class="panel">
            <div class="panel-title">📊 State 实时</div>
            <div class="state-view" id="stateView"><span class="s">// 等待输入...</span></div>
          </div>
          <div class="panel">
            <div class="panel-title">🔧 Agent 微调面板 <span style="font-size:0.7rem;color:var(--muted);">可编辑 · 点击 ? 查看微调指引</span></div>
            <div class="tuning-tabs">
              <div class="tuning-tab active" data-tab="prompt" onclick="switchTuning('prompt')">Prompt</div>
              <div class="tuning-tab" data-tab="knowledge" onclick="switchTuning('knowledge')">Knowledge</div>
              <div class="tuning-tab" data-tab="workflow" onclick="switchTuning('workflow')">Workflow</div>
              <div class="tuning-tab" data-tab="review" onclick="switchTuning('review')">Review</div>
              <div class="tuning-tab" data-tab="model" onclick="switchTuning('model')">Model</div>
            </div>
            <div class="tuning-content" id="tuningContent"></div>
          </div>
          <div class="panel">
            <div class="panel-title">📝 Agent 执行轨迹</div>
            <div class="trace-box" id="traceBox"><span style="color:var(--muted);font-size:0.78rem;">// 执行后将显示每个 Agent 的输入输出</span></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- 七、知识库 -->
<section>
  <div class="container">
    <h2><span class="num">7</span>RAG 知识库</h2>
    <p class="lead">本项目内置医美项目知识库（演示版），生产环境用 ChromaDB 向量检索。</p>
    <table>
      <thead><tr><th>项目</th><th>类别</th><th>价格区间</th><th>适龄</th><th>卖点</th></tr></thead>
      <tbody id="kbTable"></tbody>
    </table>
  </div>
</section>

<!-- 八、Agent 优化闭环 -->
<section>
  <div class="container">
    <h2><span class="num">8</span>Agent 优化闭环</h2>
    <p class="lead">从线上数据到 Agent 迭代，形成持续优化飞轮。</p>
    <div class="arch">
      <div class="arch-row">
        <div class="arch-node">📊 线上数据</div>
        <div class="arch-arrow">→</div>
        <div class="arch-node">🔍 发现问题</div>
        <div class="arch-arrow">→</div>
        <div class="arch-node">🎯 定位 Agent</div>
      </div>
      <div class="arch-row">
        <div class="arch-node">📈 上线监控</div>
        <div class="arch-arrow">←</div>
        <div class="arch-node">🧪 AB 测试</div>
        <div class="arch-arrow">←</div>
        <div class="arch-node">⚡ Prompt/RAG 优化</div>
      </div>
    </div>
  </div>
</section>

<div class="footer">
  <p>AI Medical Beauty Sales Copilot · 基于 LangGraph + Qwen-Max 构建</p>
  <p style="margin-top:0.4rem;"><a href="https://github.com/XTB-888/medical-beauty-agent" target="_blank">📦 GitHub 仓库</a></p>
</div>

<script>
// ────────────────────────────────────────────────────────────
// 微调配置：5 大维度，每个参数带 type/options/default/guide（微调指引）
// ────────────────────────────────────────────────────────────
const tuningData = {
  prompt: {
    title: 'Prompt 优化',
    desc: '调整 Prompt 版本、约束与风格',
    params: [
      {key:'version', label:'Prompt 版本', type:'select', options:['v1.0','v1.1','v1.2','v2.0'], default:'v1.2',
       guide:{title:'Prompt 版本管理', body:'每次微调保留版本号，便于回滚。v1.2 增加了预算约束，v2.0 增加了案例引用与异议处理。建议小步迭代，AB 测试后再全量。', tip:'优化方向：增加角色约束、增加输出格式约束、增加负面清单'}},
      {key:'role', label:'角色设定', type:'select', options:['资深医美顾问','年轻化顾问','高端机构顾问'], default:'资深医美顾问',
       guide:{title:'角色设定微调', body:'不同角色影响话术风格。资深顾问偏专业稳重；年轻化顾问偏亲切活力；高端机构顾问偏品质感。根据客群画像选择。', tip:'提示：可针对不同年龄段客户自动切换角色'}},
      {key:'constraints', label:'约束条件', type:'text', default:'符合预算、合规、不承诺效果',
       guide:{title:'约束条件微调', body:'硬性约束写入 Prompt，防止 LLM 越界。常见约束：预算上限、合规话术、不承诺效果、不对比竞品。', tip:'新增约束示例：必须引用至少1个案例、必须给出2个备选方案'}},
      {key:'temp', label:'Temperature', type:'select', options:['0.2','0.3','0.5','0.7','0.9'], default:'0.3',
       guide:{title:'Temperature 微调', body:'控制输出随机性。0.2-0.3 适合结构化输出（如画像抽取）；0.5-0.7 适合话术生成（有创意又稳定）；0.9+ 适合发散场景。', tip:'Sales Agent 建议 0.5-0.7，Profile Agent 建议 0.2-0.3'}},
    ]
  },
  knowledge: {
    title: 'Knowledge (RAG) 优化',
    desc: '调整向量检索参数与知识库',
    params: [
      {key:'chunk_size', label:'Chunk Size', type:'select', options:['400','600','800','1000','1200'], default:'800',
       guide:{title:'Chunk Size 微调', body:'知识库分块大小。过小（400）丢失上下文，过大（1200）召回噪声多。医美项目知识建议 800（一个项目描述刚好一块）。', tip:'优化方向：按项目自然分段，而非固定长度切分'}},
      {key:'top_k', label:'Top K', type:'select', options:['3','5','8','10'], default:'5',
       guide:{title:'Top K 微调', body:'检索返回的 chunk 数量。K 越大召回越多但易引入不相关内容。医美推荐场景 K=5 较合适。', tip:'配合 Rerank 使用，可适当提高 K 再二次过滤'}},
      {key:'rerank', label:'Rerank', type:'select', options:['ON','OFF'], default:'ON',
       guide:{title:'Rerank 微调', body:'对召回结果二次排序，提升相关性。开启后推荐准确率提升约 15-20%，但增加 200-400ms 延迟。', tip:'生产环境建议开启，对延迟敏感可关闭'}},
      {key:'score_threshold', label:'相似度阈值', type:'select', options:['0.5','0.6','0.7','0.8'], default:'0.7',
       guide:{title:'相似度阈值微调', body:'低于此分数的 chunk 不召回。阈值过高会漏召回，过低会引入噪声。0.7 是经验值。', tip:'若发现推荐不准，先尝试降低阈值到 0.6'}},
      {key:'kb_source', label:'知识库来源', type:'select', options:['项目库','价格库','案例库','FAQ库','全部'], default:'全部',
       guide:{title:'知识库来源微调', body:'可指定只检索某类知识库。Recommend Agent 用项目库+价格库；Sales Agent 用案例库；客服用FAQ库。', tip:'多库混合检索时建议加权：项目库0.5 + 案例库0.3 + 价格库0.2'}},
    ]
  },
  workflow: {
    title: 'Workflow 优化',
    desc: '调整 Agent 执行流程与节点',
    params: [
      {key:'flow', label:'当前流程', type:'select', options:['标准流程','+预算检查','+二次推荐','精简流程'], default:'+预算检查',
       guide:{title:'Workflow 流程微调', body:'标准流程：Recommend→Review。+预算检查：Recommend→BudgetCheck→Review（超预算重新推荐）。+二次推荐：Review 不通过则回到 Recommend。精简流程：跳过 Review 加速。', tip:'高客单场景务必启用预算检查；追求转化速度可用精简流程'}},
      {key:'max_retry', label:'最大重试', type:'select', options:['0','1','2','3'], default:'1',
       guide:{title:'最大重试微调', body:'Review 不通过时回到上游 Agent 重试的次数。0=不重试直接输出；1=重试1次；过多会增加延迟和成本。', tip:'建议 1-2 次，超过则转人工'}},
      {key:'parallel', label:'并行执行', type:'select', options:['OFF','ON'], default:'OFF',
       guide:{title:'并行执行微调', body:'Profile 和 NeedAnalysis 可并行执行以降低延迟。但并行后 State 合并复杂，调试难度增加。', tip:'延迟敏感场景可开启，需注意 State 字段冲突'}},
      {key:'human_handoff', label:'人工接管阈值', type:'select', options:['0.3','0.4','0.5','0.6'], default:'0.4',
       guide:{title:'人工接管阈值微调', body:'Review 评分低于此值时转人工。值越低越倾向自动处理，越高越倾向人工。医美高客单建议 0.4-0.5。', tip:'可按项目类别动态调整：手术类 0.5，轻医美 0.3'}},
    ]
  },
  review: {
    title: 'Review 审核优化',
    desc: '调整合规/预算/风险审核规则',
    params: [
      {key:'budget_tolerance', label:'预算容忍度', type:'select', options:['±5%','±10%','±15%','±20%'], default:'±10%',
       guide:{title:'预算容忍度微调', body:'推荐项目总价超出预算的比例上限。±10% 表示允许超 10%，超过则降分或重新推荐。', tip:'高客单项目建议 ±5% 严格管控，轻医美可放宽到 ±15%'}},
      {key:'compliance_level', label:'合规等级', type:'select', options:['strict','normal','loose'], default:'strict',
       guide:{title:'合规等级微调', body:'strict=启用全部违规词过滤+效果承诺拦截；normal=仅过滤明确违规词；loose=仅记录不拦截。医美行业建议 strict。', tip:'违规词库需定期更新，建议每月 review 一次'}},
      {key:'minor_block', label:'未成年拦截', type:'select', options:['ON','OFF'], default:'ON',
       guide:{title:'未成年拦截微调', body:'识别到客户未成年时强制拦截并提示需监护人签字。医美合规红线，强烈建议保持 ON。', tip:'可通过 Profile Agent 输出的 age 字段自动触发'}},
      {key:'pass_score', label:'通过评分阈值', type:'select', options:['50','60','70','80'], default:'60',
       guide:{title:'通过评分阈值微调', body:'Review 总分低于此值则不通过，触发重试或人工。60 是平衡点：过严影响转化，过松增加合规风险。', tip:'新上线建议 70 保守，稳定后降到 60'}},
    ]
  },
  model: {
    title: 'Model 模型优化',
    desc: '切换 LLM 模型与参数',
    params: [
      {key:'model', label:'LLM 模型', type:'select', options:['qwen-max','qwen-plus','qwen-turbo','deepseek-r1'], default:'qwen-max',
       guide:{title:'模型切换微调', body:'qwen-max：效果最好但贵慢，适合 Sales/Review；qwen-plus：性价比均衡，适合 Recommend；qwen-turbo：快且便宜，适合 Profile/NeedAnalysis；deepseek-r1：推理强，适合复杂审核。', tip:'按 Agent 重要度分配模型：核心用 max，辅助用 plus/turbo，可降本 40%+'}},
      {key:'max_tokens', label:'Max Tokens', type:'select', options:['128','256','512','1024'], default:'256',
       guide:{title:'Max Tokens 微调', body:'单次输出最大 token 数。话术生成建议 256-512；画像抽取 128 足够。过大会增加成本，过小会截断。', tip:'Sales Agent 话术建议 300-400 tokens'}},
      {key:'timeout', label:'超时(秒)', type:'select', options:['5','10','15','30'], default:'15',
       guide:{title:'超时微调', body:'LLM 调用超时时间。超时后降级到规则引擎或转人工。15 秒适合大多数场景。', tip:'网络抖动多时可提高到 30 秒，并加重试'}},
      {key:'fallback', label:'降级策略', type:'select', options:['规则引擎','转人工','缓存上次','报错'], default:'规则引擎',
       guide:{title:'降级策略微调', body:'LLM 不可用时的处理。规则引擎=走兜底逻辑；转人工=直接交顾问；缓存上次=用历史结果；报错=显式失败。', tip:'生产环境建议规则引擎兜底 + 告警，保证可用性'}},
    ]
  }
};

// 当前已应用的微调配置（从后端加载）
let appliedTuning = {};

function switchTuning(tab) {
  document.querySelectorAll('.tuning-tab').forEach(t => t.classList.remove('active'));
  document.querySelector('.tuning-tab[data-tab="'+tab+'"]').classList.add('active');
  const d = tuningData[tab];

  let html = '<div class="tuning-title">'+d.title+' <span class="desc">'+d.desc+'</span></div>';
  d.params.forEach((p, i) => {
    const curVal = appliedTuning[tab+'_'+p.key] || p.default;
    // input or select
    let inputHtml;
    if (p.type === 'select') {
      inputHtml = '<select class="t-input" data-tab="'+tab+'" data-key="'+p.key+'">' +
        p.options.map(o => '<option value="'+o+'"'+(o===curVal?' selected':'')+'>'+o+'</option>').join('') + '</select>';
    } else {
      inputHtml = '<input class="t-input" type="text" data-tab="'+tab+'" data-key="'+p.key+'" value="'+curVal+'">';
    }
    html += '<div class="tuning-row">' +
      '<span class="t-label">'+p.label+'</span>' +
      inputHtml +
      '<div class="t-help" onclick="toggleGuide(\''+tab+'\','+i+')" title="查看微调指引">?</div>' +
      '</div>';
    // guide（默认隐藏）
    html += '<div class="tuning-guide" id="guide_'+tab+'_'+i+'">' +
      '<div class="g-title">📖 '+p.guide.title+'</div>' +
      '<div>'+p.guide.body+'</div>' +
      '<div class="g-tip">💡 '+p.guide.tip+'</div>' +
      '</div>';
  });
  html += '<div class="tuning-saved" id="saved_'+tab+'">✓ 已保存，将应用于下次调用</div>';
  html += '<div class="tuning-actions">' +
    '<button class="tuning-btn reset" onclick="resetTuning(\''+tab+'\')">重置默认</button>' +
    '<button class="tuning-btn apply" onclick="applyTuning(\''+tab+'\')">应用微调</button>' +
    '</div>';
  document.getElementById('tuningContent').innerHTML = html;
}

function toggleGuide(tab, idx) {
  const el = document.getElementById('guide_'+tab+'_'+idx);
  const helpBtn = el.previousElementSibling.querySelector('.t-help');
  el.classList.toggle('show');
  if (helpBtn) helpBtn.classList.toggle('active');
}

async function applyTuning(tab) {
  const inputs = document.querySelectorAll('.t-input[data-tab="'+tab+'"]');
  const config = {};
  inputs.forEach(inp => {
    config[tab+'_'+inp.dataset.key] = inp.value;
  });
  try {
    await fetch('/api/tuning', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({tab: tab, config: config})
    });
    Object.assign(appliedTuning, config);
    const saved = document.getElementById('saved_'+tab);
    saved.classList.add('show');
    setTimeout(() => saved.classList.remove('show'), 2500);
  } catch(e) {
    alert('保存失败：'+e.message);
  }
}

function resetTuning(tab) {
  document.querySelectorAll('.t-input[data-tab="'+tab+'"]').forEach(inp => {
    const key = inp.dataset.key;
    const p = tuningData[tab].params.find(x => x.key === key);
    if (p) {
      if (inp.tagName === 'SELECT') {
        inp.value = p.default;
      } else {
        inp.value = p.default;
      }
    }
  });
}

// 初始化加载已保存的微调配置
async function loadTuning() {
  try {
    const resp = await fetch('/api/tuning');
    const data = await resp.json();
    appliedTuning = data.config || {};
  } catch(e) {}
  switchTuning('prompt');
}
loadTuning();

async function sendMessage() {
  const msg = document.getElementById('msgInput').value.trim();
  if (!msg) return;
  const btn = document.getElementById('sendBtn');
  const chatBox = document.getElementById('chatBox');
  chatBox.innerHTML += '<div class="chat-msg user"><div class="role">客户</div>'+msg+'</div>';
  chatBox.innerHTML += '<div class="chat-msg ai" id="thinking"><div class="role">AI</div>⏳ Agent 流水线执行中...</div>';
  chatBox.scrollTop = chatBox.scrollHeight;
  btn.disabled = true;

  // 重置 flow
  document.querySelectorAll('.flow-item').forEach(el => el.classList.remove('active', 'done'));

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({message: msg})
    });
    const data = await resp.json();
    document.getElementById('thinking').remove();
    renderResult(data, msg);
  } catch(e) {
    document.getElementById('thinking').innerHTML = '<div class="role">AI</div>❌ 错误: '+e.message;
  } finally {
    btn.disabled = false;
  }
}

function renderResult(d, userMsg) {
  const chatBox = document.getElementById('chatBox');
  const s = d.state || {};

  // 引擎标签
  document.getElementById('engineTag').textContent = '引擎: ' + (d.engine || 'unknown');

  // 逐步点亮 Agent Flow
  const trace = d.trace || [];
  trace.forEach((t, i) => {
    setTimeout(() => {
      const items = document.querySelectorAll('.flow-item');
      items.forEach(el => el.classList.remove('active'));
      const target = document.querySelector('.flow-item[data-agent="'+t.agent+'"]');
      if (target) {
        target.classList.add('active');
        // 之前的标记为 done
        items.forEach(el => {
          if (el !== target && !el.classList.contains('done')) {
            // 不处理
          }
        });
      }
      // 之前的标记 done
      for (let j = 0; j <= i; j++) {
        const agent = trace[j].agent;
        const el = document.querySelector('.flow-item[data-agent="'+agent+'"]');
        if (el && j < i) { el.classList.remove('active'); el.classList.add('done'); }
      }
    }, i * 200);
  });
  // 全部 done
  setTimeout(() => {
    document.querySelectorAll('.flow-item').forEach(el => { el.classList.remove('active'); el.classList.add('done'); });
  }, trace.length * 200 + 100);

  // State 视图
  const stateObj = {
    user_input: s.user_input, age: s.age, gender: s.gender, budget: s.budget,
    needs: s.needs, recommendations: s.recommendations,
    appointment_intent: s.appointment_intent, review_score: s.review_score,
    status: s.status
  };
  let stateHtml = '';
  for (const [k, v] of Object.entries(stateObj)) {
    const vs = JSON.stringify(v);
    stateHtml += '<div><span class="k">'+k+'</span>: <span class="s">'+vs+'</span></div>';
  }
  document.getElementById('stateView').innerHTML = stateHtml || '<span class="s">// 无数据</span>';

  // Trace
  const traceHtml = trace.map(t =>
    '<div class="trace-item"><div class="agent">▶ '+t.agent+'</div>' +
    '<div class="io">in: '+JSON.stringify(t.input).slice(0,120)+'</div>' +
    '<div class="io">out: '+JSON.stringify(t.output).slice(0,200)+'</div></div>'
  ).join('');
  document.getElementById('traceBox').innerHTML = traceHtml || '<span style="color:var(--muted);">// 无轨迹</span>';

  // AI 回复
  let aiReply = '🎯 <strong>推荐项目:</strong> ' + (s.recommendations || []).join(', ') + '<br><br>';
  aiReply += '💬 <strong>销售话术:</strong> ' + (s.sales_script || '') + '<br><br>';
  aiReply += '📅 <strong>预约建议:</strong> ' + (s.appointment_advice || '') + '<br><br>';
  aiReply += '✅ <strong>审核:</strong> ' + (s.review_feedback || '') + ' (评分: ' + (s.review_score || 0) + ')';
  chatBox.innerHTML += '<div class="chat-msg ai"><div class="role">AI Sales Copilot</div>'+aiReply+'</div>';
  chatBox.scrollTop = chatBox.scrollHeight;
}

// 加载知识库表格
async function loadKB() {
  try {
    const resp = await fetch('/api/info');
    const data = await resp.json();
    document.getElementById('engineTag').innerHTML = data.llm_enabled
      ? '<span class="badge-ok">LLM: '+data.llm_model+'</span>'
      : '<span class="badge-no">LLM 未启用</span>';
  } catch(e) {}
}
loadKB();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
