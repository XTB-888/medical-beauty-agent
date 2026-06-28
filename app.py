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

  /* Tuning 面板 */
  .tuning-tabs{display:flex;gap:0.4rem;margin-bottom:0.6rem;flex-wrap:wrap;}
  .tuning-tab{padding:0.35rem 0.7rem;background:var(--bg3);border:1px solid var(--rule);
    border-radius:6px;cursor:pointer;font-size:0.78rem;color:var(--muted);}
  .tuning-tab.active{background:var(--purple);color:white;border-color:var(--purple);}
  .tuning-content{background:var(--bg3);border-radius:8px;padding:0.8rem;font-size:0.8rem;min-height:120px;}
  .tuning-content .row{display:flex;justify-content:space-between;padding:0.35rem 0;border-bottom:1px dashed var(--rule);}
  .tuning-content .row:last-child{border-bottom:none;}
  .tuning-content .label{color:var(--muted);}
  .tuning-content .value{color:#fcd34d;font-family:monospace;}
  .tuning-content .ver{font-size:0.72rem;color:var(--green);margin-left:0.4rem;}

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
            <div class="panel-title">🔧 Agent 微调面板（核心亮点）</div>
            <div class="tuning-tabs">
              <div class="tuning-tab active" data-tab="recommend" onclick="switchTuning('recommend')">Recommend</div>
              <div class="tuning-tab" data-tab="sales" onclick="switchTuning('sales')">Sales</div>
              <div class="tuning-tab" data-tab="review" onclick="switchTuning('review')">Review</div>
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
const tuningData = {
  recommend: {
    title: 'Recommend Agent 微调',
    rows: [
      {label: 'Prompt 版本', value: 'v1.2 <span class="ver">+预算约束</span>'},
      {label: 'Chunk Size', value: '800'},
      {label: 'Top K', value: '5'},
      {label: 'Rerank', value: 'ON'},
      {label: 'Workflow', value: 'Recommend → BudgetCheck → Review'},
    ]
  },
  sales: {
    title: 'Sales Agent 微调',
    rows: [
      {label: 'Prompt 版本', value: 'v2.0 <span class="ver">+案例引用</span>'},
      {label: 'Temperature', value: '0.7'},
      {label: 'Max Tokens', value: '300'},
      {label: '合规过滤', value: 'strict'},
      {label: '异议处理库', value: 'enabled'},
    ]
  },
  review: {
    title: 'Review Agent 微调',
    rows: [
      {label: '审核维度', value: '预算/合规/风险'},
      {label: '预算阈值', value: '±10%'},
      {label: '违规词库', value: '128 条'},
      {label: '未成年拦截', value: 'ON'},
      {label: '风险评分阈值', value: '≥60 通过'},
    ]
  }
};

function switchTuning(tab) {
  document.querySelectorAll('.tuning-tab').forEach(t => t.classList.remove('active'));
  document.querySelector('.tuning-tab[data-tab="'+tab+'"]').classList.add('active');
  const d = tuningData[tab];
  const html = '<div style="color:var(--purple);font-weight:600;margin-bottom:0.5rem;font-size:0.85rem;">'+d.title+'</div>' +
    d.rows.map(r => '<div class="row"><span class="label">'+r.label+'</span><span class="value">'+r.value+'</span></div>').join('');
  document.getElementById('tuningContent').innerHTML = html;
}
switchTuning('recommend');

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
