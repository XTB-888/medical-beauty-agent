"""
AI Medical Beauty Sales Copilot — LangGraph Agent 实现

架构: Supervisor → Profile → NeedAnalysis → Recommend → Sales → Appointment → Review → END
LLM:   Qwen-Max (阿里云百炼 DashScope OpenAI 兼容接口)
"""
from __future__ import annotations

import json
import os
from typing import TypedDict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


# ────────────────────────────────────────────────────────────
# State 定义（对应 PRD 第 8 节）
# ────────────────────────────────────────────────────────────
class MedicalBeautyState(TypedDict, total=False):
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
    # 执行轨迹（供前端展示）
    trace: list


# ────────────────────────────────────────────────────────────
# LLM 工厂（DashScope OpenAI 兼容）
# ────────────────────────────────────────────────────────────
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-max")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")


def make_llm(temperature: float = 0.3) -> ChatOpenAI | None:
    if not DASHSCOPE_API_KEY:
        return None
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=temperature,
    )


LLM = make_llm()


# ────────────────────────────────────────────────────────────
# 医美知识库（用于 Recommend / Sales Agent）
# ────────────────────────────────────────────────────────────
PROJECT_KB: dict[str, dict[str, Any]] = {
    "超声炮": {
        "category": "抗衰",
        "price_range": "8000-15000",
        "suitable_age": "30-50",
        "effect": "提拉紧致、改善松弛",
        "selling_point": "无创、恢复快、效果立竿见影",
    },
    "热玛吉": {
        "category": "抗衰",
        "price_range": "12000-25000",
        "suitable_age": "35-55",
        "effect": "胶原再生、紧致肌肤",
        "selling_point": "FDA认证、效果持久2-3年",
    },
    "黄金微针": {
        "category": "抗衰/祛斑",
        "price_range": "3000-8000",
        "suitable_age": "25-45",
        "effect": "改善痘坑、淡化色斑、紧致",
        "selling_point": "性价比高、创伤小",
    },
    "光子嫩肤": {
        "category": "祛斑",
        "price_range": "1000-3000",
        "suitable_age": "20-50",
        "effect": "提亮肤色、淡化色斑",
        "selling_point": "无恢复期、可日常",
    },
    "双眼皮手术": {
        "category": "塑形",
        "price_range": "5000-15000",
        "suitable_age": "18-45",
        "effect": "眼部塑形",
        "selling_point": "成熟术式、个性化设计",
    },
}


# ────────────────────────────────────────────────────────────
# 7 个 Agent 节点实现
# ────────────────────────────────────────────────────────────
def _append_trace(state: MedicalBeautyState, agent: str, input_data: Any, output_data: Any) -> list:
    trace = state.get("trace", [])
    trace.append({
        "agent": agent,
        "input": input_data if isinstance(input_data, (str, dict, list)) else str(input_data),
        "output": output_data if isinstance(output_data, (str, dict, list)) else str(output_data),
    })
    return trace


def supervisor_node(state: MedicalBeautyState) -> dict:
    """调度中心：初始化流程"""
    trace = _append_trace(state, "Supervisor", state.get("user_input", ""), "流程启动 → Profile")
    return {"status": "profiling", "trace": trace, "retry_count": 0}


def profile_node(state: MedicalBeautyState) -> dict:
    """Profile Agent：抽取客户画像（年龄/性别/预算）"""
    user_input = state.get("user_input", "")

    if LLM is not None:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是医美客户画像抽取助手。从用户对话中抽取 age(整数), gender(male/female), budget(整数)。"
                       "严格输出 JSON，不要 markdown 代码块。示例: {{\"age\":34,\"gender\":\"female\",\"budget\":15000}}。无法判断的字段填 null。"),
            ("human", "{input}"),
        ])
        try:
            raw = (prompt | LLM | StrOutputParser()).invoke({"input": user_input})
            text = raw.strip().strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
            data = json.loads(text)
            age = int(data.get("age") or 0) if data.get("age") else 0
            gender = data.get("gender") or "unknown"
            budget = int(data.get("budget") or 0) if data.get("budget") else 0
        except Exception:
            age, gender, budget = 0, "unknown", 0
    else:
        # 规则兜底
        age, gender, budget = 0, "female", 0
        for kw in ["男"]:
            if kw in user_input:
                gender = "male"
        import re
        m = re.search(r"(\d+)\s*岁", user_input)
        if m:
            age = int(m.group(1))
        m = re.search(r"预算\s*(\d+)", user_input)
        if m:
            budget = int(m.group(1))

    result = {"age": age, "gender": gender, "budget": budget}
    trace = _append_trace(state, "Profile", user_input, result)
    return {**result, "status": "analyzing_need", "trace": trace}


def need_analysis_node(state: MedicalBeautyState) -> dict:
    """Need Analysis Agent：识别客户需求（抗衰/祛斑/塑形/双眼皮）"""
    user_input = state.get("user_input", "")
    categories = ["抗衰", "祛斑", "塑形", "双眼皮"]

    if LLM is not None:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是医美需求识别助手。从用户输入识别需求类别，可多选。"
                       "候选类别: 抗衰、祛斑、塑形、双眼皮。"
                       "严格输出 JSON: {{\"needs\":[\"抗衰\"]}}。不要 markdown。"),
            ("human", "{input}"),
        ])
        try:
            raw = (prompt | LLM | StrOutputParser()).invoke({"input": user_input})
            text = raw.strip().strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
            data = json.loads(text)
            needs = data.get("needs") or []
        except Exception:
            needs = [c for c in categories if c in user_input] or ["抗衰"]
    else:
        needs = [c for c in categories if c in user_input] or ["抗衰"]

    trace = _append_trace(state, "NeedAnalysis", user_input, {"needs": needs})
    return {"needs": needs, "status": "recommending", "trace": trace}


def recommend_node(state: MedicalBeautyState) -> dict:
    """Recommend Agent：结合知识库推荐项目"""
    needs = state.get("needs", [])
    budget = state.get("budget", 0)
    age = state.get("age", 0)

    # 知识库匹配
    candidates = []
    for proj_name, info in PROJECT_KB.items():
        if any(cat in info["category"] for cat in needs):
            candidates.append((proj_name, info))

    # 预算过滤
    if budget > 0:
        filtered = []
        for name, info in candidates:
            low, high = (int(x) for x in info["price_range"].split("-"))
            if low <= budget:
                filtered.append((name, info))
        candidates = filtered or candidates

    # 取前 2 个
    recommendations = [name for name, _ in candidates[:2]]
    rec_detail = [{"name": n, "info": i} for n, i in candidates[:2]]

    trace = _append_trace(state, "Recommend", {"needs": needs, "budget": budget}, rec_detail)
    return {"recommendations": recommendations, "status": "selling", "trace": trace}


def sales_node(state: MedicalBeautyState) -> dict:
    """Sales Agent：生成销售话术"""
    recs = state.get("recommendations", [])
    age = state.get("age", 0)
    budget = state.get("budget", 0)

    rec_info = "\n".join([f"- {n}: {PROJECT_KB.get(n, {}).get('selling_point', '')}" for n in recs])

    if LLM is not None:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是资深医美销售顾问。基于推荐项目，生成不超过 150 字的销售话术，要包含推荐理由和案例引用。"
                       "语气亲切专业，符合医美合规。"),
            ("human", "客户画像: 年龄{age}, 预算{budget}\n推荐项目:\n{rec_info}\n\n请生成销售话术。"),
        ])
        try:
            script = (prompt | LLM | StrOutputParser()).invoke({
                "age": age, "budget": budget, "rec_info": rec_info
            })
        except Exception:
            script = f"为您推荐 {', '.join(recs)}，这些项目非常适合您的年龄和需求，性价比高。"
    else:
        script = f"为您推荐 {', '.join(recs)}，{rec_info}，非常适合您的情况。"

    trace = _append_trace(state, "Sales", {"recommendations": recs}, script)
    return {"sales_script": script, "status": "appointing", "trace": trace}


def appointment_node(state: MedicalBeautyState) -> dict:
    """Appointment Agent：生成预约建议"""
    from datetime import datetime, timedelta
    # 默认本周六下午
    today = datetime.now()
    days_ahead = (5 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    sat = today + timedelta(days=days_ahead)
    advice = f"{sat.strftime('%m月%d日')}（本周六）下午 14:00-16:00 到院面诊，由资深咨询师一对一接待"

    trace = _append_trace(state, "Appointment", None, advice)
    return {"appointment_intent": True, "appointment_advice": advice, "status": "reviewing", "trace": trace}


def review_node(state: MedicalBeautyState) -> dict:
    """Review Agent：预算/合规/风险审核"""
    budget = state.get("budget", 0)
    recs = state.get("recommendations", [])

    issues = []
    score = 100.0

    # 预算审核
    if budget > 0 and recs:
        for r in recs:
            info = PROJECT_KB.get(r, {})
            if info:
                low, high = (int(x) for x in info["price_range"].split("-"))
                if high > budget:
                    issues.append(f"{r} 高端价位 {high} 超预算 {budget}，需提示分期")
                    score -= 10

    # 合规审核
    if "保证效果" in state.get("sales_script", "") or "百分百" in state.get("sales_script", ""):
        issues.append("话术含违规承诺词，需修正")
        score -= 20

    # 风险审核
    if state.get("age", 0) and state["age"] < 18:
        issues.append("未成年人需监护人签字")
        score -= 30

    score = max(score, 0)
    feedback = "通过" if not issues else "需关注：" + "; ".join(issues)

    result = {"score": score, "feedback": feedback, "issues": issues}
    trace = _append_trace(state, "Review", {"budget": budget, "recs": recs}, result)
    return {"review_score": score, "review_feedback": feedback, "status": "completed", "trace": trace}


# ────────────────────────────────────────────────────────────
# 构建 LangGraph
# ────────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(MedicalBeautyState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("profile", profile_node)
    g.add_node("need_analysis", need_analysis_node)
    g.add_node("recommend", recommend_node)
    g.add_node("sales", sales_node)
    g.add_node("appointment", appointment_node)
    g.add_node("review", review_node)

    g.set_entry_point("supervisor")
    g.add_edge("supervisor", "profile")
    g.add_edge("profile", "need_analysis")
    g.add_edge("need_analysis", "recommend")
    g.add_edge("recommend", "sales")
    g.add_edge("sales", "appointment")
    g.add_edge("appointment", "review")
    g.add_edge("review", END)

    return g.compile()


# 全局 graph 实例
GRAPH = build_graph()


def run_pipeline(user_input: str) -> dict:
    """运行完整销售 SOP Pipeline"""
    init_state: MedicalBeautyState = {"user_input": user_input, "trace": []}
    final = GRAPH.invoke(init_state)
    return final
