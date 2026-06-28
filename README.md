# AI Medical Beauty Sales Copilot

> 医美销售流程数字化、标准化、智能化 —— 用 **LangGraph** 编排 7 个 Agent，把销售 SOP 变成可观测、可调优的智能流水线。
> 接入**阿里云百炼 Qwen-Max**，配套 RAG 知识库，前端实时展示 Agent 执行轨迹与 State 变化，**5 大维度微调面板支持在线编辑 + 微调指引**。

---

## 核心特性

- 🤖 **7 Agent 流水线**：Supervisor / Profile / NeedAnalysis / Recommend / Sales / Appointment / Review，LangGraph 编排
- 🧠 **Qwen-Max 接入**：阿里云百炼 DashScope OpenAI 兼容接口，真实 LLM 调用
- 📊 **四面板可视化**：Chat / Agent Flow / State / Tuning，执行轨迹实时可见
- 🔧 **微调面板（核心亮点）**：5 大维度（Prompt / Knowledge / Workflow / Review / Model）可在线编辑，每个参数配微调指引
- 📚 **RAG 知识库**：内置医美项目库（超声炮/热玛吉/黄金微针等），支持 Chunk/TopK/Rerank 调优
- 🔄 **配置持久化**：微调配置保存到后端，变更历史可追溯，下次调用生效

---

## 一、产品定位

**不是聊天机器人，而是 AI 销售 SOP 执行引擎。**

| 角色 | 价值 |
|------|------|
| 销售 | 提升转化能力，统一话术标准 |
| 主管 | 监控销售数据，统一管理标准 |
| 机构 | 沉淀销售经验，数据驱动优化 |
| 运营 | 持续调优 Agent，闭环迭代 |

## 二、行业痛点

- 销售能力依赖个人：新人培训周期长、话术不统一
- 流程缺乏标准化：微信→客服→咨询师→预约→到院→成交，执行标准不一
- 客户流失严重：回复不及时、推荐不精准、跟进不连续
- 数据无法沉淀：无法回答"哪环节流失最多"

## 三、Agent 架构

7 个 Agent 由 LangGraph 编排，Supervisor 调度，串行执行 SOP：

```
User → Supervisor → Profile → NeedAnalysis → Recommend → Sales → Appointment → Review → END
                       ↓                        ↓
                    画像抽取                 项目知识库(RAG)
```

| Agent | 职责 |
|-------|------|
| **Supervisor** | 流程调度中心，状态判断与流程推进 |
| **Profile** | 从对话抽取客户画像（年龄/性别/预算） |
| **NeedAnalysis** | 识别需求类别（抗衰/祛斑/塑形/双眼皮） |
| **Recommend** | 结合知识库推荐项目，按预算过滤 |
| **Sales** | 生成销售话术（推荐理由+案例+异议处理） |
| **Appointment** | 生成预约建议（推荐到院时间） |
| **Review** | 预算/合规/风险审核，输出评分 |

## 四、LangGraph State

所有 Agent 共享一个 State，每个节点读写自己负责的字段：

```python
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
    trace: list  # 执行轨迹，供前端可视化
```

## 五、技术栈

| 层 | 技术 |
|----|------|
| 前端 | 内嵌 HTML（Chat + Agent Flow + State + Tuning 四面板） |
| 后端 | FastAPI |
| 编排 | **LangGraph** StateGraph |
| LLM | **Qwen-Max**（阿里云百炼 DashScope OpenAI 兼容接口） |
| 知识库 | 内置医美项目知识库（生产环境用 ChromaDB） |

## 六、快速开始

```bash
cd medical-beauty-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 配置阿里云百炼 API Key
cp .env.example .env
export DASHSCOPE_API_KEY=sk-xxxxx

python app.py
```

## 七、API 概览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 介绍页 + 在线 Demo |
| GET | `/api/info` | Agent 元信息 |
| POST | `/api/chat` | 运行完整 SOP Pipeline |
| GET | `/api/health` | 健康检查 + LLM 状态 |
| GET | `/api/tuning` | 获取当前 Agent 微调配置 |
| POST | `/api/tuning` | 保存微调配置（含变更历史） |
| DELETE | `/api/tuning` | 重置所有微调配置 |

### 对话示例

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"我34岁，预算15000，想做抗衰"}'
```

响应（含 7 个 Agent 的执行轨迹）：

```json
{
  "success": true,
  "state": {
    "age": 34, "gender": "female", "budget": 15000,
    "needs": ["抗衰"],
    "recommendations": ["超声炮", "黄金微针"],
    "sales_script": "为您推荐超声炮...",
    "appointment_advice": "本周六下午 14:00-16:00 到院面诊",
    "review_score": 90.0,
    "review_feedback": "通过"
  },
  "trace": [
    {"agent": "Supervisor", "input": "...", "output": "..."},
    {"agent": "Profile", "input": "...", "output": {"age":34, ...}},
    ...
  ],
  "engine": "langgraph-qwen-max"
}
```

### 微调配置示例

```bash
# 保存微调配置
curl -X POST http://localhost:8000/api/tuning \
  -H "Content-Type: application/json" \
  -d '{"tab":"prompt","config":{"prompt_version":"v2.0","prompt_temp":"0.7"}}'
```

响应（含变更历史）：

```json
{
  "success": true,
  "tab": "prompt",
  "applied": {"prompt_version": "v2.0", "prompt_temp": "0.7"},
  "changes": [
    {"key": "prompt_version", "old": null, "new": "v2.0"},
    {"key": "prompt_temp", "old": null, "new": "0.7"}
  ],
  "total_configs": 2,
  "message": "已保存 2 项配置，将在下次调用生效"
}
```

## 八、前端亮点

四面板实时可视化：

1. **Chat 面板**：客户对话 + AI 回复（推荐项目/话术/预约/审核）
2. **Agent Flow 面板**：7 个 Agent 逐步点亮，可视化执行进度
3. **State 面板**：实时显示 LangGraph State 的字段变化
4. **Agent 微调面板（核心亮点）**：5 大维度可在线编辑 + 微调指引

### 微调面板详细说明

微调面板从静态展示升级为**完全可编辑的交互式调优台**，支持 5 个 Tab：

| Tab | 可调参数 | 说明 |
|-----|---------|------|
| **Prompt** | 版本 / 角色 / 约束 / Temperature | 调整 Prompt 风格与约束，支持版本管理 |
| **Knowledge** | Chunk Size / Top K / Rerank / 相似度阈值 / 知识库来源 | RAG 检索参数调优 |
| **Workflow** | 流程模式 / 最大重试 / 并行执行 / 人工接管阈值 | Agent 执行流程编排 |
| **Review** | 预算容忍度 / 合规等级 / 未成年拦截 / 通过评分阈值 | 审核规则配置 |
| **Model** | LLM 模型 / Max Tokens / 超时 / 降级策略 | 模型与运行参数 |

**核心交互**：
- 每个参数支持 `select` 下拉或 `input` 文本编辑
- 每个参数右侧 `?` 图标，**点击展开微调指引**（含参数作用、影响、经验值、优化方向提示）
- 「应用微调」按钮保存到后端，下次 chat 调用生效，显示变更历史
- 「重置默认」一键恢复
- 配置持久化：刷新页面后从后端加载已保存配置

## 九、知识库

内置医美项目知识库（演示版）：

| 项目 | 类别 | 价格区间 | 适龄 | 卖点 |
|------|------|---------|------|------|
| 超声炮 | 抗衰 | 8000-15000 | 30-50 | 无创、恢复快 |
| 热玛吉 | 抗衰 | 12000-25000 | 35-55 | FDA认证、持久2-3年 |
| 黄金微针 | 抗衰/祛斑 | 3000-8000 | 25-45 | 性价比高、创伤小 |
| 光子嫩肤 | 祛斑 | 1000-3000 | 20-50 | 无恢复期 |
| 双眼皮手术 | 塑形 | 5000-15000 | 18-45 | 成熟术式 |

## 十、Agent 优化闭环

```
线上数据 → 发现问题 → 定位 Agent → Prompt/RAG 优化 → AB 测试 → 上线 → 持续监控
```

## 十一、项目结构

```
medical-beauty-agent/
├── agent.py            # LangGraph 实现（7 Agent + State + Qwen-Max）
├── app.py              # FastAPI + 介绍页 HTML
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

## License

MIT
