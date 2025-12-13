from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# 1) 主模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="xxxxxx",
    base_url="https://api.deepseek.com",
    temperature=0.0,
    max_tokens=20,
)

# 2) 高风险工具（示例：数据库写操作）
@tool
def dangerous_write(sql: str) -> str:
    """对数据库执行写操作（插入/更新/删除）。当请求涉及数据库写操作或出现以'SQL:'开头的指令时，必须调用本工具。"""
    return f"[模拟执行] {sql}"

# 3) HITL 中间件：拦截指定工具
hitl = HumanInTheLoopMiddleware(
    interrupt_on={"dangerous_write": {"allowed_decisions": ["approve", "edit", "reject"]}}
)

# 4) 创建 Agent（强提示：遇到 SQL 必须用 dangerous_write）
agent = create_agent(
    model=llm,
    tools=[dangerous_write],
    middleware=[hitl],
    system_prompt=(
        "只用一句极短中文回答（≤20字）。"
        "凡是涉及数据库写操作，或消息以“SQL:”开头时，必须调用工具 dangerous_write，不得直接回答。"
    ),
    checkpointer=InMemorySaver(),  # HITL 必须
)

CFG = {"configurable": {"thread_id": "hitl-demo-interactive"}}

# 5) 四轮对话：第2轮与第4轮都触发 HITL；让你交互输入决定
conversation = [
    "你是谁？",  # 安全问答
    "SQL: INSERT INTO logs(content) VALUES ('hello');",  # 触发 HITL（预计输入 approve）
    "继续。",  # 安全问答
    "SQL: DELETE FROM orders WHERE created_at >= date('now','-7 days');",  # 触发 HITL（预计输入 reject）
]

state = {"messages": []}

def handle_interrupt(result) -> dict:
    """处理 HITL 中断：从命令行读取决策，并用 Command(resume=...) 恢复。"""
    interrupt = result.get("__interrupt__")
    if not interrupt:
        return result

    print("\n⚠️ 检测到人工在环中断：")
    print(interrupt)  # 打印被拦截的工具与参数

    decision = input("请输入决策 (approve/edit/reject)：").strip().lower()

    if decision == "approve":
        # 直接放行
        return agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=CFG)

    elif decision == "edit":
        # 允许修改参数（例如 SQL），再继续
        new_sql = input("请输入修改后的 SQL：").strip()
        return agent.invoke(
            Command(
                resume={
                    "decisions": [{
                        "type": "edit",
                        "updates": {"sql": new_sql}
                    }]
                }
            ),
            config=CFG
        )

    elif decision == "reject":
        # 拒绝执行，并给出替代返回（不执行工具）
        return agent.invoke(
            Command(
                resume={
                    "decisions": [{
                        "type": "reject",
                        "override": {"content": "[操作已被人工拒绝]"}
                    }]
                }
            ),
            config=CFG
        )
    else:
        print("输入无效，按拒绝处理。")
        return agent.invoke(
            Command(
                resume={
                    "decisions": [{
                        "type": "reject",
                        "override": {"content": "[操作已被人工拒绝]"}
                    }]
                }
            ),
            config=CFG
        )

for i, q in enumerate(conversation, 1):
    user_msg = HumanMessage(content=q)
    result = agent.invoke({"messages": state["messages"] + [user_msg]}, config=CFG)

    # 命中中断：让你做决策 → 再 resume
    if result.get("__interrupt__"):
        result = handle_interrupt(result)

    state = result
    ans = state["messages"][-1].content

    print(f"\n🧩 第 {i} 轮")
    print(f"Q: {q}")
    print(f"A: {ans}")

print("\n✅ 对话结束。")
