from typing import Any, Callable
import time
import math

from langchain.agents import create_agent
from langchain.agents.middleware import (
    before_agent,
    after_agent,
    before_model,
    after_model,
    wrap_model_call,
    dynamic_prompt,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from langchain_openai import ChatOpenAI


# ========= 便捷装饰器：动态系统提示 =========
@dynamic_prompt
def personalized_prompt(req: ModelRequest) -> str:
    runtime = getattr(req, "runtime", None)
    user_id = "访客"
    if runtime and getattr(runtime, "context", None):
        user_id = runtime.context.get("user_id", "访客")
    return f"你是一名贴心的中文助手，正在为用户「{user_id}」提供帮助。回答时要简洁、自然。"

# ========= 节点式：Agent 级别前置/后置（一次调用各触发一次） =========
@before_agent
def log_before_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"[before_agent] 本次会话开始，已有消息数：{len(state.get('messages', []))}")
    return None

@after_agent
def log_after_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"[after_agent] 会话结束，最终消息数：{len(state.get('messages', []))}")
    return None


# ========= 节点式：模型调用前/后（每次模型调用都会触发） =========
@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"[before_model] 准备进行模型调用，当前消息数：{len(state['messages'])}")
    return None

@after_model(can_jump_to=["end"])
def validate_output(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """简单的输出校验：若模型输出包含“BLOCKED_CN”，则改写消息并跳转到 end。"""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and "BLOCKED_CN" in (last.content or ""):
        print("[after_model] 触发安全规则：检测到 BLOCKED_CN，跳转到 end")
        return {
            "messages": [AIMessage("该请求触发了安全校验，无法继续。")],
            "jump_to": "end",
        }
    return None

# ========= 包裹式：为模型调用加重试与耗时统计 =========
@wrap_model_call
def retry_and_timing(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    max_retries = 2
    start = time.time()
    try:
        for i in range(max_retries + 1):
            try:
                return handler(request)
            except Exception as e:
                if i == max_retries:
                    raise
                # 退避等待（100ms, 200ms）
                backoff = 0.1 * math.pow(2, i)
                print(f"[wrap_model_call] 调用失败，将在 {backoff:.2f}s 后重试：{e}")
                time.sleep(backoff)
    finally:
        cost = (time.time() - start) * 1000
        print(f"[wrap_model_call] 本次模型调用耗时：{cost:.0f} ms")

# ========= 组装 Agent（DeepSeek 模型） =========
# 你也可以通过环境变量配置：OPENAI_API_KEY/DEEPSEEK_API_KEY
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="xxxxxx",
    base_url="https://api.deepseek.com",
    temperature=0.3,
)

agent = create_agent(
    model=llm,
    tools=[],  # 如需可加工具，此处保持最小化
    middleware=[
        personalized_prompt,   # 便捷：动态系统提示
        log_before_agent,      # 节点：Agent 级前置
        log_before_model,      # 节点：模型前
        retry_and_timing,      # 包裹：重试与计时
        validate_output,       # 节点：模型后（含 jump_to）
        log_after_agent,       # 节点：Agent 级后置
    ],
)

# ========= 最小演示 =========
if __name__ == "__main__":
    # 1) 正常问答（不会触发安全跳转）
    res1 = agent.invoke(
        {"messages": [HumanMessage("用一句话解释 LangGraph 是什么。")]},
        config={"context": {"user_id": "alice"}},
    )
    print("\n[Result-1]", res1["messages"][-1].content)
    
    print("--------------------------------")

    # 2) 触发 after_model 的阻断（让模型输出包含关键字）
    res2 = agent.invoke(
        {"messages": [HumanMessage("请只回复：BLOCKED_CN")]},
        config={"context": {"user_id": "bob"}},
    )
    print("\n[Result-2]", res2["messages"][-1].content)
