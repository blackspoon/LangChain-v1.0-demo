from typing import Any, Callable
import time, math

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware, AgentState, ModelRequest, ModelResponse,
)
from langchain.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from langchain_openai import ChatOpenAI

# ========== 中间件 A：日志 + 安全拦截（Node-style） ==========
class PolicyGuardMiddleware(AgentMiddleware):
    """在关键节点打印日志，并在 after_model 命中关键词时跳转 end"""

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"[before_agent] 会话开始，消息数：{len(state.get('messages', []))}")
        return None

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"[before_model] 准备进行模型调用，当前消息数：{len(state['messages'])}")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and "BLOCKED_CN" in (last.content or ""):
            print("[after_model] 触发安全规则：检测到 BLOCKED_CN，跳转 end")
            return {
                "messages": [AIMessage("该请求触发了安全校验，无法继续。")],
                "jump_to": "end",
            }
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"[after_agent] 会话结束，最终消息数：{len(state.get('messages', []))}")
        return None


# ========== 中间件 B：重试 + 耗时统计（Wrap-style） ==========
class RetryAndMetricsMiddleware(AgentMiddleware):
    """包裹每次模型调用，做退避重试与耗时统计"""

    def __init__(self, max_retries: int = 2, base_delay: float = 0.1):
        self.max_retries = max_retries
        self.base_delay = base_delay  # 秒

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        start = time.time()
        try:
            for i in range(self.max_retries + 1):
                try:
                    return handler(request)
                except Exception as e:
                    if i == self.max_retries:
                        raise
                    backoff = self.base_delay * math.pow(2, i)
                    print(f"[wrap_model_call] 失败，将在 {backoff:.2f}s 后重试：{e}")
                    time.sleep(backoff)
        finally:
            cost = (time.time() - start) * 1000
            print(f"[wrap_model_call] 本次模型调用耗时：{cost:.0f} ms")


# ========== 组装 Agent（DeepSeek 模型） ==========
llm = ChatOpenAI(
    model="deepseek-chat", 
    api_key="xxxxxx",
    base_url="https://api.deepseek.com",
    temperature=0.3
)

# 注意执行顺序：
# - wrap 型按列表“从外到内”包裹；我们希望“重试/计时”最外层，所以把 RetryAndMetricsMiddleware放前面
# - node 型按列表顺序执行 before_*，after_* 逆序回卷
agent = create_agent(
    model=llm,
    tools=[],
    middleware=[
        RetryAndMetricsMiddleware(max_retries=2, base_delay=0.1),  # 外层：重试+计时
        PolicyGuardMiddleware(),                                   # 内层：日志+安全拦截
    ],
)

if __name__ == "__main__":
    # 1) 正常问答
    res1 = agent.invoke({"messages": [HumanMessage("用一句话解释 LangGraph 是什么。")]})
    print("\n[Result-1]", res1["messages"][-1].content)

    print("\n" + "-" * 64 + "\n")

    # 2) 触发安全跳转（让模型回复包含关键字）
    res2 = agent.invoke({"messages": [HumanMessage("请只回复：BLOCKED_CN")]})
    print("\n[Result-2]", res2["messages"][-1].content)
