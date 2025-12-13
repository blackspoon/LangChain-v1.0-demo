"""
中间件总装配（覆盖三类形态）：
1) 预置（Built-in）中间件：直接复用官方提供的能力（示例：SummarizationMiddleware）
2) 基于“装饰器”的中间件：使用 before_agent / after_agent / before_model / after_model /
   wrap_model_call / dynamic_prompt 六种钩子组合，轻量、函数式、低侵入
3) 基于“类”的中间件：继承 AgentMiddleware，集中实现多个阶段的治理逻辑，适合复杂策略

参考文档：
- Built-in middleware:  https://docs.langchain.com/oss/python/langchain/middleware#built-in-middleware
- Decorator-based:      https://docs.langchain.com/oss/python/langchain/middleware#decorator-based-middleware
- Class-based:          https://docs.langchain.com/oss/python/langchain/middleware#class-based-middleware
"""
from __future__ import annotations

import time
from typing import Optional, Awaitable, Callable

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import (
    before_agent,
    after_agent,
    before_model,
    after_model,
    dynamic_prompt,
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    SummarizationMiddleware,
)

# =========================================================
# 1) 预置（Built-in）中间件
# =========================================================
def install_builtin_middlewares():
    """
    官方内置中间件：自动摘要对话上下文
    使用独立的 summarization 模型，并设置触发/保留策略。
    """
    print("[中间件] 开始安装官方内置 SummarizationMiddleware。")
    summ_llm = ChatOpenAI(
        model="deepseek-chat",
        api_key="xxxxxx",
        base_url="https://api.deepseek.com",
        temperature=0.0,
    )

    summarizer = SummarizationMiddleware(
        model=summ_llm,
        max_tokens_before_summary=3000,
        messages_to_keep=20,
    )
    print("[中间件] 官方内置 SummarizationMiddleware 装载完成。")
    return [summarizer]


# =========================================================
# 2) 装饰器（Decorator-based）中间件
# =========================================================

# 2.1 限频（在 Agent 入场前）
def _rate_limit_guard(state: AgentState) -> Optional[dict]:
    now = time.time()
    last = state.get("_last_ts", 0.0)
    print(f"[中间件] rate_guard 触发检查：now={now}, last={last}")
    if now - last < 1.0:
        msg = AIMessage(content="请求太快啦，请 1 秒后再试。")
        print("[中间件] rate_guard 生效：拒绝请求并终止流程。")
        return {"messages": state.get("messages", []) + [msg], "jump_to": "end"}
    state["_last_ts"] = now
    print("[中间件] rate_guard 放行：更新最近访问时间。")
    return None


@before_agent(name="rate_guard")
def rate_guard(state: AgentState, *_args, **_kw) -> Optional[dict]:
    return _rate_limit_guard(state)

# 2.2 在模型前做参数澄清引导（天气缺城市 / 数学缺表达式）
def _clarify_intent_guard(state: AgentState) -> Optional[dict]:
    msgs = state.get("messages", [])
    last = msgs[-1] if msgs else None
    if not isinstance(last, HumanMessage):
        print("[中间件] clarify_missing_params 略过：最后一条消息不是用户。")
        return None

    text = (last.content or "").strip()
    print(f"[中间件] clarify_missing_params 检查文本：{text}")
    need_weather = any(k in text for k in ["天气", "温度", "weather"])
    has_city = any(k in text for k in ["北京", "上海", "广州", "深圳", "杭州", "新加坡", "Singapore"])
    if need_weather and not has_city:
        print("[中间件] clarify_missing_params 提醒：天气查询缺少城市。")
        return {"system_prompt": "若用户查询天气但未提供城市，请先礼貌询问城市名称，仅提出一个澄清问题。"}

    need_math = any(k in text for k in ["计算", "结果", "乘", "加", "减", "除", "表达式"])
    has_expr = any(ch in text for ch in ["+", "-", "*", "/", "(", ")"])
    if need_math and not has_expr:
        print("[中间件] clarify_missing_params 提醒：数学计算缺少表达式。")
        return {"system_prompt": "若用户需要计算但缺少表达式，请先让其提供明确的表达式，然后再调用计算工具。"}

    print("[中间件] clarify_missing_params 未发现澄清需求。")
    return None


@before_model(name="clarify_missing_params")
def clarify_missing_params(state: AgentState, *_args, **_kw) -> Optional[dict]:
    return _clarify_intent_guard(state)

# 2.3 模型后：轻度标注
def _post_model_annotate(state: AgentState) -> None:
    metrics = state.setdefault("_metrics", {})
    metrics["post_checked"] = True
    print("[中间件] post_format_checker 已执行：标记模型输出已检查。")


@after_model(name="post_format_checker")
def post_format_checker(state: AgentState, *_args, **_kw) -> None:
    _post_model_annotate(state)

# 2.4 Agent 后：打点
def _log_after_agent(state: AgentState) -> None:
    state["_last_done"] = time.time()
    print(f"[中间件] post_agent_logger 已执行：记录完成时间 {state['_last_done']}")


@after_agent(name="post_agent_logger")
def post_agent_logger(state: AgentState, *_args, **_kw) -> None:
    _log_after_agent(state)

# 2.5 动态提示（示例：夜间更简洁）
def _night_briefing_rule(state: AgentState) -> Optional[str]:
    hour = time.localtime().tm_hour
    print(f"[中间件] night_briefing 检查当前小时：{hour}")
    if hour >= 23 or hour < 7:
        print("[中间件] night_briefing 生效：进入夜间简洁模式。")
        return "夜间模式：请尽量简洁作答，不要过度展开。"
    return None


@dynamic_prompt()
def night_briefing(state: AgentState, *_args, **_kw) -> Optional[str]:
    return _night_briefing_rule(state)


def install_decorator_middlewares():
    """组合装饰器型中间件（顺序会影响效果）"""
    print("[中间件] 开始安装装饰器型中间件。")
    return [
        rate_guard,
        night_briefing,
        clarify_missing_params,
        post_format_checker,
        post_agent_logger,
    ]


# =========================================================
# 3) 类（Class-based）中间件
# =========================================================
def _record_latency(req: ModelRequest, latency_ms: float) -> None:
    metrics = req.state.setdefault("_metrics", {})
    metrics["last_model_latency_ms"] = latency_ms
    print(f"[中间件] timing_and_retry 记录模型耗时：{latency_ms} ms")


def _timed_retry_wrapper_sync(
    req: ModelRequest,
    call: Callable[[ModelRequest], ModelResponse | AIMessage],
    retries: int = 1,
) -> ModelResponse | AIMessage:
    print("[中间件] timing_and_retry 同步封装开始。")
    t0 = time.time()
    try:
        resp = call(req)
    except Exception:
        print("[中间件] timing_and_retry 捕获异常，尝试重试。")
        if retries > 0:
            resp = call(req)
        else:
            raise
    latency_ms = round((time.time() - t0) * 1000, 2)
    _record_latency(req, latency_ms)
    print("[中间件] timing_and_retry 同步封装结束。")
    return resp


async def _timed_retry_wrapper_async(
    req: ModelRequest,
    call: Callable[[ModelRequest], Awaitable[ModelResponse | AIMessage]],
    retries: int = 1,
) -> ModelResponse | AIMessage:
    print("[中间件] timing_and_retry 异步封装开始。")
    t0 = time.time()
    try:
        resp = await call(req)
    except Exception:
        print("[中间件] timing_and_retry 捕获异步异常，尝试重试。")
        if retries > 0:
            resp = await call(req)
        else:
            raise
    latency_ms = round((time.time() - t0) * 1000, 2)
    _record_latency(req, latency_ms)
    print("[中间件] timing_and_retry 异步封装结束。")
    return resp


class TimingAndRetryMiddleware(AgentMiddleware):
    """
    wrap_model_call 风格示例：
    - 计算模型调用耗时
    - 捕获异常并简单重试一次
    """

    @property
    def name(self) -> str:
        return "timing_and_retry"

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse | AIMessage],
    ) -> ModelResponse | AIMessage:
        return _timed_retry_wrapper_sync(request, handler)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse | AIMessage]],
    ) -> ModelResponse | AIMessage:
        return await _timed_retry_wrapper_async(request, handler)


class ConversationGuardMiddleware(AgentMiddleware):
    """
    会话治理中间件（示例）：
    - before_agent：输入长度校验
    - before_model：历史过多时先总结
    - after_model：输出过长时截断
    """

    def __init__(self, max_input_chars: int = 3000, max_output_chars: int = 4000):
        self.max_input_chars = max_input_chars
        self.max_output_chars = max_output_chars

    @property
    def name(self) -> str:
        return "conversation_guard"

    def before_agent(self, state: AgentState, runtime=None, **kwargs) -> dict | None:  # noqa: ARG002
        print("[中间件] ConversationGuard.before_agent 检查输入长度。")
        last = state.get("messages", [])[-1] if state.get("messages") else None
        if isinstance(last, HumanMessage) and len(last.content or "") > self.max_input_chars:
            msg = AIMessage(content="输入过长，已拒绝处理。请缩短后重试。")
            print("[中间件] ConversationGuard.before_agent 拒绝：输入过长。")
            return {"messages": state["messages"] + [msg], "jump_to": "end"}
        return None

    def before_model(self, state: AgentState, runtime=None, **kwargs) -> dict | None:  # noqa: ARG002
        print("[中间件] ConversationGuard.before_model 检查消息数量。")
        if len(state.get("messages", [])) > 20:
            print("[中间件] ConversationGuard.before_model 生效：历史消息过多，提示先总结。")
            return {"system_prompt": "历史较多，请先简要总结要点，再给出答案。"}
        return None

    def after_model(self, state: AgentState, runtime=None, **kwargs) -> None:  # noqa: ARG002
        print("[中间件] ConversationGuard.after_model 检查输出长度。")
        messages = state.get("messages", [])
        if not messages:
            return
        last = messages[-1]
        if isinstance(last, AIMessage) and len(last.content or "") > self.max_output_chars:
            last.content = last.content[: self.max_output_chars] + "\n…（已自动截断过长输出）"
            print("[中间件] ConversationGuard.after_model 生效：输出过长已截断。")

# =========================================================
# 4) 统一装配出口
# =========================================================
def install_default_middlewares():
    """
    统一给 runner 调用的装配函数：
    - 先挂上装饰器型“轻治理”
    - 再挂上类中间件“重治理”（含 wrap_model_call）
    - 追加官方“预置中间件”
    返回：可直接传入 create_agent(..., middleware=[...]) 的列表
    """
    print("[中间件] 开始安装默认中间件栈。")
    middlewares = []
    middlewares.extend(install_decorator_middlewares())
    print("[中间件] 装饰器型中间件安装完成。")
    middlewares.append(TimingAndRetryMiddleware())
    print("[中间件] 计时重试中间件添加完成。")
    middlewares.append(ConversationGuardMiddleware())
    print("[中间件] 会话治理中间件添加完成。")
    middlewares.extend(install_builtin_middlewares())
    print("[中间件] 官方内置中间件添加完成。")
    print(f"[中间件] 中间件列表就绪：{[mw.name if hasattr(mw, 'name') else type(mw).__name__ for mw in middlewares]}")
    return middlewares
