# app_streamlit.py
"""
Streamlit 交互界面主入口。

职责:
- 提供调研任务输入入口，将请求发送给 Deep Agent;
- 在存在 Human-in-the-loop 中断时，展示审批界面并提交决策;
- 在任务全部执行完成后，自动展示调研结果;
- 在侧边栏展示执行日志与会话信息。
"""

import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from agent.runner import (
    build_deep_research_agent,
    invoke_with_hitl,
    resume_with_decisions,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _get_agent() -> Any:
    """从会话状态中获取 Deep Agent 实例，如不存在则创建新的实例。"""
    if "agent" not in st.session_state:
        st.session_state.agent = build_deep_research_agent(PROJECT_ROOT)
    return st.session_state.agent


def _get_thread_id() -> str:
    """获取当前会话对应的线程 ID，用于保证 Deep Agent 状态一致。"""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    return st.session_state.thread_id


def _set_pending_interrupt(interrupt: Optional[Dict[str, Any]]) -> None:
    """在会话状态中记录或清除当前待处理的中断信息。"""
    if interrupt is None:
        st.session_state.pop("pending_interrupt", None)
    else:
        st.session_state["pending_interrupt"] = interrupt


def _extract_path_info(file_path: str) -> Tuple[str, str]:
    """
    从文件路径中提取任务名称和文件名。

    返回:
        (task_name, file_name)
    """
    if not file_path or not isinstance(file_path, str) or file_path == "None":
        return "", ""

    clean_path = file_path.replace("/workspace/", "").strip("/")
    if not clean_path:
        return "", ""

    parts = [p.strip() for p in clean_path.split("/") if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[-1]
    elif len(parts) == 1:
        file_name = parts[0]
        task_name = ""
        if st.session_state.current_task:
            words = (
                st.session_state.current_task[:20]
                .replace("调研", "")
                .replace("分析", "")
                .strip()
            )
            if words:
                task_name = words[:8] if len(words) > 8 else words
        return task_name, file_name
    return "", ""


def _extract_action_path(action_args: Dict[str, Any]) -> str:
    """
    从工具调用参数中解析文件路径，兼容不同字段命名。

    优先从 path / file_path / filepath 中获取，若不存在则尝试匹配包含 'path' 的字段。
    """
    if not isinstance(action_args, dict):
        return ""

    file_path = (
        action_args.get("path")
        or action_args.get("file_path")
        or action_args.get("filepath")
    )
    if isinstance(file_path, str) and file_path.strip():
        return file_path.strip()

    for key, value in action_args.items():
        if "path" in key.lower() and value:
            return str(value).strip()

    return ""


def _check_overwrite(file_path: str) -> bool:
    """
    检查文件路径是否指向已存在的文件，用于在审批界面提示“覆盖写入”风险。
    """
    if not file_path or not isinstance(file_path, str):
        return False
    if not file_path.startswith("/workspace/"):
        return False

    actual_path = file_path.replace("/workspace/", "").strip("/")
    if not actual_path:
        return False

    full_path = os.path.join(PROJECT_ROOT, "files", actual_path)
    return os.path.exists(full_path)


def _mark_task_finished(result: Dict[str, Any]) -> None:
    """
    将当前任务标记为已完成，记录最终结果并刷新页面展示。
    """
    st.session_state.last_result = result
    st.session_state.final_result = result
    st.session_state.is_processing = False
    st.session_state.execution_log.append("✅ 任务执行完成")
    st.rerun()


def main() -> None:
    """
    Streamlit 应用主入口。
    """
    st.set_page_config(
        page_title="Deep Research Agent",
        page_icon="🔍",
        layout="wide",
    )

    agent = _get_agent()
    thread_id = _get_thread_id()

    # 初始化会话状态
    if "task_submitted" not in st.session_state:
        st.session_state.task_submitted = False
    if "current_task" not in st.session_state:
        st.session_state.current_task = ""
    if "execution_log" not in st.session_state:
        st.session_state.execution_log: List[str] = []
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "rejected_paths" not in st.session_state:
        st.session_state.rejected_paths: List[str] = []
    if "rejected_actions" not in st.session_state:
        st.session_state.rejected_actions: List[str] = []

    # 页面标题
    st.title("🔍 Deep Research Planner")
    st.caption("通用型智能调研助手，可用于技术、产品、市场等多种场景")

    # 侧边栏: 会话信息 + 执行日志
    with st.sidebar:
        st.subheader("会话信息")
        st.write(f"会话 ID: `{thread_id}`")

        if st.button("重置会话", type="secondary", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        st.markdown("---")

        pending_interrupt = st.session_state.get("pending_interrupt")
        if pending_interrupt:
            st.warning("当前存在待处理的审批请求。")
        else:
            st.info("当前没有待处理的审批请求。")

        st.markdown("---")
        st.subheader("执行日志")
        log_text = "\n".join(st.session_state.execution_log) or "尚无执行记录。"
        st.code(log_text, language=None)

    # ===== 主区域：上 - 调研任务输入，下 - 审批与结果 =====

    # 上：调研任务
    st.subheader("调研任务")

    user_input = st.text_area(
        "请输入调研任务描述：",
        height=160,
        value=st.session_state.current_task,
        placeholder=(
            "例如：请帮我调研当前主流的向量数据库产品，包括 Milvus、Pinecone、Weaviate、Qdrant 等，"
            "对比它们的性能特点、适用场景、部署方式和成本，输出一份 2～3 页 A4 纸的 Markdown 选型报告。"
        ),
    )

    if st.button("提交任务并开始调研", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("请先输入调研任务描述。")
        else:
            st.session_state.current_task = user_input.strip()
            st.session_state.task_submitted = True
            st.session_state.execution_log.append(
                f"📝 接收到调研任务：{st.session_state.current_task}"
            )
            st.session_state.is_processing = True

            with st.spinner("正在执行调研任务..."):
                result = invoke_with_hitl(agent, user_input, thread_id)

            st.session_state.execution_log.append("🤖 初次执行完成")
            st.session_state.last_result = result

            interrupt_list = result.get("__interrupt__")
            if interrupt_list and len(interrupt_list) > 0:
                _set_pending_interrupt(interrupt_list[0].value)
                st.session_state.execution_log.append("⚠️ 检测到需要审批的操作")
                st.rerun()
            else:
                _mark_task_finished(result)

    st.markdown("---")

    # 下：审批与调研结果展示
    st.subheader("审批与结果")

    pending_interrupt = st.session_state.get("pending_interrupt")
    final_result = st.session_state.get("final_result")
    last_result = st.session_state.get("last_result")

    # 1. 若有待审批的中断，则优先展示审批界面
    if pending_interrupt:
        interrupt = pending_interrupt
        action_requests = interrupt.get("action_requests", [])
        review_configs = interrupt.get("review_configs", [])

        if not action_requests:
            st.error("中断信息中缺少操作请求，已取消本次审批。")
            _set_pending_interrupt(None)
            st.session_state.is_processing = False
        else:
            st.info("Agent 在执行过程中需要确认某些敏感操作，请完成审批。")

            st.markdown("### 📋 待审批的操作")

            for idx, action in enumerate(action_requests):
                action_name = action.get("name", "")
                action_args = action.get("args", {}) or {}
                cfg = review_configs[idx] if idx < len(review_configs) else {}

                if action_name == "write_file":
                    file_path = _extract_action_path(action_args)
                    if not file_path:
                        display_path = "(未提供路径)"
                        task_name = ""
                        file_name = ""
                    else:
                        task_name, file_name = _extract_path_info(file_path)
                        display_path = file_path

                    is_overwrite = _check_overwrite(file_path)

                    st.markdown(
                        f"**#{idx + 1} 写入文件**\n\n"
                        f"- 目标路径：`{display_path}`\n"
                        f"- 文件名：`{file_name or '(未知)'}`\n"
                        f"- 任务名称：`{task_name or '(未知)'}`\n"
                    )

                    path_error = ""
                    if not file_path.startswith("/workspace/"):
                        path_error = "路径未以 `/workspace/` 开头。"
                    else:
                        parts = file_path.replace("/workspace/", "").strip("/").split("/")
                        if len(parts) < 2:
                            path_error = "路径中缺少任务名称目录。"

                    if path_error:
                        st.error(f"⚠️ 路径格式错误：{path_error}")
                        st.warning(
                            "正确格式为：`/workspace/{任务名称}/文件名`，"
                            "例如：`/workspace/向量数据库选型/research_notes.md`。"
                        )
                    elif is_overwrite:
                        st.error("⚠️ 该操作将覆盖已有文件，请谨慎确认。")
                    else:
                        st.success("✅ 路径格式校验通过。")
                else:
                    st.markdown(
                        f"**#{idx + 1} 工具调用：{action_name or '(未知工具)'}**"
                    )
                    st.json(action_args)

                allowed_decisions = cfg.get("allowed_decisions", [])
                if allowed_decisions:
                    st.caption(f"允许的决策类型: {', '.join(allowed_decisions)}")

            st.markdown("### ✅ 提交审批决策")

            with st.form(key="approval_form", clear_on_submit=False):
                decision_type = st.radio(
                    "统一决策",
                    options=["approve", "reject"],
                    format_func=lambda x: "✅ 批准" if x == "approve" else "❌ 拒绝",
                    index=0,
                    horizontal=True,
                )
                submitted = st.form_submit_button(
                    "提交决策并继续执行", use_container_width=True, type="primary"
                )

            if submitted:
                with st.spinner("正在根据审批决策继续执行..."):
                    decisions = [{"type": decision_type} for _ in action_requests]
                    result = resume_with_decisions(agent, decisions, thread_id)

                decision_action = "批准" if decision_type == "approve" else "拒绝"
                first_args = action_requests[0].get("args", {}) or {}
                action_name = action_requests[0].get("name", "")
                action_path = _extract_action_path(first_args)

                st.session_state.execution_log.append(
                    f"✓ 已{decision_action}操作: {action_name or '(未知工具)'}"
                )
                if action_path:
                    st.session_state.execution_log.append(
                        f"  文件路径: {action_path}"
                    )

                if decision_type == "reject":
                    if action_path:
                        st.session_state.rejected_paths.append(action_path)
                    reject_key = f"{action_name}:{action_path}"
                    st.session_state.rejected_actions.append(reject_key)
                    st.session_state.execution_log.append(
                        "⚠️ 操作已被拒绝。"
                    )

                _set_pending_interrupt(None)

                new_interrupt_list = result.get("__interrupt__") or []
                if not new_interrupt_list:
                    _mark_task_finished(result)
                else:
                    new_interrupt = new_interrupt_list[0].value
                    new_action_requests = new_interrupt.get("action_requests", [])

                    is_duplicate = False
                    if new_action_requests:
                        new_args = new_action_requests[0].get("args", {}) or {}
                        new_action_name = new_action_requests[0].get("name", "")
                        new_action_path = _extract_action_path(new_args)
                        new_action_key = f"{new_action_name}:{new_action_path}"

                        if (
                            new_action_key in st.session_state.rejected_actions
                            or (
                                new_action_path
                                and new_action_path in st.session_state.rejected_paths
                            )
                        ):
                            is_duplicate = True

                    if is_duplicate:
                        st.session_state.execution_log.append(
                            "❌ 检测到重复操作（已被拒绝），自动跳过并继续执行。"
                        )
                        with st.spinner("自动跳过重复操作并继续执行..."):
                            auto_decisions = [{"type": "reject"}]
                            auto_result = resume_with_decisions(
                                agent, auto_decisions, thread_id
                            )

                        auto_interrupt_list = auto_result.get("__interrupt__") or []
                        if not auto_interrupt_list:
                            _mark_task_finished(auto_result)
                        else:
                            _set_pending_interrupt(auto_interrupt_list[0].value)
                            st.session_state.is_processing = True
                            st.rerun()
                    else:
                        _set_pending_interrupt(new_interrupt)
                        st.session_state.is_processing = True
                        st.rerun()

    # 2. 若没有待审批中断，则展示调研结果
    else:
        result_to_show = final_result or last_result
        if not result_to_show:
            st.info("尚无调研结果，请在上方提交调研任务。")
        else:
            st.markdown("### 🧾 调研结果")
            messages = result_to_show.get("messages", [])
            if messages:
                st.markdown(messages[-1].content)
            else:
                st.json(result_to_show)


if __name__ == "__main__":
    main()
