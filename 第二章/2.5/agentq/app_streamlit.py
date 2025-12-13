from __future__ import annotations

"""
Streamlit 前端入口：
- 提供一个最小聊天界面，用于演示 AgentRunner 的问答能力
- 包含快捷示例、对话历史、输入表单与侧边栏重置逻辑
"""

import streamlit as st
from agent.runner import build_runner


st.set_page_config(
    page_title="AgentQ — 问答智能体",
    page_icon="🤖",
    layout="centered",
)

# --- 初始化 ---
if "runner" not in st.session_state:
    # AgentRunner：封装模型+工具+中间件（复用同一个实例）
    st.session_state.runner = build_runner()
if "history" not in st.session_state:
    # history：前端展示的聊天记录（列表保存角色和文本）
    st.session_state.history = []
if "_prefill" not in st.session_state:
    # _prefill：输入框的预填内容，示例按钮会写入它
    st.session_state._prefill = ""

st.title("AgentQ — 问答智能体")

# 快捷示例（不改后端，仅便于输入）
with st.expander("示例问题（点一下即可带入输入框）", expanded=False):
    c1, c2, c3 = st.columns(3)
    if c1.button("今天上海天气怎样？"):
        st.session_state._prefill = "今天上海天气怎样？"
        st.rerun()
    if c2.button("计算 (3+5)*12"):
        st.session_state._prefill = "计算 (3+5)*12"
        st.rerun()
    if c3.button("帮我总结：LangChain 作用是什么？"):
        st.session_state._prefill = "帮我总结：LangChain 作用是什么？"
        st.rerun()

# 展示历史
for role, content in st.session_state.history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content)

# 输入区域：使用 form + text_input，支持预填
with st.form(key="chat_form", clear_on_submit=True):
    user_text = st.text_input(
        "输入",
        value=st.session_state._prefill,
        placeholder="例如：今天上海天气怎样？ 或 计算 (3+5)*12",
        label_visibility="collapsed",
    )
    sent = st.form_submit_button("发送")

# 发送
if sent:
    st.session_state._prefill = ""
    prompt = (user_text or "").strip()
    if prompt:
        st.session_state.history.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            # 调用 AgentRunner（runner.py中）的 invoke 方法，传入用户输入的 prompt
            reply = st.session_state.runner.invoke(prompt)
        except Exception as e:
            reply = f"抱歉，发生错误：{e}"

        st.session_state.history.append(("assistant", reply))
        with st.chat_message("assistant"):
            st.markdown(reply)

# 侧边栏：重置
with st.sidebar:
    st.subheader("会话控制")
    if st.button("清空会话 / 重置记忆", use_container_width=True):
        # 重建 runner（清空短期记忆），清空历史
        st.session_state.runner = build_runner()
        st.session_state.history = []
        st.session_state._prefill = ""
        st.rerun()

