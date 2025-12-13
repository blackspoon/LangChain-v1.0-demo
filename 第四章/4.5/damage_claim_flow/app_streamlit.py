# app_streamlit.py
import uuid
import streamlit as st

from agent.runner import invoke_once, resume_with_decision
from agent.state import ClaimState

st.set_page_config(page_title="破损理赔智能助手", layout="wide")

st.title("🧾 破损商品智能理赔助手")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("理赔请求输入")

    order_id = st.text_input("订单号", "ORDER-001")
    user_id = st.text_input("用户 ID", "USER-001")
    order_amount = st.number_input("订单金额", min_value=0.0, value=199.0, step=1.0)
    order_category = st.selectbox(
        "商品品类",
        ["fragile", "electronics", "virtual", "other"],
        index=0,
    )
    damage_description = st.text_area("破损描述", "收到时外包装严重破损，商品玻璃杯碎裂。")

    st.markdown("—— 风险 mock 参数 ——")
    total_claims_count = st.number_input("历史理赔总次数", 0, 100, 1)
    recent_claims_count = st.number_input("最近 30 天理赔次数", 0, 100, 0)
    total_orders_count = st.number_input("历史总订单数", 1, 1000, 10)

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if st.button("提交理赔请求", type="primary"):
        st.session_state.thread_id = str(uuid.uuid4())  # 每次点击新开线程

        init_state: ClaimState = {
            "request_id": f"REQ-{st.session_state.thread_id}",
            "user_id": user_id,
            "order_id": order_id,
            "order_amount": float(order_amount),
            "order_category": order_category,
            "damage_description": damage_description,
            "total_claims_count": int(total_claims_count),
            "recent_claims_count": int(recent_claims_count),
            "total_orders_count": int(total_orders_count),
            "logs": [],
        }

        result = invoke_once(init_state, thread_id=st.session_state.thread_id)
        st.session_state.last_result = result

with col_right:
    st.subheader("执行结果 / 日志")

    result = st.session_state.get("last_result")
    if not result:
        st.info("请先在左侧提交一条理赔请求。")
    else:
        interrupt_payload = result.get("__interrupt__")
        logs = result.get("logs", [])

        if logs:
            st.markdown("### 流程日志")
            for line in logs:
                st.write(line)

        if interrupt_payload:
            st.warning("当前流程等待人工审批。请在下方给出审批结果，然后点击继续执行。")
            st.json(interrupt_payload)

            decision = st.selectbox("审批决策", ["approve", "modify", "reject"])
            comment = st.text_area("审批备注", "同意系统推荐方案。")
            override_solution = st.text_area("覆盖推荐方案（可选）", "")

            if st.button("提交审批并继续执行"):
                payload = {
                    "decision": decision,
                    "comment": comment,
                    "override_solution": override_solution or None,
                }
                resumed = resume_with_decision(payload, thread_id=st.session_state.thread_id)
                st.session_state.last_result = resumed
                st.rerun()
        else:
            st.success("本次理赔流程已完成。")
            st.markdown("### 最终结果说明")
            st.write(result.get("user_notice", "(缺少 user_notice 字段)"))
            st.markdown("---")
            st.json(result)
