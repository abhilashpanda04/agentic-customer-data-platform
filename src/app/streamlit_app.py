"""
MarTech Intelligence Copilot & Customer Data Platform UI
Features:
1. Customer 360 & Unified Identity Explorer
2. RFM Segmentation & Cohort Performance
3. Multi-Touch Attribution & Channel Journey Insights
4. Agentic Copilot Chat with Human-in-the-Loop Campaign Activation Gateway
"""
import sys
from pathlib import Path

# Add project root to sys.path so 'src' is always discoverable
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.cdp.database import CDPDatabase
from src.agents.supervisor import MarTechSupervisorAgent
from src.agents.governance import GovernanceGuardrails

st.set_page_config(
    page_title="MarTech Intelligence Copilot | CDP",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize singletons in session
if "db" not in st.session_state:
    st.session_state.db = CDPDatabase()
if "agent" not in st.session_state:
    st.session_state.agent = MarTechSupervisorAgent(st.session_state.db)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_approval_token" not in st.session_state:
    st.session_state.pending_approval_token = None

db = st.session_state.db
agent = st.session_state.agent

# Sidebar Navigation
st.sidebar.title("🎯 MarTech CDP Copilot")
st.sidebar.caption("Enterprise Customer Intelligence & Activation")
nav_selection = st.sidebar.radio(
    "Navigation",
    ["Conversational Copilot", "Customer 360 Explorer", "RFM & CLTV Analytics", "Multi-Touch Attribution", "Governance & Audit"]
)

# ----------------------------------------------------
# 1. CONVERSATIONAL COPILOT & HUMAN-IN-THE-LOOP ACTIVATION
# ----------------------------------------------------
if nav_selection == "Conversational Copilot":
    st.title("🤖 Agentic Marketing Intelligence Copilot")
    st.markdown("""
    Ask natural-language questions to discover at-risk audiences, model high-value lookalikes, 
    diagnose multi-touch channel attribution, or formulate governed marketing campaigns.
    """)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("💬 Chat with Copilot")

        # Quick sample prompts
        st.caption("Quick actions:")
        qp_col1, qp_col2, qp_col3 = st.columns(3)
        sample_query = None
        if qp_col1.button("🎯 At-Risk Win-back"):
            sample_query = "Identify at-risk high-value customers for win-back"
        if qp_col2.button("👥 Lookalike Expansion"):
            sample_query = "Generate a lookalike audience of our Champions"
        if qp_col3.button("📊 Attribution Comparison"):
            sample_query = "Compare attribution models across our channels"

        # Chat display
        for chat in st.session_state.chat_history:
            with st.chat_message(chat["role"]):
                st.write(chat["content"])
                if "tool_data" in chat and chat["tool_data"]:
                    with st.expander("🛠️ Structured Agent Tool Result"):
                        st.json(chat["tool_data"])

        user_input = st.chat_input("Ask a marketing question or type an activation instruction...")
        prompt_to_run = sample_query if sample_query else user_input

        if prompt_to_run:
            st.session_state.chat_history.append({"role": "user", "content": prompt_to_run})
            with st.chat_message("user"):
                st.write(prompt_to_run)

            with st.spinner("Agent analyzing customer graphs & running models..."):
                response = agent.process_query(prompt_to_run)
                bot_text = response.get("insights", "Analysis complete.")
                if response.get("recommended_next_action"):
                    bot_text += f"\n\n**Next Recommended Action:** {response['recommended_next_action']}"
                
                # Check if approval is needed
                tool_out = response.get("tool_output", {})
                if isinstance(tool_out, dict) and "required_approval_token" in tool_out:
                    st.session_state.pending_approval_token = tool_out["required_approval_token"]

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": bot_text,
                    "tool_data": response.get("tool_output")
                })
                with st.chat_message("assistant"):
                    st.write(bot_text)
                    if response.get("tool_output"):
                        with st.expander("🛠️ Structured Agent Tool Result"):
                            st.json(response["tool_output"])

    with col2:
        st.subheader("🛡️ Governance & Approval Gateway")
        st.info("Human-in-the-loop: Autonomous activations are strictly gated by policy.")

        if st.session_state.pending_approval_token:
            st.warning(f"⚠️ **Pending Approval Request**\nToken: `{st.session_state.pending_approval_token}`")
            col_a, col_b = st.columns(2)
            if col_a.button("✅ Approve & Activate"):
                act_res = agent.process_query(f"Activate campaign MKT_APPRV_{st.session_state.pending_approval_token.split('_')[-1]}")
                st.success(act_res.get("insights", "Campaign Activated!"))
                st.session_state.pending_approval_token = None
            if col_b.button("❌ Reject Plan"):
                st.info("Campaign rejected and canceled.")
                st.session_state.pending_approval_token = None
        else:
            st.success("✅ No pending campaigns waiting for human sign-off.")

        st.markdown("---")
        st.subheader("📋 Active Campaigns")
        activations = db.query("SELECT * FROM campaign_activation ORDER BY activated_at DESC LIMIT 5;")
        if activations.empty:
            st.caption("No activated campaigns yet.")
        else:
            st.dataframe(activations[["campaign_name", "target_channel", "audience_size", "approval_status"]], use_container_width=True)

# ----------------------------------------------------
# 2. CUSTOMER 360 EXPLORER
# ----------------------------------------------------
elif nav_selection == "Customer 360 Explorer":
    st.title("👤 Customer 360 Unified Golden Records")
    st.markdown("Explore profiles unified across devices, emails, CRM, and cookies via graph identity resolution.")

    profiles = db.query("""
        SELECT p.unified_customer_id, p.first_name, p.last_name, p.primary_email,
               p.subscription_status, f.rfm_segment, f.predicted_cltv, f.lead_score,
               f.recency_days, f.frequency, f.monetary
        FROM customer_profile p
        JOIN customer_features f ON p.unified_customer_id = f.unified_customer_id
        ORDER BY f.predicted_cltv DESC
        LIMIT 200;
    """)

    safe_profiles = GovernanceGuardrails.mask_dataframe_pii(profiles)
    st.dataframe(safe_profiles, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Deep Profile Inspection")
    ucid_list = safe_profiles["unified_customer_id"].tolist()
    selected_ucid = st.selectbox("Select Unified Customer ID", ucid_list)

    if selected_ucid:
        c1, c2, c3 = st.columns(3)
        cust_row = safe_profiles[safe_profiles["unified_customer_id"] == selected_ucid].iloc[0]
        
        c1.metric("Predicted 180-Day CLTV", f"${cust_row['predicted_cltv']:,.2f}")
        c2.metric("Total Historical Revenue", f"${cust_row['monetary']:,.2f}")
        c3.metric("Lead Conversion Propensity", f"{cust_row['lead_score']*100:.1f}%")

        # Identity linkages graph
        st.write("#### 🔗 Linked Identifiers (Identity Map)")
        idm = db.query(f"SELECT identifier_type, identifier_value FROM identity_map WHERE unified_customer_id = '{selected_ucid}';")
        st.dataframe(idm, use_container_width=True)

        # Recent Purchases
        st.write("#### 🛍️ Transaction History")
        tx_hist = db.query(f"SELECT order_id, order_timestamp, revenue, gross_margin, channel FROM transaction_fact WHERE unified_customer_id = '{selected_ucid}' ORDER BY order_timestamp DESC;")
        if tx_hist.empty:
            st.caption("No purchases recorded yet.")
        else:
            st.dataframe(tx_hist, use_container_width=True)

# ----------------------------------------------------
# 3. RFM & CLTV ANALYTICS
# ----------------------------------------------------
elif nav_selection == "RFM & CLTV Analytics":
    st.title("📈 RFM Segmentation & Customer Lifetime Value")

    col1, col2 = st.columns(2)
    rfm_summary = db.query("""
        SELECT rfm_segment, COUNT(*) as count, AVG(predicted_cltv) as avg_cltv, AVG(monetary) as avg_rev
        FROM customer_features GROUP BY rfm_segment;
    """)

    with col1:
        st.subheader("Segment Distribution")
        fig_pie = px.pie(rfm_summary, names="rfm_segment", values="count", hole=0.4, title="Customer Share by Segment")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Predicted CLTV by Segment")
        fig_bar = px.bar(rfm_summary, x="rfm_segment", y="avg_cltv", color="rfm_segment", title="Average Forward CLTV ($)")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Recency vs. Frequency (Bubble Size = Predicted CLTV)")
    scatter_df = db.query("SELECT recency_days, frequency, predicted_cltv, rfm_segment FROM customer_features LIMIT 500;")
    fig_scatter = px.scatter(
        scatter_df, x="recency_days", y="frequency", size="predicted_cltv",
        color="rfm_segment", hover_name="rfm_segment", title="Behavioral Matrix"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ----------------------------------------------------
# 4. MULTI-TOUCH ATTRIBUTION
# ----------------------------------------------------
elif nav_selection == "Multi-Touch Attribution":
    st.title("⚖️ Multi-Touch Attribution Engine")
    st.markdown("Compare credit distribution across marketing touchpoints using First-Touch, Last-Touch, Linear, Time-Decay, and Markov Chain models.")

    mta_res = agent.tools.run_mta_attribution_comparison()["attribution_comparison"]
    df_mta = pd.DataFrame(mta_res)

    st.dataframe(df_mta, use_container_width=True)

    # Comparison Chart
    st.subheader("Attributed Conversions by Channel & Model")
    melted = df_mta.melt(
        id_vars=["channel"],
        value_vars=["first_touch_conversions", "last_touch_conversions", "linear_conversions", "time_decay_conversions"],
        var_name="attribution_model",
        value_name="conversions"
    )
    fig_mta = px.bar(melted, x="channel", y="conversions", color="attribution_model", barmode="group")
    st.plotly_chart(fig_mta, use_container_width=True)

# ----------------------------------------------------
# 5. GOVERNANCE & AUDIT LOGS
# ----------------------------------------------------
elif nav_selection == "Governance & Audit":
    st.title("🛡️ Enterprise Governance & Consent Compliance")
    st.markdown("Ensures CCPA/GDPR compliance, PII redaction, suppression lists, and minimum privacy thresholds.")

    c_summary = db.query("""
        SELECT 
            SUM(CASE WHEN marketing_opt_in THEN 1 ELSE 0 END) as marketing_opted_in,
            SUM(CASE WHEN NOT marketing_opt_in THEN 1 ELSE 0 END) as marketing_opted_out,
            SUM(CASE WHEN do_not_track THEN 1 ELSE 0 END) as do_not_track_active
        FROM consent_preference;
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Marketing Opt-In", int(c_summary["marketing_opted_in"].iloc[0]))
    c2.metric("Suppressed (Opt-Out)", int(c_summary["marketing_opted_out"].iloc[0]))
    c3.metric("Do Not Track (DNT)", int(c_summary["do_not_track_active"].iloc[0]))

    st.subheader("PII Masking In Action")
    sample_text = "Contact VIP customer alice.smith@corp.com at (415) 890-1234 regarding renewal."
    st.code(f"Input:  {sample_text}\nMasked: {GovernanceGuardrails.mask_pii(sample_text)}", language="text")
