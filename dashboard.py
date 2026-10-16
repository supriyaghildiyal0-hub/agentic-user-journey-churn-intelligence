from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from src.local_visitor_investigator import (
    investigate_visitor,
    format_investigation,
    save_investigation,
)
# ============================================================
# Visitor Churn Intelligence Dashboard
# ============================================================

st.set_page_config(
    page_title="Churn Intelligence | AI Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
REPORT_PATH = BASE_DIR / "data" / "processed" / "visitor_root_cause_report.csv"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #0b1020; color: #e5e7eb; }
    [data-testid="stHeader"] { background: rgba(11,16,32,.96); }
    [data-testid="stSidebar"] { background: #101729; border-right: 1px solid #263148; }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] small { color: #dbe4f3 !important; }
    h1,h2,h3,h4 { color: #f8fafc !important; }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li { color: #dbe4f3; }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
        color: #cbd5e1 !important; font-size: 14px !important;
    }
    [data-testid="stMetricValue"] {
        color: #f8fafc !important; font-size: 30px !important; font-weight: 700 !important;
    }
    [data-testid="stSelectbox"] label,
    [data-testid="stTextInput"] label,
    [data-testid="stSlider"] label,
    [data-testid="stMultiSelect"] label { color: #dbe4f3 !important; }
    [data-testid="stDataFrame"] { border: 1px solid #263148; border-radius: 12px; }
    .hero {
        padding: 25px 28px; border: 1px solid #263651; border-radius: 18px;
        background: linear-gradient(120deg, #172641, #101729); margin-bottom: 22px;
    }
    .hero-title { font-size: 30px; font-weight: 800; color: #f8fafc; margin-bottom: 8px; }
    .hero-subtitle { color: #b9c7df; font-size: 14px; }
    .metric-card {
        background: #121a2c; border: 1px solid #263148; border-radius: 14px;
        padding: 18px; min-height: 126px;
    }
    .metric-label { color: #b9c7df; font-size: 13px; margin-bottom: 10px; }
    .metric-value { color: #f8fafc; font-size: 29px; font-weight: 800; }
    .metric-note { color: #9eacc4; font-size: 12px; margin-top: 7px; }
    .section-title { font-size: 20px; font-weight: 700; color: #f8fafc; margin-top: 24px; margin-bottom: 10px; }
    .profile-card {
        background: #121a2c; border: 1px solid #2c3b56; border-radius: 14px;
        padding: 18px; margin-bottom: 12px; min-height: 105px;
    }
    .profile-label { color: #b9c7df; font-size: 13px; margin-bottom: 8px; }
    .profile-value { color: #f8fafc; font-size: 25px; font-weight: 750; overflow-wrap: anywhere; }
    .risk-high { color: #fb7185; font-weight: 700; }
    .risk-medium { color: #fbbf24; font-weight: 700; }
    .risk-low { color: #34d399; font-weight: 700; }
    hr { border-color: #263148; }
    
/* ==========================================
   DARK PREMIUM EXPANDER STYLING
   ========================================== */

[data-testid="stExpander"] {
    background: #111a2d !important;
    border: 1px solid #293954 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    margin-bottom: 12px !important;
}

[data-testid="stExpander"] details {
    background: #111a2d !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] summary {
    background: #111a2d !important;
    color: #e8eef9 !important;
    padding: 14px 16px !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] summary:hover {
    background: #18243a !important;
    color: #ffffff !important;
}

[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary svg {
    color: #e8eef9 !important;
    fill: #e8eef9 !important;
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: #0d1526 !important;
    color: #dbe4f3 !important;
    border-top: 1px solid #293954 !important;
    padding: 14px 18px !important;
}

[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] li {
    color: #dbe4f3 !important;
    line-height: 1.7 !important;
}

[data-testid="stExpander"] [data-testid="stMarkdownContainer"] strong {
    color: #ffffff !important;
}

[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #ff4b5c, #e93650) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    padding: 10px 16px !important;
}

[data-testid="stDownloadButton"] button:hover {
    background: #d92d48 !important;
    border: none !important;
}

    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_report(csv_path: str, modified_time: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = [
        "visitorid",
        "inactivity_risk_score",
        "risk_band",
        "root_cause_indicators",
        "recommended_actions",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    df["inactivity_risk_score"] = pd.to_numeric(
        df["inactivity_risk_score"], errors="coerce"
    )
    df["risk_band"] = df["risk_band"].astype(str).str.strip().str.title()
    df["root_cause_indicators"] = df["root_cause_indicators"].fillna(
        "No diagnosis available"
    )
    df["recommended_actions"] = df["recommended_actions"].fillna(
        "Review visitor behavior"
    )
    df = df.dropna(subset=["visitorid", "inactivity_risk_score"])
    df["visitorid"] = df["visitorid"].astype(str)
    return df


# Load report
st.sidebar.markdown("## 📊 Churn Intelligence")
st.sidebar.caption("AI-powered visitor behavior analytics")
st.sidebar.divider()

if not REPORT_PATH.exists():
    st.error("The root-cause report was not found.")
    st.code(str(REPORT_PATH))
    st.info(
        "Run src/root_cause_diagnoser.py first. The dashboard expects "
        "visitor_root_cause_report.csv inside data/processed."
    )
    st.stop()

try:
    report = load_report(str(REPORT_PATH), REPORT_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load the report: {exc}")
    st.stop()

if report.empty:
    st.warning("The report contains no valid visitor records.")
    st.stop()

# Sidebar filters
st.sidebar.markdown("### Filters")
risk_order = ["High", "Medium", "Low"]
available_risks = [risk for risk in risk_order if risk in report["risk_band"].unique()]
selected_risks = st.sidebar.multiselect(
    "Risk categories", options=available_risks, default=available_risks
)
score_range = st.sidebar.slider(
    "Inactivity risk score", min_value=0.0, max_value=1.0,
    value=(0.0, 1.0), step=0.01,
)

filtered = report[
    report["risk_band"].isin(selected_risks)
    & report["inactivity_risk_score"].between(score_range[0], score_range[1])
].copy()

st.sidebar.divider()
st.sidebar.caption(f"Loaded report: {len(report):,} visitors")
st.sidebar.caption(f"Filtered records: {len(filtered):,}")

# Header
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Visitor Churn Intelligence</div>
        <div class="hero-subtitle">
            Behavioral risk scoring · Root-cause diagnosis ·
            Data-driven retention recommendations
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Retailrocket clickstream dataset · Risk represents predicted inactivity, "
    "not confirmed churn."
)

if filtered.empty:
    st.warning("No visitors match the selected filters. Adjust the sidebar filters.")
    st.stop()

# KPI cards
total_visitors = len(filtered)
high_count = int((filtered["risk_band"] == "High").sum())
medium_count = int((filtered["risk_band"] == "Medium").sum())
low_count = int((filtered["risk_band"] == "Low").sum())
average_risk = float(filtered["inactivity_risk_score"].mean())
high_percentage = high_count / total_visitors * 100 if total_visitors else 0

k1, k2, k3, k4 = st.columns(4)
metrics = [
    (k1, "VISITORS ANALYZED", f"{total_visitors:,}", "Records after active filters"),
    (k2, "HIGH-RISK VISITORS", f"{high_count:,}", f"{high_percentage:.1f}% of filtered visitors"),
    (k3, "AVERAGE RISK SCORE", f"{average_risk:.1%}", "Mean predicted inactivity probability"),
    (k4, "MEDIUM + LOW RISK", f"{medium_count + low_count:,}", "Outside the high-risk band"),
]
for col, label, value, note in metrics:
    with col:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>""",
            unsafe_allow_html=True,
        )

# Risk distribution
st.markdown('<div class="section-title">Risk Distribution</div>', unsafe_allow_html=True)
left, right = st.columns(2)
risk_colors = {"High": "#fb7185", "Medium": "#fbbf24", "Low": "#34d399"}
risk_counts = (
    filtered["risk_band"].value_counts().reindex(risk_order, fill_value=0)
    .rename_axis("Risk Band").reset_index(name="Visitors")
)

with left:
    fig_donut = px.pie(
        risk_counts, names="Risk Band", values="Visitors", hole=0.66,
        color="Risk Band", color_discrete_map=risk_colors,
        title="Visitor Risk Composition",
    )
    fig_donut.update_traces(
        textposition="inside", textinfo="percent+label",
        hovertemplate="%{label}<br>Visitors: %{value:,}<extra></extra>",
    )
    fig_donut.update_layout(
        template="plotly_dark", paper_bgcolor="#121a2c", plot_bgcolor="#121a2c",
        font=dict(color="#e5e7eb"), title_font=dict(color="#f8fafc", size=17),
        margin=dict(l=20, r=20, t=60, b=20), legend_title_text="Risk category",
    )
    st.plotly_chart(fig_donut, width="stretch")

with right:
    fig_bar = px.bar(
        risk_counts, x="Risk Band", y="Visitors", color="Risk Band",
        color_discrete_map=risk_colors, text="Visitors",
        title="Visitors by Risk Category",
    )
    fig_bar.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
    fig_bar.update_layout(
        template="plotly_dark", paper_bgcolor="#121a2c", plot_bgcolor="#121a2c",
        font=dict(color="#e5e7eb"), title_font=dict(color="#f8fafc", size=17),
        showlegend=False, margin=dict(l=20, r=30, t=60, b=20),
        yaxis_title="Number of visitors", xaxis_title="Risk category",
    )
    st.plotly_chart(fig_bar, width="stretch")

# Risk score analysis
st.markdown('<div class="section-title">Risk Score Analysis</div>', unsafe_allow_html=True)
fig_hist = px.histogram(
    filtered, x="inactivity_risk_score", color="risk_band", nbins=30,
    color_discrete_map=risk_colors, barmode="overlay", opacity=0.8,
    title="Distribution of Predicted Inactivity Risk",
    labels={
        "inactivity_risk_score": "Predicted inactivity risk",
        "count": "Visitors", "risk_band": "Risk band",
    },
)
fig_hist.update_layout(
    template="plotly_dark", paper_bgcolor="#121a2c", plot_bgcolor="#121a2c",
    font=dict(color="#e5e7eb"), title_font=dict(color="#f8fafc", size=17),
    margin=dict(l=20, r=20, t=60, b=20),
)
st.plotly_chart(fig_hist, width="stretch")

# Root-cause intelligence
st.markdown('<div class="section-title">Root-Cause Intelligence</div>', unsafe_allow_html=True)
cause_counter = {}
for indicator_text in filtered["root_cause_indicators"].dropna():
    for cause in str(indicator_text).split("|"):
        cause = cause.strip()
        if cause and cause != "No diagnosis available":
            cause_counter[cause] = cause_counter.get(cause, 0) + 1

if cause_counter:
    cause_df = pd.DataFrame(
        [{"Root Cause Indicator": cause, "Occurrences": count}
         for cause, count in cause_counter.items()]
    ).sort_values("Occurrences", ascending=True).tail(12)
    fig_causes = px.bar(
        cause_df, x="Occurrences", y="Root Cause Indicator",
        orientation="h", text="Occurrences",
        title="Most Frequent Diagnostic Indicators",
    )
    fig_causes.update_traces(
        marker_color="#60a5fa", textposition="outside", cliponaxis=False
    )
    fig_causes.update_layout(
        template="plotly_dark", paper_bgcolor="#121a2c", plot_bgcolor="#121a2c",
        font=dict(color="#e5e7eb"), title_font=dict(color="#f8fafc", size=17),
        margin=dict(l=20, r=40, t=60, b=20), yaxis_title="",
        xaxis_title="Indicator occurrences",
    )
    st.plotly_chart(fig_causes, width="stretch")
else:
    st.info("No root-cause indicators are available for these filters.")

st.caption(
    "Counts represent indicator occurrences. A visitor can have multiple indicators; "
    "these indicators are not proof of causation."
)

# Visitor investigation
st.markdown('<div class="section-title">Visitor Investigation</div>', unsafe_allow_html=True)
search_id = st.text_input("Search visitor by ID", placeholder="Enter a visitor ID, e.g. 1013608")
visitor_view = filtered.copy()
if search_id.strip():
    visitor_view = visitor_view[
        visitor_view["visitorid"].str.contains(search_id.strip(), regex=False)
    ]

sort_option = st.selectbox(
    "Sort visitors", ["Highest risk first", "Lowest risk first", "Visitor ID"]
)
if sort_option == "Highest risk first":
    visitor_view = visitor_view.sort_values("inactivity_risk_score", ascending=False)
elif sort_option == "Lowest risk first":
    visitor_view = visitor_view.sort_values("inactivity_risk_score", ascending=True)
else:
    visitor_view = visitor_view.sort_values("visitorid")

table_columns = [
    "visitorid", "inactivity_risk_score", "risk_band",
    "root_cause_indicators", "recommended_actions",
]
st.dataframe(
    visitor_view[table_columns], width="stretch", hide_index=True, height=360,
    column_config={
        "visitorid": st.column_config.TextColumn("Visitor ID"),
        "inactivity_risk_score": st.column_config.ProgressColumn(
            "Inactivity Risk", min_value=0.0, max_value=1.0, format="%.1%%"
        ),
        "risk_band": st.column_config.TextColumn("Risk Band"),
        "root_cause_indicators": st.column_config.TextColumn(
            "Diagnostic Indicators", width="large"
        ),
        "recommended_actions": st.column_config.TextColumn(
            "Recommended Actions", width="large"
        ),
    },
)

# Individual visitor profile
st.markdown('<div class="section-title">Individual Visitor Profile</div>', unsafe_allow_html=True)
if visitor_view.empty:
    st.info("No visitors match the current search and filters.")
else:
    visitor_ids = visitor_view["visitorid"].drop_duplicates().tolist()
    selected_id = st.selectbox(
        "Choose a visitor", visitor_ids, key="individual_visitor_select"
    )
    selected_rows = visitor_view[visitor_view["visitorid"] == selected_id]

    if not selected_rows.empty:
        visitor = selected_rows.iloc[0]
        profile_1, profile_2, profile_3 = st.columns(3)

        with profile_1:
            st.markdown(
                f"""<div class="profile-card">
                    <div class="profile-label">Visitor ID</div>
                    <div class="profile-value">{visitor["visitorid"]}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with profile_2:
            st.markdown(
                f"""<div class="profile-card">
                    <div class="profile-label">Predicted inactivity risk</div>
                    <div class="profile-value">{visitor["inactivity_risk_score"]:.2%}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with profile_3:
            risk_band = str(visitor["risk_band"])
            risk_class = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}.get(
                risk_band, ""
            )
            st.markdown(
                f"""<div class="profile-card">
                    <div class="profile-label">Risk category</div>
                    <div class="profile-value {risk_class}">{risk_band}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        diagnosis_col, actions_col = st.columns(2)
        with diagnosis_col:
            st.markdown("#### Diagnostic indicators")
            for cause in str(visitor["root_cause_indicators"]).split("|"):
                cause = cause.strip()
                if cause:
                    st.markdown(f"- {cause}")
        with actions_col:
            st.markdown("#### Recommended actions")
            for action in str(visitor["recommended_actions"]).split("|"):
                action = action.strip()
                if action:
                    st.markdown(f"- {action}")

# ============================================================
# LOCAL VISITOR INVESTIGATION AGENT
# ============================================================

st.markdown(
    '<div class="section-title">Local AI-Style Investigation Agent</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Rule-based investigation using recorded visitor behavior. "
    "Runs locally without an external API."
)

investigation_ids = (
    filtered["visitorid"]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .sort_values()
    .tolist()
)

if not investigation_ids:
    st.info("No visitors are available for investigation.")
else:
    investigation_id = st.selectbox(
        "Select a visitor to investigate",
        investigation_ids,
        key="local_investigation_visitor",
    )

    if st.button(
        "Investigate Visitor",
        key="run_local_investigation",
        type="primary",
    ):
        matching_rows = filtered[
            filtered["visitorid"].astype(str) == investigation_id
        ]

        if matching_rows.empty:
            st.warning("The selected visitor is not available.")
        else:
            visitor_row = matching_rows.iloc[0]
            investigation = investigate_visitor(visitor_row)

            st.session_state["local_investigation_result"] = investigation
            st.session_state["local_investigation_markdown"] = (
                format_investigation(investigation)
            )
            st.session_state["local_investigation_saved_path"] = str(
                save_investigation(investigation)
            )

    result = st.session_state.get("local_investigation_result")

    if result:
        st.divider()

        st.markdown("### Investigation Summary")
        st.write(result["summary"])

        metric_1, metric_2 = st.columns(2)
        metric_1.metric(
            "Predicted Inactivity Risk",
            f"{result['risk_score']:.2%}",
        )
        metric_2.metric(
            "Risk Category",
            result["risk_band"],
        )

        with st.expander("Observed Evidence", expanded=True):
            for item in result["evidence"]:
                st.markdown(f"- {item}")

        with st.expander("Potential Root Causes", expanded=True):
            for item in result["hypotheses"]:
                st.markdown(f"- {item}")

        with st.expander("Recommended Actions", expanded=True):
            for item in result["actions"]:
                st.markdown(f"- {item}")

        with st.expander("Experiments and Success Metrics"):
            for item in result["experiments"]:
                st.markdown(f"- {item}")

        with st.expander("Limitations and Confidence"):
            for item in result["limitations"]:
                st.markdown(f"- {item}")

        st.download_button(
            "Download Investigation Report",
            data=st.session_state["local_investigation_markdown"],
            file_name=(
                f"visitor_{result['visitor_id']}_investigation.md"
            ),
            mime="text/markdown",
            key="download_local_investigation",
        )

        st.caption(
            "Saved locally to: "
            + st.session_state["local_investigation_saved_path"]
        )

# Export
st.markdown('<div class="section-title">Export Results</div>', unsafe_allow_html=True)
csv_data = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇ Download filtered visitor report",
    data=csv_data,
    file_name="filtered_churn_intelligence_report.csv",
    mime="text/csv",
)

st.divider()
st.caption(
    "Agentic User Journey Intelligence & Churn Root-Cause Diagnoser | "
    "Python · Streamlit · Plotly · Pandas · Machine Learning"
)
st.caption(
    "Risk scores are model-estimated inactivity probabilities. Rule-based indicators "
    "are behavioral clues, not verified causal explanations. Inactivity is used as "
    "a proxy for churn in this dataset."
)
