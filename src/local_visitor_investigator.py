
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visitor_root_cause_report.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "ai_investigations"


def _number(row, column, default=0.0):
    """Read a numeric value safely from a visitor row."""
    value = pd.to_numeric(row.get(column), errors="coerce")
    return default if pd.isna(value) else float(value)



def investigate_visitor(row):
    """Create an evidence-backed, local visitor investigation."""

    def number(column, default=0.0):
        value = pd.to_numeric(row.get(column), errors="coerce")
        return default if pd.isna(value) else float(value)

    def unique(items):
        return list(dict.fromkeys(items))

    visitor_id = str(row.get("visitorid", "Unknown"))
    risk = number("inactivity_risk_score")
    risk_band = str(row.get("risk_band", "Unknown")).title()

    gap = number("days_since_last_activity")
    lifetime = number("days_since_first_activity")
    sessions = number("session_count")
    active_days = number("active_days")
    views = number("view_events")
    carts = number("cart_events")
    transactions = number("transaction_events")

    evidence = []
    hypotheses = []
    actions = []
    experiments = []

    # --------------------------------------------------
    # 1. EXECUTIVE SUMMARY
    # --------------------------------------------------
    summary = (
        f"Visitor {visitor_id} is in the {risk_band} risk band. "
        f"The model score is {risk:.1%}. "
        "This is a predicted inactivity score, not a confirmed "
        "probability of permanent customer churn."
    )

    # --------------------------------------------------
    # 2. RECENCY / ACTIVITY GAP
    # --------------------------------------------------
    if gap >= 14:
        evidence.append(
            f"HIGH-SIGNAL OBSERVATION: {gap:.1f} days have elapsed "
            "since the last recorded activity."
        )
        hypotheses.append(
            "Recent engagement appears limited: the long recorded "
            "activity gap is consistent with a visitor who has stopped "
            "returning. The data does not reveal the reason."
        )
        actions.append(
            "P1 — If the visitor has valid marketing permission, test "
            "one relevant, low-frequency re-engagement message."
        )
        experiments.append(
            "Primary metric: return-visit rate within a predefined "
            "follow-up window, compared with a randomized holdout group."
        )
    elif gap >= 7:
        evidence.append(
            f"OBSERVATION: The last recorded activity was "
            f"{gap:.1f} days ago."
        )
        hypotheses.append(
            "The activity gap may indicate declining engagement, "
            "although some visitors naturally return infrequently."
        )
        actions.append(
            "P2 — Monitor for a return visit and test relevant discovery "
            "content where appropriate."
        )
        experiments.append(
            "Measure return-visit rate against a comparable holdout group."
        )
    else:
        evidence.append(
            f"The recorded activity gap is {gap:.1f} days."
        )

    # --------------------------------------------------
    # 3. SESSION DEPTH / REPEAT ENGAGEMENT
    # --------------------------------------------------
    if sessions <= 1 and active_days <= 1:
        evidence.append(
            f"Only {sessions:.0f} session(s) across "
            f"{active_days:.0f} active day(s) are recorded."
        )
        hypotheses.append(
            "The observed journey has little repeat engagement. "
            "This could reflect low interest, a short visit, or "
            "incomplete observation; the available fields cannot "
            "distinguish these explanations."
        )
        actions.append(
            "P1 — Improve first-visit discovery with clear categories, "
            "useful product recommendations, and simple navigation."
        )
        experiments.append(
            "Measure the proportion of visitors who return and view "
            "a product after the discovery experience is changed."
        )
    elif sessions <= 2:
        evidence.append(
            f"The report contains {sessions:.0f} recorded session(s)."
        )
        hypotheses.append(
            "Repeat engagement is limited in the available observation "
            "window, but the session count alone does not establish why."
        )

    # --------------------------------------------------
    # 4. PRODUCT DISCOVERY
    # --------------------------------------------------
    if views <= 2:
        evidence.append(
            f"Only {views:.0f} product-view event(s) are recorded."
        )
        hypotheses.append(
            "The visitor may have had limited product exploration. "
            "This is an observed browsing pattern, not proof of a "
            "navigation or catalog problem."
        )
        actions.append(
            "P2 — Test clearer product discovery, category navigation, "
            "or recommendations relevant to the current journey."
        )
        experiments.append(
            "Track product-detail views per visitor and add-to-cart "
            "rate before and after the discovery change."
        )
    else:
        evidence.append(
            f"{views:.0f} product-view event(s) are recorded."
        )

    # --------------------------------------------------
    # 5. CART / TRANSACTION PATTERN
    # --------------------------------------------------
    if carts > 0 and transactions == 0:
        evidence.append(
            f"The report contains {carts:.0f} cart event(s) and "
            "zero recorded transaction events."
        )
        hypotheses.append(
            "This pattern is consistent with possible cart abandonment. "
            "It does not prove that checkout failed, that the visitor "
            "intended to purchase, or that all transactions were captured."
        )
        actions.append(
            "P1 — Review checkout usability and event tracking. If "
            "permission and policy allow, test a helpful cart reminder."
        )
        experiments.append(
            "Measure transaction conversion among eligible cart visitors "
            "using a randomized reminder group and holdout group."
        )
    elif transactions > 0:
        evidence.append(
            f"{transactions:.0f} transaction event(s) are recorded."
        )
        hypotheses.append(
            "At least one transaction event is present in the report. "
            "The available fields do not establish subsequent satisfaction "
            "or long-term retention."
        )

    # --------------------------------------------------
    # 6. EXISTING RULE-BASED DIAGNOSIS
    # --------------------------------------------------
    existing_causes = str(
        row.get("root_cause_indicators", "")
    ).strip()
    existing_actions = str(
        row.get("recommended_actions", "")
    ).strip()

    if existing_causes and existing_causes.lower() != "nan":
        evidence.append(
            "Existing diagnostic flags: " + existing_causes
        )

    # --------------------------------------------------
    # 7. PRIORITIZE / DEDUPLICATE ACTIONS
    # --------------------------------------------------
    if not evidence:
        evidence.append(
            "Insufficient behavioral fields are available to produce "
            "a detailed visitor investigation."
        )

    if not hypotheses:
        hypotheses.append(
            "The supplied fields do not support a specific behavioral "
            "hypothesis. Inspect the journey data before intervening."
        )

    if not actions:
        actions.append(
            "P3 — Continue monitoring and validate any intervention "
            "with a controlled experiment."
        )

    actions = unique(actions)[:4]
    experiments = unique(experiments)[:4]

    if not experiments:
        experiments.append(
            "Define a baseline, select one primary engagement metric, "
            "and compare a treatment group with a suitable holdout group."
        )

    if existing_actions and existing_actions.lower() != "nan":
        evidence.append(
            "Existing system recommendations for reference: "
            + existing_actions
        )

    # Confidence describes evidence coverage, not probability of churn.
    evidence_count = sum([
        gap >= 7,
        sessions <= 2,
        views <= 2,
        carts > 0 and transactions == 0,
    ])

    if evidence_count >= 3:
        confidence = (
            "Moderate evidence coverage: several behavioral signals "
            "are available, but their causes remain unconfirmed."
        )
    elif evidence_count >= 1:
        confidence = (
            "Limited-to-moderate evidence coverage: one or more signals "
            "support the hypotheses, but alternative explanations remain."
        )
    else:
        confidence = (
            "Limited evidence coverage: the available behavioral signals "
            "do not strongly distinguish possible explanations."
        )

    limitations = [
        "The score represents model-predicted inactivity, not verified "
        "permanent churn or a guaranteed probability of churn.",
        "Root-cause statements are hypotheses based on observed behavior; "
        "this rule-based engine does not establish causality.",
        "Historical clickstream data may omit activity and does not reveal "
        "visitor motivations or demographics.",
        "The confidence description measures evidence coverage, not a "
        "statistically calibrated confidence probability.",
        "Evaluate recommended interventions with suitable holdout groups "
        "before claiming business impact.",
    ]

    return {
        "visitor_id": visitor_id,
        "risk_score": risk,
        "risk_band": risk_band,
        "summary": summary,
        "evidence": evidence,
        "hypotheses": hypotheses,
        "actions": actions,
        "experiments": experiments,
        "limitations": [confidence] + limitations,
    }


def format_investigation(result):
    """Format an investigation as a Markdown report."""
    lines = [
        "# Local Visitor Investigation",
        "",
        f"- Visitor ID: {result['visitor_id']}",
        f"- Risk band: {result['risk_band']}",
        f"- Predicted inactivity score: {result['risk_score']:.2%}",
        "- Engine: Local rule-based investigation",
        "",
        "## 1. Executive Summary",
        result["summary"],
        "",
        "## 2. Observed Evidence",
    ]

    lines.extend(f"- {item}" for item in result["evidence"])
    lines.extend(["", "## 3. Potential Root Causes"])
    lines.extend(f"- {item}" for item in result["hypotheses"])
    lines.extend(["", "## 4. Recommended Actions"])
    lines.extend(f"- {item}" for item in result["actions"])
    lines.extend(["", "## 5. Experiments and Success Metrics"])
    lines.extend(f"- {item}" for item in result["experiments"])
    lines.extend(["", "## 6. Limitations and Confidence"])
    lines.extend(f"- {item}" for item in result["limitations"])

    return "\n".join(lines) + "\n"


def save_investigation(result):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    safe_id = "".join(
        ch for ch in result["visitor_id"]
        if ch.isalnum() or ch in "-_"
    )

    output_path = OUTPUT_DIR / f"visitor_{safe_id}_local_investigation.md"
    output_path.write_text(
        format_investigation(result),
        encoding="utf-8"
    )
    return output_path


def load_report():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Report not found: {REPORT_PATH}"
        )

    df = pd.read_csv(REPORT_PATH)

    required = [
        "visitorid",
        "inactivity_risk_score",
        "risk_band",
        "days_since_last_activity",
        "days_since_first_activity",
        "session_count",
        "active_days",
        "view_events",
        "cart_events",
        "transaction_events",
        "root_cause_indicators",
        "recommended_actions",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing report columns: {missing}")

    df["visitorid"] = df["visitorid"].astype(str)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a local visitor investigation."
    )
    parser.add_argument("--visitor", required=True)
    args = parser.parse_args()

    report = load_report()
    matches = report[report["visitorid"] == str(args.visitor)]

    if matches.empty:
        raise SystemExit(
            f"Visitor {args.visitor} was not found in the report."
        )

    visitor = matches.iloc[-1]
    result = investigate_visitor(visitor)
    print(format_investigation(result))

    saved_path = save_investigation(result)
    print(f"\nSaved investigation to: {saved_path}")
