"""Streamlit dashboard. All data comes from the FastAPI service (API_URL)."""

from __future__ import annotations

from typing import Any

import os
import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="Vantara Customer Intelligence", layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def get(path: str, **params: Any) -> Any:
    """GET JSON from the API (cached 5 minutes so views load fast).

    Retries brief failures: the free API caps concurrent requests and
    answers the excess with 503 instead of queueing them.
    """
    last: requests.RequestException | None = None
    for attempt in range(4):
        try:
            r = requests.get(f"{API}{path}", params=params, timeout=30)
            if r.status_code == 503:
                raise requests.HTTPError("API busy (503)", response=r)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:  # noqa: PERF203
            last = exc
            time.sleep(1.5 * (attempt + 1))
    assert last is not None
    raise last


def api_up(timeout: float = 5) -> bool:
    """True when the API health check passes."""
    try:
        return requests.get(f"{API}/health", timeout=timeout).json().get("status") == "ok"
    except Exception:  # noqa: BLE001
        return False


def wait_for_api(max_seconds: int = 180) -> bool:
    """Poll the health endpoint while the free API wakes from sleep."""
    deadline = time.monotonic() + max_seconds
    while time.monotonic() < deadline:
        if api_up(timeout=20):
            return True
        time.sleep(3)
    return False


st.title("Vantara Customer Intelligence")
st.caption(
    "Churn risk, 90-day value and segments for the Online Retail II customer base. "
    "Public UK retail data used for training and demonstration, not live business data."
)
if not api_up():
    with st.spinner(
        "Waking up the backend. Free hosting sleeps after inactivity, "
        "so the first load can take from under a minute to several minutes. "
        "No need to refresh."
    ):
        ready = wait_for_api()
    if ready:
        st.rerun()
    st.error(
        f"The API at {API} is still not reachable after a few minutes. "
        "It may still be waking up; press Retry to keep waiting, or reload the page shortly."
    )
    st.link_button(
        "Open the API wake-up page in a new tab, then come back and press Retry",
        f"{API}/health",
    )
    if st.button("Retry"):
        st.rerun()
    st.stop()

view = st.sidebar.radio(
    "View", ["Segments", "Churn leaderboard", "Revenue trends", "Customer explanation", "Batch scoring", "Model report"]
)

if view == "Segments":
    st.subheader("Customer segments")
    c1, c2, c3 = st.columns(3)
    customers = pd.DataFrame(get("/customers", limit=6000))
    seg_names = sorted(customers["segment_name"].unique())
    f_seg = c1.multiselect("Segment", seg_names, default=seg_names)
    f_country = c2.multiselect("Country", sorted(customers["country"].unique()))
    f_tier = c3.selectbox("Value tier (historical spend terciles)", ["all", "high", "medium", "low"])
    q = customers[customers["segment_name"].isin(f_seg)]
    if f_country:
        q = q[q["country"].isin(f_country)]
    if f_tier != "all":
        lo, hi = customers["clv_historical"].quantile([1 / 3, 2 / 3])
        q = q[
            {
                "high": q["clv_historical"] >= hi,
                "medium": (q["clv_historical"] >= lo) & (q["clv_historical"] < hi),
                "low": q["clv_historical"] < lo,
            }[f_tier]
        ]
    a, b, c = st.columns(3)
    a.metric("Customers", f"{len(q):,}")
    b.metric("Average churn probability", f"{q['churn_probability'].mean():.1%}" if len(q) else "-")
    c.metric("Predicted 90-day value per customer", f"{q['clv_predicted_90d'].mean():,.0f}" if len(q) else "-")
    if len(q):
        left, right = st.columns(2)
        left.plotly_chart(
            px.histogram(
                q, x="churn_probability", color="segment_name", nbins=30, title="Churn probability by segment"
            ),
            use_container_width=True,
        )
        right.plotly_chart(
            px.scatter(
                q,
                x="frequency",
                y="monetary_total",
                color="segment_name",
                log_x=True,
                log_y=True,
                hover_data=["customer_id"],
                title="Orders vs total spend",
            ),
            use_container_width=True,
        )
    prof = pd.DataFrame(get("/segments"))
    st.markdown("**Segment profiles**")
    st.dataframe(prof.drop(columns=["segment_id"]).round(2), use_container_width=True, hide_index=True)
    st.caption(
        "predicted_* columns are model outputs. observed_* columns are what actually happened in the 90 days after the "
        "cutoff, shown for model evaluation only; they would not be known when scoring a live customer."
    )

elif view == "Churn leaderboard":
    st.subheader("Who to call first: biggest value at risk")
    n = st.slider("Customers to show", 10, 100, 25)
    lb = pd.DataFrame(get("/customers/leaderboard", n=n))
    st.dataframe(
        lb[
            [
                "customer_id",
                "country",
                "segment_name",
                "churn_probability",
                "churn_risk_level",
                "clv_historical",
                "value_at_risk",
                "recency_days",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "value_at_risk = churn probability x historical spend. The score is a ranking aid, not a decision. "
        "Historical value is net of returns and cancellations, so it can be negative."
    )

elif view == "Revenue trends":
    st.subheader("Monthly revenue with a simple forecast")
    t = get("/revenue/trend")
    h = pd.DataFrame(t["history"]).assign(kind="actual")
    f = pd.DataFrame(t["forecast"]).assign(kind="forecast")
    st.plotly_chart(
        px.line(pd.concat([h, f]), x="month", y="revenue", color="kind", markers=True), use_container_width=True
    )
    st.caption(t["method"])

elif view == "Customer explanation":
    st.subheader("Why is this customer flagged?")
    cid = st.number_input("Customer ID", min_value=0, value=12347, step=1)
    if st.button("Explain"):
        r = requests.get(f"{API}/customers/{int(cid)}", timeout=30)
        if r.status_code == 404:
            st.warning("No customer with that ID. Try 12347, 12348, 12349 or any ID from the leaderboard.")
        else:
            cust = r.json()
            ex = requests.get(f"{API}/customers/{int(cid)}/explain", timeout=60).json()
            a, p_, b, c = st.columns(4)
            a.metric(
                "Churn probability",
                f"{cust['churn_probability']:.0%}",
                f"{cust['churn_risk_level']} risk",
                delta_color="inverse" if cust["churn_risk_level"] == "high" else "off",
                delta_arrow="off",
            )
            p_.metric(
                "Purchase probability, next 90 days",
                f"{cust['purchase_probability_90d']:.0%}",
                help="1 minus the churn probability: the chance of a purchase in the next 90 days",
            )
            b.metric(
                "Predicted 90-day value",
                f"{cust['clv_predicted_90d']:,.0f}",
                help=f"Historical value {cust['clv_historical']:,.0f} + predicted 90-day spend = "
                f"{cust['clv_total_projected']:,.0f} (horizon-limited CLV)",
            )
            c.markdown(f"**Segment**\n\n{cust['segment_name']}")
            if cust["clv_historical"] < 0:
                st.warning(
                    f"Historical value is negative ({cust['clv_historical']:,.0f}): this customer's returns and "
                    "cancellations exceed their purchases in the source data. "
                    "The figure is the real net record, not an error."
                )
            st.info(ex["explanation"])
            fac = pd.DataFrame(ex["top_factors"])
            st.plotly_chart(
                px.bar(
                    fac.iloc[::-1],
                    x="shap",
                    y="feature",
                    orientation="h",
                    title="Contribution to churn risk, log-odds units (right = raises risk)",
                ),
                use_container_width=True,
            )
            lm = requests.get(f"{API}/customers/{int(cid)}/explain/lime", timeout=120).json()
            lf = pd.DataFrame(lm["lime_factors"])
            st.plotly_chart(
                px.bar(
                    lf.iloc[::-1],
                    x="weight",
                    y="rule",
                    orientation="h",
                    title="LIME local explanation (right = raises churn probability)",
                ),
                use_container_width=True,
            )
            st.caption(
                f"SHAP top 5: {', '.join(lm['shap_top5'])}. Features that appear in both SHAP's and LIME's top 5: "
                f"{', '.join(lm['agree_with_shap_top5']) or 'none'}."
            )
            rec = requests.get(f"{API}/customers/{int(cid)}/recommendations", params={"n": 5}, timeout=30)
            if rec.status_code == 200:
                st.markdown("**Recommended products** (based on what similar customers also bought)")
                st.dataframe(pd.DataFrame(rec.json()["recommendations"]), hide_index=True, use_container_width=True)
            if cust["data_split"] == "train":
                st.caption("This customer was in the model's training data, so its score is an in-sample score.")

elif view == "Batch scoring":
    st.subheader("Score a CSV")
    st.write(
        "Upload a CSV with a `customer_id` column (known customers) or with at least "
        "`recency_days`, `frequency`, `monetary_total` columns."
    )
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        files = {"file": (up.name, up.getvalue(), "text/csv")}
        r = requests.post(f"{API}/predict/batch", files=files, timeout=120)
        if r.status_code != 200:
            st.error(r.json().get("detail", r.text))
        else:
            df = pd.DataFrame(r.json())
            st.dataframe(df, use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            c1.download_button(
                "Download CSV",
                requests.post(f"{API}/predict/batch", params={"format": "csv"}, files=files).content,
                "scores.csv",
                "text/csv",
            )
            c2.download_button(
                "Download PDF",
                requests.post(f"{API}/predict/batch", params={"format": "pdf"}, files=files).content,
                "scores.pdf",
                "application/pdf",
            )

else:
    st.subheader("Model report")
    m = get("/model/metadata")
    st.write(
        f"Production churn model: **{m['production_churn_model']}**. CLV model: **{m['clv_model']}**. "
        f"{m['n_customers']:,} customers; churn = no purchase in the {m['horizon_days']} days after "
        f"{m['cutoff_date']}."
    )
    res = pd.DataFrame(
        [
            {
                "model": k,
                "test ROC-AUC": v["test"]["roc_auc"],
                "test recall": v["test"]["recall"],
                "test F1": v["test"]["f1"],
            }
            for k, v in m["churn_results"].items()
        ]
    ).sort_values("test ROC-AUC", ascending=False)
    st.dataframe(res.round(3), hide_index=True, use_container_width=True)
    st.write("Targets met:", m["targets_met"])
    if m.get("recommender"):
        rv = m["recommender"]
        st.write(
            f"Product recommender (offline check on {rv['test_buyers_evaluated']} test buyers): hit rate@10 "
            f"{rv['hit_rate_at_10']:.1%} vs {rv['popularity_baseline_hit_rate_at_10']:.1%} for a popularity baseline. "
            "No live A/B test."
        )
