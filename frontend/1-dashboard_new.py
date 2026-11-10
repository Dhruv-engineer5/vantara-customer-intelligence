"""Streamlit dashboard. All data comes from the FastAPI service (API_URL)."""

from __future__ import annotations

from typing import Any

import base64
import hashlib
import os
import secrets
import time
import urllib.parse

import pandas as pd
import plotly.express as px
import requests
from api_client import get
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="Vantara Customer Intelligence", layout="wide")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
AUTH_ON = bool(SUPABASE_URL and SUPABASE_ANON_KEY)


def _sb(path: str, payload: dict) -> requests.Response:
    """Call the Supabase Auth (GoTrue) REST endpoint."""
    return requests.post(
        f"{SUPABASE_URL}/auth/v1/{path}", json=payload, headers={"apikey": SUPABASE_ANON_KEY}, timeout=30
    )


DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "").rstrip("/")

# Email/password sign-in stays hidden until real email delivery is configured
# (Supabase's default SMTP only mails project team members). Flip on for
# controlled testing with EMAIL_AUTH_ENABLED=true.
EMAIL_AUTH_ON = os.environ.get("EMAIL_AUTH_ENABLED", "").lower() in ("1", "true", "yes")

# Short-lived server-side store of PKCE verifiers for in-flight OAuth sign-ins.
# Keyed by a random nonce carried in the redirect URL; entries are used once and
# expire after 10 minutes. cache_resource makes this one process-wide singleton:
# Streamlit re-executes the script for every rerun/session with fresh module
# state, so a plain module-level dict is reset and the verifier is lost before
# the OAuth callback returns. Nothing here survives a restart, which is fine:
# an interrupted sign-in just starts over.
@st.cache_resource
def _pkce_store() -> dict:
    return {}


def _oauth_url(provider: str) -> str:
    """Build the Supabase authorize URL for a social provider (PKCE flow)."""
    verifier = base64.urlsafe_b64encode(os.urandom(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    nonce = secrets.token_urlsafe(18)
    store = _pkce_store()
    store[nonce] = (verifier, time.time() + 600)
    for k in [k for k, (_, exp) in store.items() if exp < time.time()]:
        del store[k]
    redirect = f"{DASHBOARD_URL}/?s={nonce}"
    params = {
        "provider": provider,
        "redirect_to": redirect,
        "flow_type": "pkce",
        "code_challenge": challenge,
        "code_challenge_method": "s256",
    }
    return f"{SUPABASE_URL}/auth/v1/authorize?{urllib.parse.urlencode(params)}"


def _handle_oauth_callback() -> None:
    """Exchange a Supabase OAuth PKCE callback (?code=...) for a stored session."""
    code = st.query_params.get("code")
    nonce = st.query_params.get("s")
    error = st.query_params.get("error_description") or st.query_params.get("error")
    if error:
        st.query_params.clear()
        st.sidebar.error(f"Sign-in was not completed: {error}")
        return
    if not code:
        return
    store = _pkce_store()
    entry = store.pop(nonce, None) if nonce else None
    st.query_params.clear()
    if not entry or entry[1] < time.time():
        reason = "missing" if not entry else "expired"
        print(f"oauth callback failed: pkce entry {reason}; store_size={len(store)}", flush=True)
        st.session_state.auth_error = "That sign-in attempt expired. Please try again."
        return
    r = requests.post(
        f"{SUPABASE_URL}/auth/v1/token?grant_type=pkce",
        json={"auth_code": code, "code_verifier": entry[0]},
        headers={"apikey": SUPABASE_ANON_KEY},
        timeout=30,
    )
    if r.status_code == 200 and r.json().get("access_token"):
        user = r.json().get("user") or {}
        meta = user.get("user_metadata") or {}
        name = user.get("email") or meta.get("user_name") or meta.get("name") or "GitHub user"
        print(f"oauth callback: token exchange ok, storing session (pkce store size={len(store)})", flush=True)
        _store_session(r.json(), name)
        st.rerun()
    else:
        snippet = (r.text or "")[:160].replace("\n", " ")
        print(f"oauth callback failed: token exchange status={r.status_code} body={snippet!r}", flush=True)
        st.session_state.auth_error = "Sign-in failed. Please try again."


def _store_session(data: dict, email: str) -> None:
    st.session_state.sb = {
        "access_token": data["access_token"],
        "refresh_token": data["refresh_token"],
        "expires_at": time.time() + int(data.get("expires_in", 3600)) - 60,
        "email": email,
    }


def current_token() -> str | None:
    """A valid access token for the signed-in user, refreshing it when close to expiry."""
    sb = st.session_state.get("sb")
    if not sb:
        return None
    if time.time() > sb["expires_at"]:
        r = _sb("token?grant_type=refresh_token", {"refresh_token": sb["refresh_token"]})
        if r.status_code == 200:
            _store_session(r.json(), sb["email"])
        else:
            del st.session_state.sb
            return None
    return st.session_state.sb["access_token"]


def api_get(path: str, **params: Any) -> requests.Response:
    """Uncached GET against the current workspace (for one-shot interactions)."""
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    return requests.get(f"{API}{path}", params={**params, "dataset": DS}, headers=headers, timeout=120)


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


if AUTH_ON:
    _handle_oauth_callback()

st.title("Vantara Customer Intelligence")
st.caption(
    "Churn risk, 90-day value and segments for the Online Retail II customer base. "
    "Public UK retail data used for training and demonstration, not live business data."
)
if not api_up():
    with st.spinner(
        "Waking up the backend. Free hosting sleeps after inactivity, "
        "so the first load can take about a minute. No need to refresh."
    ):
        ready = wait_for_api()
    if ready:
        st.rerun()
    st.error(
        f"The API at {API} is still not reachable after a few minutes. "
        "It may be restarting; press Retry or reload the page shortly."
    )
    if st.button("Retry"):
        st.rerun()
    st.stop()

st.sidebar.title("Account")
if not AUTH_ON:
    st.sidebar.caption("Sign-in is not configured on this deployment.")
elif "sb" not in st.session_state:
    _ae = st.session_state.pop("auth_error", None)
    if _ae:
        st.sidebar.error(_ae)
    if DASHBOARD_URL:
        st.sidebar.link_button(
            "Continue with GitHub", _oauth_url("github"), type="primary", use_container_width=True
        )
        st.sidebar.link_button(
            "Continue with Google",
            _oauth_url("google"),
            use_container_width=True,
            disabled=True,
            help="Google sign-in is in limited testing - please use GitHub for now.",
        )
        st.sidebar.caption("Google sign-in is in limited testing - please use GitHub for now.")
        st.sidebar.caption("Uses your GitHub account. We never see or store your password.")
    if EMAIL_AUTH_ON:
        _form_parent = st.sidebar.expander("Use email instead") if DASHBOARD_URL else st.sidebar
    else:
        _form_parent = None
    if _form_parent is None:
        do_in = do_up = False
        email = password = ""
    else:
        with _form_parent.form("signin"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            in_, up_ = st.columns(2)
            do_in = in_.form_submit_button("Sign in")
            do_up = up_.form_submit_button("Create account")
    if do_in or do_up:
        if not email or len(password) < 8:
            st.sidebar.error("Enter an email and a password of at least 8 characters.")
        else:
            r = _sb("token?grant_type=password", {"email": email, "password": password}) if do_in else _sb(
                "signup", {"email": email, "password": password}
            )
            if r.status_code == 200 and r.json().get("access_token"):
                _store_session(r.json(), email)
                st.rerun()
            elif do_up and r.status_code == 200:
                st.sidebar.success("Account created. Check your email to confirm it, then sign in.")
            else:
                st.sidebar.error(r.json().get("msg") or r.json().get("error_description") or "Sign-in failed.")
else:
    st.sidebar.write(f"Signed in as **{st.session_state.sb['email']}**")
    if st.sidebar.button("Sign out"):
        try:
            requests.post(
                f"{SUPABASE_URL}/auth/v1/logout",
                headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {st.session_state.sb['access_token']}"},
                timeout=10,
            )
        except Exception:  # noqa: BLE001 - local sign-out must still complete
            pass
        del st.session_state.sb
        st.cache_data.clear()
        st.rerun()

TOKEN = current_token()
workspaces = {"Sample (public demo)": "sample"}
if TOKEN:
    try:
        for d in get("/datasets", token=TOKEN, dataset="sample"):
            if d["status"] == "ready":
                workspaces[f"{d['name']} ({d['n_customers']:,} customers)"] = d["dataset_id"]
    except Exception:  # noqa: BLE001
        st.sidebar.warning("Could not load your datasets right now.")
DS = workspaces[st.sidebar.selectbox("Workspace", list(workspaces))]
if DS != "sample":
    st.sidebar.caption(
        "Your private workspace. Scores come from the demo models trained on the public UK retail sample "
        "(2009-2011); treat them as illustrative, not as business advice."
    )

view = st.sidebar.radio(
    "View",
    ["Segments", "Churn leaderboard", "Revenue trends", "Customer explanation", "Batch scoring", "Model report"]
    + (["My data"] if TOKEN else []),
)

if view == "Segments":
    st.subheader("Customer segments")
    c1, c2, c3 = st.columns(3)
    customers = pd.DataFrame(get("/customers", token=TOKEN, dataset=DS, limit=6000))
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
    prof = pd.DataFrame(get("/segments", token=TOKEN, dataset=DS))
    st.markdown("**Segment profiles**")
    st.dataframe(prof.drop(columns=["segment_id"]).round(2), use_container_width=True, hide_index=True)
    if DS == "sample":
        st.caption(
            "predicted_* columns are model outputs. observed_* columns are what actually happened in the 90 days "
            "after the cutoff, shown for model evaluation only; they would not be known when scoring a live customer."
        )

elif view == "Churn leaderboard":
    st.subheader("Who to call first: biggest value at risk")
    n = st.slider("Customers to show", 10, 100, 25)
    lb = pd.DataFrame(get("/customers/leaderboard", token=TOKEN, dataset=DS, n=n))
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
    t = get("/revenue/trend", token=TOKEN, dataset=DS)
    h = pd.DataFrame(t["history"]).assign(kind="actual")
    f = pd.DataFrame(t["forecast"]).assign(kind="forecast")
    st.plotly_chart(
        px.line(pd.concat([h, f]), x="month", y="revenue", color="kind", markers=True), use_container_width=True
    )
    st.caption(t["method"])

elif view == "Customer explanation":
    st.subheader("Why is this customer flagged?")
    if DS == "sample":
        cid = str(st.number_input("Customer ID", min_value=0, value=12347, step=1))
    else:
        cid = st.text_input("Customer ID", value="", help="Copy an ID from the Segments or leaderboard view.").strip()
    if st.button("Explain") and cid:
        r = api_get(f"/customers/{cid}")
        if r.status_code == 404:
            st.warning("No customer with that ID in this workspace. Copy one from the leaderboard.")
        else:
            cust = r.json()
            ex = api_get(f"/customers/{cid}/explain").json()
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
            lm = api_get(f"/customers/{cid}/explain/lime").json()
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
            rec = api_get(f"/customers/{cid}/recommendations", n=5)
            if rec.status_code == 200:
                st.markdown("**Recommended products** (based on what similar customers also bought)")
                st.dataframe(pd.DataFrame(rec.json()["recommendations"]), hide_index=True, use_container_width=True)
            if cust["data_split"] == "train":
                st.caption("This customer was in the model's training data, so its score is an in-sample score.")
            elif cust["data_split"] == "user":
                st.caption(
                    "Scored by the demo models trained on the public UK retail sample, applied to your data. "
                    "Treat the scores as illustrative."
                )

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

elif view == "My data":
    st.subheader("Your private datasets")
    st.write(
        "Upload a transaction CSV with the same columns as the sample data: "
        "`Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country`. "
        "Only you can see what you upload. Limits: 5 MB / 50,000 rows per file, at least 20 customers and "
        "30 days of history, up to 3 datasets per account."
    )
    with st.form("upload"):
        nm = st.text_input("Dataset name", value="My dataset")
        up = st.file_uploader("Transactions CSV", type="csv")
        go = st.form_submit_button("Upload and score")
    if go and up is not None:
        headers = {"Authorization": f"Bearer {TOKEN}"}
        with st.spinner("Cleaning, scoring and storing your data. This can take a minute on the free tier."):
            r = requests.post(
                f"{API}/datasets",
                files={"file": (up.name, up.getvalue(), "text/csv")},
                data={"name": nm},
                headers=headers,
                timeout=600,
            )
        if r.status_code == 201:
            st.cache_data.clear()
            d = r.json()["dataset"]
            st.success(
                f"Ready: {d['n_customers']:,} customers from {d['n_rows']:,} rows "
                f"({d['first_date']} to {d['last_date']}). Pick it in the Workspace selector on the left."
            )
        else:
            st.error(r.json().get("detail", r.text))
    st.markdown("**Your datasets**")
    try:
        mine = get("/datasets", token=TOKEN, dataset="sample")
    except Exception as exc:  # noqa: BLE001
        mine = []
        st.warning(f"Could not load your datasets: {exc}")
    if not mine:
        st.caption("No datasets yet. Upload your first CSV above and it shows up here once scoring finishes.")
    for d in mine:
        c1, c2, c3 = st.columns([4, 2, 1])
        label = d["name"] if d["status"] == "ready" else f"{d['name']} ({d['status']}: {d.get('error') or ''})"
        c1.write(label)
        c2.write(f"{d['n_customers']:,} customers" if d["status"] == "ready" else "-")
        if c3.button("Delete", key=f"del_{d['dataset_id']}"):
            requests.delete(
                f"{API}/datasets/{d['dataset_id']}", headers={"Authorization": f"Bearer {TOKEN}"}, timeout=30
            )
            st.cache_data.clear()
            st.rerun()

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
