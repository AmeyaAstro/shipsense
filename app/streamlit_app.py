"""ShipSense demo UI: risk prediction, the ops copilot, and the drift summary.

Why a thin client: the UI calls the FastAPI service over HTTP instead of loading the model itself,
so it shows exactly what any other caller of the API gets (one source of truth for predictions).

Run:  make serve   (terminal 1)   then   make demo   (terminal 2, opens http://localhost:8501)
Point at another API with SHIPSENSE_API_URL (e.g. the deployed AWS URL in Week 4).
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st

API_URL = os.getenv("SHIPSENSE_API_URL", "http://127.0.0.1:8000")
DRIFT_SUMMARY = Path(__file__).resolve().parents[1] / "reports" / "drift" / "summary.md"

MODES = ["Standard Class", "Second Class", "First Class", "Same Day"]
PAYMENTS = ["DEBIT", "TRANSFER", "PAYMENT", "CASH"]
MARKETS = ["LATAM", "Europe", "Pacific Asia", "USCA", "Africa"]
SEGMENTS = ["Consumer", "Corporate", "Home Office"]
# Exact values from the training data. Note "Health and Beauty " has a trailing space there; sending it
# without the space would be an unseen category to the model, so we keep values exact and trim for display.
DEPARTMENTS = ["Apparel", "Book Shop", "Discs Shop", "Fan Shop", "Fitness", "Footwear", "Golf",
               "Health and Beauty ", "Outdoors", "Pet Shop", "Technology"]


def call_api(method: str, path: str, **kwargs) -> tuple[dict | None, str | None]:
    """Return (json, None) on success or (None, error message) so the UI never crashes."""
    try:
        r = httpx.request(method, f"{API_URL}{path}", timeout=180, **kwargs)
    except httpx.HTTPError:
        return None, f"Can't reach the API at {API_URL}. Start it with `make serve`."
    if r.status_code != 200:
        return None, f"API returned {r.status_code}: {r.json().get('detail', r.text)}"
    return r.json(), None


def risk_tab() -> None:
    st.subheader("Late-delivery risk for a new order")
    c1, c2, c3 = st.columns(3)
    mode = c1.selectbox("Shipping mode", MODES)
    payment = c1.selectbox("Payment type", PAYMENTS)
    segment = c1.selectbox("Customer segment", SEGMENTS)
    market = c2.selectbox("Market", MARKETS)
    region = c2.text_input("Order region", "Central America")
    country = c2.text_input("Order country", "México")
    dept = c3.selectbox("Main department", DEPARTMENTS, index=3, format_func=str.strip)
    n_items = c3.number_input("Items", 1, 100, 2)
    quantity = c3.number_input("Total quantity", 1, 1000, 3)
    sales = st.number_input("Order value ($)", 0.01, 100000.0, 420.5)
    discount = st.slider("Average discount rate", 0.0, 0.5, 0.1)
    n_depts = st.number_input("Departments in order", 1, 100, 1)
    order_ts = st.text_input("Order time", datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))

    if st.button("Predict risk", type="primary"):
        order = {"order_ts": order_ts, "shipping_mode": mode, "payment_type": payment, "market": market,
                 "order_region": region, "order_country": country, "customer_segment": segment,
                 "main_department": dept, "n_items": int(n_items), "total_quantity": int(quantity),
                 "total_sales": float(sales), "avg_discount_rate": float(discount), "n_departments": int(n_depts)}
        result, error = call_api("POST", "/predict", json=order)
        if error:
            st.error(error)
            return
        p = result["late_probability"]
        st.metric("Late probability", f"{p:.1%}", help=f"flagged high risk at >= {result['threshold']:.3f}")
        st.progress(min(p, 1.0))
        if result["is_late_risk"]:
            st.warning(f"HIGH RISK (threshold {result['threshold']}). See the High-Risk Order Handling policy.")
        else:
            st.success(f"Not flagged (threshold {result['threshold']}).")
        st.caption(f"Model version: {result['model_version']}")


def copilot_tab() -> None:
    st.subheader("Ask the ops copilot")
    st.caption("Answers come from fictional ShipSense policies, the shipment database and the risk model. "
               "Runs on a local LLM, so expect ~10-30 s per answer.")
    examples = ["How much credit does a Consumer customer get if their order is 2 days late?",
                "What happened to order 62489?",
                "Is order 62477 high risk, and what should we do for high-risk orders?"]
    question = st.text_input("Question", examples[0])
    st.caption("Try: " + " · ".join(examples[1:]))
    if st.button("Ask", type="primary"):
        with st.spinner("Thinking (retrieving policies, calling tools)..."):
            result, error = call_api("POST", "/ask", json={"question": question})
        if error:
            st.error(error)
            return
        st.markdown(result["answer"])
        if result["invalid_citations"]:
            st.warning("Cited but never retrieved (made up by the LLM): " + ", ".join(result["invalid_citations"]))
        with st.expander(f"Sources retrieved ({len(result['sources'])})"):
            for s in result["sources"]:
                st.write("- " + s)
        with st.expander(f"Tool calls ({len(result['tool_calls'])}) · {result['steps']} LLM step(s) · "
                         f"{result['llm_model']}"):
            st.json(result["tool_calls"])


def drift_tab() -> None:
    st.subheader("Drift monitoring")
    if DRIFT_SUMMARY.exists():
        st.markdown(DRIFT_SUMMARY.read_text())
    else:
        st.info("No drift summary yet. Run `make drift ARGS=--summary`.")


def main() -> None:
    st.set_page_config(page_title="ShipSense", layout="wide")
    st.title("ShipSense")
    st.caption("Late-delivery risk + logistics ops copilot · portfolio project · policies are fictional")
    health, error = call_api("GET", "/health")
    st.sidebar.write(f"API: `{API_URL}`")
    if health:
        st.sidebar.success(f"Connected · {health['model_version']}")
    else:
        st.sidebar.error(error)
    tab1, tab2, tab3 = st.tabs(["Risk prediction", "Ask the copilot", "Drift"])
    with tab1:
        risk_tab()
    with tab2:
        copilot_tab()
    with tab3:
        drift_tab()


main()
