"""Streamlit frontend for the CartWise recommendation API."""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from cartwise.ui.api_client import (
    DEFAULT_API_BASE_URL,
    ApiError,
    ApiResult,
    CartWiseApiClient,
)

BACKEND_START_COMMAND = "uvicorn cartwise.api.main:app --reload"
QUERY_PLACEHOLDER = (
    'Describe what you need, e.g. "a quiet guitar practice setup for my apartment"'
)


def main() -> None:
    st.set_page_config(
        page_title="CartWise",
        layout="wide",
    )
    _inject_styles()

    api_base_url, user_id = render_sidebar()
    client = CartWiseApiClient(api_base_url)

    st.title("CartWise")
    st.caption("AI-powered music gear recommendations from product data and reviews")

    readiness = client.check_ready()
    render_backend_status(readiness, in_sidebar=False)

    with st.form("recommendation-search"):
        query = st.text_area(
            "What are you shopping for?",
            placeholder=QUERY_PLACEHOLDER,
            height=110,
        )
        top_k = st.number_input("Top K", min_value=1, max_value=50, value=3, step=1)
        submitted = st.form_submit_button("Find recommendations", type="primary")

    if submitted:
        if not query.strip():
            st.warning("Enter a shopping need before requesting recommendations.")
            return

        with st.spinner("Finding products and reading review evidence..."):
            st.session_state["last_recommendation_result"] = client.recommend(
                query.strip(),
                user_id=user_id,
                top_k=int(top_k),
            )
            st.session_state["last_recommendation_query"] = query.strip()

    last_result = st.session_state.get("last_recommendation_result")
    if isinstance(last_result, ApiResult):
        render_recommendation_result(
            last_result,
            original_query=st.session_state.get("last_recommendation_query", ""),
        )


def render_sidebar() -> tuple[str, str | None]:
    default_base_url = os.environ.get("CARTWISE_API_BASE_URL", DEFAULT_API_BASE_URL)
    with st.sidebar:
        st.header("Backend")
        api_base_url = st.text_input("API base URL", value=default_base_url)
        sidebar_client = CartWiseApiClient(api_base_url)
        render_backend_status(sidebar_client.check_ready(), in_sidebar=True)

        with st.expander("Advanced options"):
            user_id = st.text_input("User ID", value="")
    return api_base_url.strip() or DEFAULT_API_BASE_URL, user_id.strip() or None


def render_backend_status(result: ApiResult, *, in_sidebar: bool) -> None:
    target = st.sidebar if in_sidebar else st
    if result.ok:
        target.success("Backend ready")
        return

    target.warning(
        "Backend is not ready. Start FastAPI with:\n\n"
        f"`{BACKEND_START_COMMAND}`"
    )
    if result.error is not None:
        with target.expander("Backend status details"):
            st.write(result.error.message)
            if result.error.detail:
                st.json(result.error.detail)


def render_recommendation_result(result: ApiResult, *, original_query: str) -> None:
    if not result.ok:
        render_api_error(result.error)
        return

    payload = result.data or {}
    results = payload.get("results") or []
    latency = payload.get("latency_ms", result.elapsed_ms)
    st.success(f"Found {len(results)} recommendations in {format_latency(latency)}")

    search_query = payload.get("search_query")
    if search_query and search_query != original_query:
        st.caption(f"Search query used: {search_query}")

    if not results:
        st.info("No recommendations found. Try a broader query or remove constraints.")
    for item in results:
        render_result_card(item)

    diagnostics = payload.get("diagnostics") or []
    if diagnostics:
        with st.expander("System notes"):
            for diagnostic in diagnostics:
                component = diagnostic.get("component", "system")
                message = diagnostic.get("message", "")
                error_type = diagnostic.get("error_type", "note")
                st.write(f"**{component}** - {error_type}: {message}")

    with st.expander("Developer details"):
        st.json(payload)


def render_result_card(item: dict[str, Any]) -> None:
    title = item.get("title") or "Untitled product"
    rank = item.get("rank", "")
    brand = item.get("brand") or "Unknown brand"
    price = format_price(item.get("price"))
    reason = item.get("reason") or "No recommendation reason returned."
    potential_cons = item.get("potential_cons") or "No potential cons returned."

    with st.container():
        st.markdown('<div class="cw-card">', unsafe_allow_html=True)
        st.markdown(f"### #{rank} {title}")
        st.caption(f"{brand} - {price}")

        left, right = st.columns(2)
        with left:
            st.markdown("**Recommendation reason**")
            st.write(reason)
        with right:
            st.markdown("**Potential cons**")
            st.write(potential_cons)

        with st.expander("Review evidence"):
            render_evidence(item.get("evidence") or [])

        with st.expander("Retrieval details"):
            st.write("Sources:", ", ".join(item.get("sources") or []) or "None")
            st.write("Fusion score:", item.get("fusion_score"))
            st.write("Fallback explanation:", bool(item.get("fallback")))
            st.write("Source ranks")
            st.json(item.get("source_ranks") or {})
            st.write("Source scores")
            st.json(item.get("source_scores") or {})
        st.markdown("</div>", unsafe_allow_html=True)


def render_evidence(evidence_items: list[dict[str, Any]]) -> None:
    if not evidence_items:
        st.caption("No review evidence returned for this item.")
        return

    for evidence in evidence_items:
        rating = evidence.get("rating")
        score = evidence.get("score")
        text = evidence.get("chunk_text") or evidence.get("text") or ""
        rating_label = f"Rating: {rating}" if rating is not None else "Rating unavailable"
        score_label = f"Score: {score:.3f}" if isinstance(score, int | float) else "Score unavailable"
        st.markdown(f"**{rating_label} - {score_label}**")
        st.write(text or "No evidence text returned.")


def render_api_error(error: ApiError | None) -> None:
    if error is None:
        st.error("The request failed, but no error details were returned.")
        return
    st.error(error.message)
    if error.status_code == 422:
        st.info("Check that the query is not blank and Top K is between 1 and 50.")
    elif error.status_code == 503:
        st.info("Check FastAPI, Qdrant, model files, indexes, and the LLM key.")
    elif error.status_code and error.status_code >= 500:
        st.info("Check the backend logs for the internal failure.")

    if error.detail:
        with st.expander("Error details"):
            st.json(error.detail)


def format_price(price: Any) -> str:
    if price is None:
        return "Price unavailable"
    try:
        return f"${float(price):,.2f}"
    except (TypeError, ValueError):
        return "Price unavailable"


def format_latency(latency_ms: Any) -> str:
    try:
        seconds = float(latency_ms) / 1000
    except (TypeError, ValueError):
        return "unknown time"
    return f"{seconds:.1f}s"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .cw-card {
            border: 1px solid #d9dee7;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin: 1rem 0;
            background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
