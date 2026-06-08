"""Streamlit frontend for the CartWise recommendation API."""

from __future__ import annotations

import os
from html import escape
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
TITLE_PREVIEW_CHARS = 120
EVIDENCE_PREVIEW_CHARS = 520


def main() -> None:
    st.set_page_config(
        page_title="CartWise",
        layout="wide",
    )
    _inject_styles()

    api_base_url, user_id = render_sidebar()
    client = CartWiseApiClient(api_base_url)

    readiness = client.check_ready()
    render_backend_status(readiness, in_sidebar=False)

    render_hero()

    st.markdown('<div class="cw-search-card">', unsafe_allow_html=True)
    with st.form("recommendation-search"):
        search_col, controls_col = st.columns([5, 1.45], vertical_alignment="bottom")
        with search_col:
            query = st.text_area(
                "What are you shopping for?",
                placeholder=QUERY_PLACEHOLDER,
                height=88,
            )
        with controls_col:
            top_k = st.number_input("Top K", min_value=1, max_value=50, value=3, step=1)
            submitted = st.form_submit_button("Find recommendations", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="cw-content-spacer"></div>', unsafe_allow_html=True)

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
        if in_sidebar:
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


def render_hero() -> None:
    st.markdown(
        """
        <section class="cw-hero">
          <div class="cw-brand-kicker">CartWise</div>
          <h1>Find music gear that fits how you play.</h1>
          <p>AI-powered music gear recommendations from product data and reviews.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_result(result: ApiResult, *, original_query: str) -> None:
    if not result.ok:
        render_api_error(result.error)
        return

    payload = result.data or {}
    results = payload.get("results") or []
    latency = payload.get("latency_ms", result.elapsed_ms)
    summary = f"{len(results)} recommendations · {format_latency(latency)}"
    st.markdown(
        f'<div class="cw-summary-bar">{escape(summary)}</div>',
        unsafe_allow_html=True,
    )

    search_query = payload.get("search_query")
    if search_query and search_query != original_query:
        with st.expander("Search query used", expanded=False):
            st.write(search_query)

    if not results:
        st.info("No recommendations found. Try a broader query or remove constraints.")
    for item in results:
        render_result_card(item)

    render_developer_details(payload)


def render_result_card(item: dict[str, Any]) -> None:
    title = clean_text(item.get("title"), "Untitled product")
    title_preview, title_was_truncated = preview_text(title, TITLE_PREVIEW_CHARS)
    rank = item.get("rank", "")
    brand = clean_text(item.get("brand"), "Unknown brand")
    price = format_price(item.get("price"))
    reason = clean_text(item.get("reason"), "No recommendation reason returned.")
    potential_cons = clean_text(item.get("potential_cons"), "No potential cons returned.")
    category = infer_placeholder_category(title)
    evidence_items = item.get("evidence") or []
    source_count = len(item.get("sources") or [])
    fallback = bool(item.get("fallback"))

    with st.container():
        st.markdown('<div class="cw-card">', unsafe_allow_html=True)
        visual_col, details_col = st.columns([1.15, 3], gap="large", vertical_alignment="top")
        with visual_col:
            render_placeholder_visual(category)
        with details_col:
            st.markdown(
                f"""
                <div class="cw-product-head">
                  <span class="cw-rank-badge">#{escape(str(rank))}</span>
                  <span class="cw-match-pill">{source_count or "No"} match signals</span>
                  {('<span class="cw-muted-pill">Fallback explanation</span>' if fallback else '')}
                </div>
                <h2 class="cw-product-title">{escape(title_preview)}</h2>
                <div class="cw-meta-row">
                  <span>{escape(brand)}</span>
                  <span>{escape(price)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if title_was_truncated:
                with st.expander("Full product title", expanded=False):
                    st.write(title)

        left, right = st.columns(2)
        with left:
            render_text_panel("Recommendation reason", reason, "reason")
        with right:
            render_text_panel("Potential cons", potential_cons, "cons")

        with st.expander("Review evidence"):
            render_evidence(evidence_items)

        with st.expander("Retrieval details"):
            sources = item.get("sources") or []
            if sources:
                st.markdown(
                    " ".join(
                        f'<span class="cw-source-pill">{escape(str(source))}</span>'
                        for source in sources
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.write("Sources: None")
            st.write("Fusion score:", item.get("fusion_score"))
            st.write("Fallback explanation:", bool(item.get("fallback")))
            st.write("Source ranks")
            st.json(item.get("source_ranks") or {})
            st.write("Source scores")
            st.json(item.get("source_scores") or {})
        st.markdown("</div>", unsafe_allow_html=True)


def render_evidence(evidence_items: list[dict[str, Any]]) -> None:
    if not evidence_items:
        st.info("No review evidence returned for this item.")
        return

    for index, evidence in enumerate(evidence_items, start=1):
        rating = evidence.get("rating")
        score = evidence.get("score")
        evidence_id = evidence.get("review_id") or evidence.get("chunk_id") or f"evidence-{index}"
        text = clean_text(evidence.get("chunk_text") or evidence.get("text"), "")
        rating_label = f"Rating: {rating}" if rating is not None else "Rating unavailable"
        score_label = f"Score: {score:.3f}" if isinstance(score, int | float) else "Score unavailable"
        preview, was_truncated = preview_text(text or "No evidence text returned.", EVIDENCE_PREVIEW_CHARS)
        st.markdown(
            f"""
            <div class="cw-evidence-card">
              <div class="cw-evidence-meta">
                <span>{escape(rating_label)}</span>
                <span>{escape(score_label)}</span>
                <span>{escape(str(evidence_id))}</span>
              </div>
              <div class="cw-evidence-text">{escape(preview)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if was_truncated:
            with st.expander("Show full evidence text", expanded=False):
                st.write(text)


def render_placeholder_visual(category: tuple[str, str]) -> None:
    emoji, label = category
    st.markdown(
        f"""
        <div class="cw-product-visual">
          <div class="cw-product-emoji">{escape(emoji)}</div>
          <div class="cw-product-category">{escape(label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_text_panel(title: str, body: str, variant: str) -> None:
    st.markdown(
        f"""
        <div class="cw-copy-panel cw-copy-{escape(variant)}">
          <div class="cw-copy-title">{escape(title)}</div>
          <div class="cw-copy-body">{escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_developer_details(payload: dict[str, Any]) -> None:
    with st.expander("Developer details", expanded=False):
        diagnostics = payload.get("diagnostics") or []
        if diagnostics:
            st.markdown("**Diagnostics**")
            for diagnostic in diagnostics:
                component = diagnostic.get("component", "system")
                message = diagnostic.get("message", "")
                error_type = diagnostic.get("error_type", "note")
                st.write(f"**{component}** - {error_type}: {message}")

        debug_payload = {
            "search_query": payload.get("search_query"),
            "latency_ms": payload.get("latency_ms"),
            "applied_constraints": payload.get("applied_constraints"),
            "diagnostics": diagnostics,
            "raw_response": payload,
        }
        st.json(debug_payload)


def clean_text(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    if not text or text.lower() == "none":
        return fallback
    return text


def preview_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    trimmed = text[: max_chars - 1].rstrip()
    return f"{trimmed}…", True


def infer_placeholder_category(title: str) -> tuple[str, str]:
    normalized = title.lower()
    checks = [
        (("guitar", "bass", "ukulele", "mandolin"), ("🎸", "Guitar")),
        (("drum", "cymbal", "percussion", "snare"), ("🥁", "Drum")),
        (("microphone", "mic", "vocal"), ("🎙️", "Microphone")),
        (("keyboard", "piano", "synth", "midi"), ("🎹", "Keyboard")),
        (("cable", "strap", "stand", "case", "bag", "pick", "capo", "tuner"), ("🎛️", "Accessory")),
    ]
    for keywords, category in checks:
        if any(keyword in normalized for keyword in keywords):
            return category
    return "✨", "CartWise pick"


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
        :root {
            --cw-ink: #17202a;
            --cw-muted: #637083;
            --cw-line: #dce4ef;
            --cw-soft: #f7f9fc;
            --cw-primary: #2f6f73;
            --cw-primary-dark: #24585b;
            --cw-blue-soft: #edf6ff;
            --cw-green-soft: #eef8f3;
            --cw-orange-soft: #fff5ea;
        }
        .block-container {
            max-width: 1060px;
            padding-top: 2.1rem;
            padding-bottom: 4rem;
        }
        h1, h2, h3, p {
            letter-spacing: 0;
        }
        .cw-hero {
            margin: 0 0 0.95rem 0;
            padding: 0.25rem 0 0.45rem 0;
        }
        .cw-brand-kicker {
            color: var(--cw-primary);
            font-size: 0.95rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .cw-hero h1 {
            color: var(--cw-ink);
            font-size: 2.55rem;
            line-height: 1.06;
            font-weight: 800;
            margin: 0;
            max-width: 760px;
        }
        .cw-hero p {
            color: var(--cw-muted);
            font-size: 1rem;
            line-height: 1.45;
            margin: 0.55rem 0 0 0;
            max-width: 620px;
        }
        .cw-search-card {
            border: 1px solid var(--cw-line);
            border-radius: 16px;
            background: linear-gradient(180deg, #ffffff 0%, #f7fafc 100%);
            box-shadow: 0 14px 34px rgba(25, 42, 62, 0.08);
            padding: 1.05rem 1.15rem 0.85rem;
            margin: 0.8rem 0 1.1rem;
        }
        .cw-search-card label {
            color: #263342 !important;
            font-weight: 700 !important;
        }
        .cw-search-card textarea,
        .cw-search-card input {
            border-color: #cbd6e2 !important;
            border-radius: 12px !important;
        }
        div.stButton > button[kind="primary"] {
            background: var(--cw-primary);
            border-color: var(--cw-primary);
            color: #ffffff;
            border-radius: 999px;
            font-weight: 760;
            min-height: 2.75rem;
            box-shadow: 0 8px 18px rgba(47, 111, 115, 0.22);
        }
        div.stButton > button[kind="primary"]:hover {
            background: var(--cw-primary-dark);
            border-color: var(--cw-primary-dark);
            color: #ffffff;
        }
        .cw-content-spacer {
            height: 0.1rem;
        }
        .cw-summary-bar {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            color: #284457;
            background: #edf5f7;
            border: 1px solid #d5e6ea;
            border-radius: 999px;
            padding: 0.45rem 0.75rem;
            margin: 0.4rem 0 0.75rem 0;
            font-size: 0.94rem;
            font-weight: 700;
        }
        .cw-card {
            border: 1px solid var(--cw-line);
            border-radius: 16px;
            padding: 1.15rem 1.2rem 1.05rem;
            margin: 1rem 0 1.25rem;
            background: #ffffff;
            box-shadow: 0 12px 30px rgba(29, 44, 60, 0.08);
        }
        .cw-product-visual {
            min-height: 172px;
            border-radius: 14px;
            border: 1px solid #d8e4ec;
            background:
                radial-gradient(circle at 25% 20%, rgba(255,255,255,0.65), transparent 26%),
                linear-gradient(135deg, #e8f3f2 0%, #f7f3e8 58%, #edf0ff 100%);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            overflow: hidden;
        }
        .cw-product-emoji {
            font-size: 3.2rem;
            line-height: 1;
            margin-bottom: 0.65rem;
        }
        .cw-product-category {
            color: #2e4853;
            font-size: 0.78rem;
            text-transform: uppercase;
            font-weight: 800;
        }
        .cw-product-head {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            align-items: center;
            margin-bottom: 0.45rem;
        }
        .cw-rank-badge,
        .cw-match-pill,
        .cw-muted-pill,
        .cw-source-pill {
            display: inline-flex;
            align-items: center;
            width: fit-content;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 800;
            line-height: 1;
            padding: 0.36rem 0.58rem;
            white-space: nowrap;
        }
        .cw-rank-badge {
            color: #ffffff;
            background: var(--cw-primary);
        }
        .cw-match-pill {
            color: #31545c;
            background: #e8f3f4;
            border: 1px solid #d4e5e8;
        }
        .cw-muted-pill,
        .cw-source-pill {
            color: #5c6775;
            background: #f2f5f8;
            border: 1px solid #e0e7ee;
        }
        .cw-source-pill {
            margin: 0 0.35rem 0.35rem 0;
        }
        .cw-product-title {
            color: var(--cw-ink);
            font-size: 1.32rem;
            line-height: 1.24;
            font-weight: 790;
            margin: 0.15rem 0 0.5rem 0;
            max-width: 720px;
        }
        .cw-meta-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem 0.8rem;
            color: var(--cw-muted);
            font-size: 0.95rem;
            line-height: 1.35;
            margin-bottom: 0.4rem;
        }
        .cw-meta-row span + span::before {
            content: "•";
            color: #a6b1bd;
            margin-right: 0.8rem;
        }
        .cw-copy-panel {
            min-height: 154px;
            border-radius: 14px;
            border: 1px solid rgba(30, 53, 67, 0.08);
            padding: 0.9rem 0.95rem;
            margin: 0.9rem 0 0.4rem 0;
        }
        .cw-copy-reason {
            background: linear-gradient(180deg, var(--cw-green-soft), #f8fcfb);
        }
        .cw-copy-cons {
            background: linear-gradient(180deg, var(--cw-orange-soft), #fbfaf7);
        }
        .cw-copy-title {
            color: #22313f;
            font-size: 0.82rem;
            font-weight: 850;
            margin-bottom: 0.45rem;
            text-transform: uppercase;
        }
        .cw-copy-body {
            color: #2d3a45;
            font-size: 0.96rem;
            line-height: 1.62;
        }
        .cw-evidence-card {
            border: 1px solid #e0e7ef;
            border-radius: 12px;
            background: #fbfcfe;
            padding: 0.82rem 0.9rem;
            margin: 0.65rem 0;
        }
        .cw-evidence-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-bottom: 0.5rem;
        }
        .cw-evidence-meta span {
            border-radius: 999px;
            background: #eef3f8;
            color: #4a5969;
            font-size: 0.76rem;
            font-weight: 760;
            padding: 0.25rem 0.5rem;
        }
        .cw-evidence-text {
            color: #2f3b48;
            font-size: 0.93rem;
            line-height: 1.58;
        }
        div[data-testid="stExpander"] {
            border-color: #e0e7ef;
            border-radius: 12px;
        }
        @media (max-width: 760px) {
            .block-container {
                padding-top: 1.2rem;
            }
            .cw-hero h1 {
                font-size: 2rem;
            }
            .cw-search-card,
            .cw-card {
                padding: 0.95rem;
                border-radius: 14px;
            }
            .cw-product-visual {
                min-height: 130px;
            }
            .cw-product-title {
                font-size: 1.16rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
