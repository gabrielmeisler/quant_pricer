"""Applied theory articles and interactive pricing demonstrations."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import streamlit as st

from pricing.contract import DerivativeContract
from pricing.exercise_rights import AmericanExercise, EuropeanExercise
from pricing.engines.base import PricingResult
from pricing.engines.monte_carlo import MonteCarloEngine
from pricing.markets import MarketData
from pricing.payoff import VanillaPayoff
from pricing.stochastic_process import GeometricBrownianMotion

ARTICLE_PATH = Path(__file__).resolve().parent / "articles" / "american_dividend.md"
ARTICLE_LSM_POLY_DEGREE = 4
PRICING_DEMO_MARKERS = (
    "<!-- Monte Carlo pricing demo goes here -->",
    "<!-- Dividend call pricing demo goes here -->",
    "<!-- Put pricing demo goes here -->",
)
THEORY_ARTICLES = (
    (
        "american-call",
        "American calls: To exercise, or not to exercise, that is the question",
        "Options & exercise",
        ARTICLE_PATH,
        True,
    ),
    (
        "rannacher",
        "Rannacher initialization: fixing oscillations in Crank-Nicolson",
        None,
        None,
        False,
    ),
    (
        "hull-white",
        "Hull-White model: Binomial trees for Asian payoffs",
        None,
        None,
        False,
    ),
)


def price_american_option(
    exercise_style: Literal["European", "American"],
    option_kind: Literal["Call", "Put"],
    dividend_yield: float = 0.0,
    spot: float = 100.0,
) -> PricingResult:
    maturity = 1.0
    exercise_rights = (
        EuropeanExercise(maturity=maturity)
        if exercise_style == "European"
        else AmericanExercise(maturity=maturity)
    )
    contract = DerivativeContract(
        payoff=VanillaPayoff(strike=100.0, is_call=option_kind == "Call"),
        exercise_rights=exercise_rights,
    )
    engine = MonteCarloEngine(
        n_paths=100_000,
        n_steps=100,
        seed=42,
        poly_degree=ARTICLE_LSM_POLY_DEGREE,
    )
    return engine.price(
        derivative=contract,
        process=GeometricBrownianMotion(volatility=0.20),
        initial_spot=spot,
        market_data=MarketData(
            risk_free_rate=0.05,
            dividend_yield=dividend_yield,
        ),
    )


def _store_option_price(
    exercise_style: Literal["European", "American"],
    result_key: str,
    option_kind: Literal["Call", "Put"],
    dividend_yield: float,
    spot: float,
) -> None:
    st.session_state[result_key] = price_american_option(
        exercise_style,
        option_kind,
        dividend_yield,
        spot,
    )


def _render_pricing_card(
    exercise_style: Literal["European", "American"],
    key: str,
    option_kind: Literal["Call", "Put"] = "Call",
    dividend_yield: float = 0.0,
    spot: float = 100.0,
) -> None:
    with st.container(key=key):
        summary, pricing = st.columns(
            [1.55, 1], gap="large", vertical_alignment="center"
        )
        with summary:
            st.markdown(
                f"""
                <div class="theory-contract-summary">
                  <div class="theory-contract-title theory-{exercise_style.lower()}-title">
                    {exercise_style} Vanilla {option_kind}
                  </div>
                  <div class="theory-contract-terms">
                    <span><small>Payoff</small>Vanilla</span>
                    <span class="theory-exercise-term theory-{exercise_style.lower()}-accent">
                      <small>Exercise</small>{exercise_style}
                    </span>
                    <span><small>Spot</small>${spot:.0f}</span>
                    <span><small>Strike</small>$100</span>
                    <span><small>Maturity</small>1 year</span>
                    <span><small>Volatility</small>20%</span>
                    <span><small>Risk-free rate</small>5%</span>
                    <span class="{'theory-dividend-highlight' if dividend_yield else ''}">
                      <small>Dividend yield</small>{dividend_yield:.0%}
                    </span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with pricing:
            result_key = f"{key}-result-degree-{ARTICLE_LSM_POLY_DEGREE}"
            result = st.session_state.get(result_key)
            if result is None:
                with st.container(key=f"{key}-button-slot"):
                    st.write("")
                    st.write("")
                    st.button(
                        "Run Monte Carlo",
                        key=f"{key}-price",
                        type="secondary",
                        use_container_width=True,
                        on_click=_store_option_price,
                        args=(
                            exercise_style,
                            result_key,
                            option_kind,
                            dividend_yield,
                            spot,
                        ),
                    )
            else:
                interval = result.ci95
                interval_markup = (
                    f'<div class="theory-price-interval">95% interval · '
                    f'${interval[0]:.4f} – ${interval[1]:.4f}</div>'
                    if interval is not None
                    else ""
                )
                error_markup = (
                    f'<div class="theory-price-error">Standard error · '
                    f'${result.std_error:.4f}</div>'
                    if result.std_error is not None
                    else ""
                )
                st.markdown(
                    f"""
                    <div class="theory-price-result">
                      <div class="theory-price-label">Estimated value</div>
                      <div class="theory-price-value">${result.price:.4f}</div>
                      {error_markup}
                      {interval_markup}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def _select_theory_article(article_id: str) -> None:
    st.session_state.applied_theory_article = article_id


def _render_article_list(*, compact: bool) -> None:
    for article_id, title, topic, article_path, available in THEORY_ARTICLES:
        available = available and article_path is not None and article_path.is_file()
        with st.container(key=f"theory-article-row-{article_id}"):
            title_column, status_column = st.columns(
                [6, 1], gap="small", vertical_alignment="center"
            )
            with title_column:
                st.button(
                    title,
                    key=f"theory-article-select-{article_id}-{compact}",
                    type="tertiary",
                    width="stretch",
                    disabled=not available,
                    on_click=_select_theory_article if available else None,
                    args=(article_id,) if available else (),
                )
            with status_column:
                    status_class = (
                        "theory-article-status"
                        if available
                        else "theory-article-status theory-article-coming"
                    )
                    st.markdown(
                        f'<span class="{status_class}">'
                        f'{topic if available else "Coming soon"}'
                        "</span>",
                        unsafe_allow_html=True,
                )


def render_applied_theory() -> None:
    """Render the article directory or the selected article."""
    selected_article_id = st.session_state.get("applied_theory_article")
    selected_article = next(
        (entry for entry in THEORY_ARTICLES if entry[0] == selected_article_id),
        None,
    )
    if (
        selected_article is None
        or not selected_article[4]
        or selected_article[3] is None
    ):
        st.session_state.pop("applied_theory_article", None)
        with st.container(key="theory-article-directory"):
            st.markdown(
                '<div class="theory-directory-title">Available & Planned Articles</div>',
                unsafe_allow_html=True,
            )
            _render_article_list(compact=False)
        return

    article_path = selected_article[3]
    article = article_path.read_text(encoding="utf-8")
    if any(article.count(marker) != 1 for marker in PRICING_DEMO_MARKERS):
        raise ValueError(
            f"{article_path} must contain each pricing demo marker exactly once."
        )
    marker_positions = [article.index(marker) for marker in PRICING_DEMO_MARKERS]
    if marker_positions != sorted(marker_positions):
        raise ValueError(f"{ARTICLE_PATH} has pricing demo markers in the wrong order.")

    article_parts = []
    remaining_article = article
    for marker in PRICING_DEMO_MARKERS:
        before_marker, remaining_article = remaining_article.split(marker, maxsplit=1)
        article_parts.append(before_marker)
    article_parts.append(remaining_article)
    with st.container(key="applied-theory-page"):
        with st.expander(
            f"Browse articles · {len(THEORY_ARTICLES)} articles",
            expanded=False,
        ):
            _render_article_list(compact=True)
        st.markdown(
            f'<div class="theory-article-topic">{selected_article[2]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(article_parts[0].strip())
        _render_pricing_card("European", "theory-european-call")
        _render_pricing_card("American", "theory-american-call")
        st.caption(
            "Both runs use the same Monte Carlo settings: 100,000 paths, "
            "100 time steps, a degree-4 LSM polynomial basis, seed 42."
        )
        st.markdown(article_parts[1].strip())

        _render_pricing_card(
            "European",
            "theory-dividend-european-call",
            dividend_yield=0.04,
            spot=150.0,
        )
        _render_pricing_card(
            "American",
            "theory-dividend-american-call",
            dividend_yield=0.04,
            spot=150.0,
        )
        st.caption(
            "Both runs use a 4% continuous dividend yield and a $150 spot "
            "to illustrate a deep in-the-money call; all other inputs match "
            "the opening example."
        )
        st.markdown(article_parts[2].strip())

        _render_pricing_card(
            "European",
            "theory-put-european-put",
            option_kind="Put",
        )
        _render_pricing_card(
            "American",
            "theory-put-american-put",
            option_kind="Put",
        )
        st.caption(
            "Both put runs use the same assumptions as the opening example, "
            "with a vanilla put payoff."
        )
        st.markdown(article_parts[3].strip())
