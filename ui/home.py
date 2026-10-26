"""Landing page content and navigation actions."""
from __future__ import annotations

import streamlit as st

from ui.navigation import (
    PAGE_APPLIED_THEORY,
    PAGE_ENGINE_DESIGN,
    PAGE_PRICING_LAB,
)


def _set_current_page(page_name: str) -> None:
    st.session_state.current_page = page_name


def _open_engine_design() -> None:
    _set_current_page(PAGE_ENGINE_DESIGN)


def render_home() -> None:
    """Render the landing page hero, learning links, and contact footer."""
    st.markdown('<div class="landing-page">', unsafe_allow_html=True)

    hero_copy, hero_about = st.columns(
        [1.1, 0.9], gap="large", vertical_alignment="top"
    )
    with hero_copy:
        st.markdown(
            """
            <div class="landing-hero">
                <div class="landing-eyebrow">Quantitative derivatives · Research & tools</div>
                <h1 class="landing-title">A derivative pricing interactive tool</h1>
                <p class="landing-description">
                    Explore how contracts, market assumptions and numerical methods
                    come together to shape a derivative’s value.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Open the pricing lab",
            key="landing_hero_lab",
            type="primary",
            use_container_width=True,
            on_click=_set_current_page,
            args=(PAGE_PRICING_LAB,),
        )
        st.markdown(
            '<div class="landing-meta" style="margin-top:28px;">'
            "Monte Carlo · Binomial trees · Finite differences</div>",
            unsafe_allow_html=True,
        )

    with hero_about:
        with st.container(key="landing-suggestions"):
            st.markdown(
                '<div class="landing-learning-title">'
                "Learning · Applied theory & implementation notes</div>",
                unsafe_allow_html=True,
            )
            with st.container(key="landing-implementation"):
                st.markdown(
                    '<div class="landing-submenu-heading">Engine design</div>',
                    unsafe_allow_html=True,
                )
                with st.container(key="landing-design-row"):
                    title_column, action_column = st.columns(
                        [3.4, 1], vertical_alignment="center", gap="small"
                    )
                    with title_column:
                        st.markdown(
                            '<div class="landing-note-label">'
                            "Inside the pricing engine</div>",
                            unsafe_allow_html=True,
                        )
                    with action_column:
                        st.button(
                            "Go to  →",
                            key="landing-design-link",
                            type="tertiary",
                            use_container_width=True,
                            on_click=_open_engine_design,
                        )

            with st.container(key="landing-methods"):
                st.markdown(
                    '<div class="landing-submenu-heading">'
                    "Applied theory · Work in progress</div>",
                    unsafe_allow_html=True,
                )
                with st.container(key="landing-note-row-intuition"):
                    title_column, action_column = st.columns(
                        [3.4, 1], vertical_alignment="center", gap="small"
                    )
                    with title_column:
                        st.markdown(
                            '<div class="landing-note-label">American calls: '
                            "To exercise, or not to exercise, that is the question"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                    with action_column:
                        st.button(
                            "Read article  →",
                            key="landing-suggestion-intuition",
                            type="tertiary",
                            use_container_width=True,
                            on_click=_set_current_page,
                            args=(PAGE_APPLIED_THEORY,),
                        )

                with st.container(key="landing-note-row-solver"):
                    title_column, action_column = st.columns(
                        [3.4, 1], vertical_alignment="center", gap="small"
                    )
                    with title_column:
                        st.markdown(
                            '<div class="landing-note-label">'
                            "Rannacher initialization: fixing oscillations in "
                            "Crank-Nicolson</div>",
                            unsafe_allow_html=True,
                        )
                    with action_column:
                        st.button(
                            "Coming soon",
                            key="landing-suggestion-solver",
                            type="tertiary",
                            use_container_width=True,
                            disabled=True,
                        )

                with st.container(key="landing-note-row-monte-carlo"):
                    title_column, action_column = st.columns(
                        [3.4, 1], vertical_alignment="center", gap="small"
                    )
                    with title_column:
                        st.markdown(
                            '<div class="landing-note-label">'
                            "Hull-White model: Binomial trees for Asian payoffs"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                    with action_column:
                        st.button(
                            "Coming soon",
                            key="landing-suggestion-monte-carlo",
                            type="tertiary",
                            use_container_width=True,
                            disabled=True,
                        )

    st.markdown(
        '<div class="landing-footer">'
        '<div class="landing-footer-attribution">'
        "Developed by Gabriel Meisler · Pricing engine is built with Python "
        "and relies mostly on NumPy · The Streamlit frontend was built with "
        "significant help from Copilot."
        "</div>"
        '<div class="landing-contact">'
        '<div class="landing-contact-title">Contact me</div>'
        '<div class="landing-footer-links">'
        '<a href="https://www.linkedin.com/in/gabriel-meisler/" '
        'target="_blank" rel="noopener noreferrer">LinkedIn profile</a>'
        '<span class="landing-contact-email">meisler1999@gmail.com</span>'
        "</div></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
