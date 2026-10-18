"""Streamlit entry point for the QuantPricer application."""

from __future__ import annotations

import streamlit as st

from ui.engine_design import render_engine_design
from ui.pricing_lab import render_pricing_lab
from ui.theme import apply_theme


st.set_page_config(
    page_title="QuantPricer | Financial Derivatives Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light"

apply_theme()

if "current_page" not in st.session_state:
    st.session_state.current_page = "Landing Page"


def set_page(page_name: str):
    st.session_state.current_page = page_name


def set_theme_mode():
    st.session_state.theme_mode = st.session_state["app-nav-theme"]


def open_engine_design():
    st.session_state.current_page = "Dev & Architecture"


# Shared navigation and theme controls are rendered on every page.
with st.container(key="app-top-nav"):
    brand, home, lab, engine, notes, theme = st.columns(
        [2.5, 0.72, 1.0, 1.2, 1.1, 1.15],
        vertical_alignment="center",
        gap="small",
    )
    with brand:
        st.markdown(
            '<div class="app-nav-brand"><span class="app-nav-brand-initials">QP</span>'
            '<span class="app-nav-brand-divider">/</span><span>QuantPricer</span></div>',
            unsafe_allow_html=True,
        )
    destinations = (
        (home, "Home", "Landing Page", "app-nav-home"),
        (lab, "Pricing lab", "Interactive Lab", "app-nav-lab"),
        (engine, "Engine design", "Dev & Architecture", "app-nav-engine"),
        (notes, "Applied theory", "Theoretical Blog", "app-nav-notes"),
    )
    for column, label, route, key in destinations:
        with column:
            st.button(
                label,
                key=key,
                type="primary" if st.session_state.current_page == route else "secondary",
                use_container_width=True,
                on_click=set_page,
                args=(route,),
            )
    with theme:
        st.segmented_control(
            "Theme",
            ["Light", "Dark"],
            key="app-nav-theme",
            default=st.session_state.theme_mode,
            label_visibility="collapsed",
            width="stretch",
            on_change=set_theme_mode,
        )


# Route page content to the extracted renderer or simple placeholder.
if st.session_state.current_page == "Landing Page":
    st.markdown('<div class="landing-page">', unsafe_allow_html=True)

    hero_copy, hero_about = st.columns(
        [1.1, 0.9], gap="large", vertical_alignment="center"
    )
    with hero_copy:
        st.markdown("""
        <div class="landing-hero">
            <div class="landing-eyebrow">Quantitative derivatives · Research & tools</div>
            <h1 class="landing-title">A derivative pricing interactive tool</h1>
            <p class="landing-description">
                Explore how contracts, market assumptions and numerical methods
                come together to shape a derivative’s value.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button(
            "Open the pricing lab",
            key="landing_hero_lab",
            type="primary",
            use_container_width=True,
        ):
            set_page("Interactive Lab")
            st.rerun()
        st.markdown(
            '<div class="landing-meta" style="margin-top:28px;">Monte Carlo · Binomial trees · Finite differences</div>',
            unsafe_allow_html=True,
        )

    with hero_about:
        with st.container(key="landing-suggestions"):
            st.markdown(
                '<div class="landing-learning-title">Learning · Applied theory & implementation notes</div>',
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
                            '<div class="landing-note-label">Inside the pricing engine</div>',
                            unsafe_allow_html=True,
                        )
                    with action_column:
                        st.button(
                            "Go to  →",
                            key="landing-design-link",
                            type="tertiary",
                            use_container_width=True,
                            on_click=open_engine_design,
                        )

            with st.container(key="landing-methods"):
                st.markdown(
                    '<div class="landing-submenu-heading">Applied theory · Work in progress</div>',
                    unsafe_allow_html=True,
                )
                with st.container(key="landing-note-row-intuition"):
                    title_column, action_column = st.columns(
                        [3.4, 1], vertical_alignment="center", gap="small"
                    )
                    with title_column:
                        st.markdown(
                            '<div class="landing-note-label">Explicit Euler, Implicit Euler '
                            'and Crank-Nicolson error comparison</div>',
                            unsafe_allow_html=True,
                        )
                    with action_column:
                        st.button(
                            "Coming soon",
                            key="landing-suggestion-intuition",
                            type="tertiary",
                            use_container_width=True,
                            disabled=True,
                        )

                with st.container(key="landing-note-row-solver"):
                    title_column, action_column = st.columns(
                        [3.4, 1], vertical_alignment="center", gap="small"
                    )
                    with title_column:
                        st.markdown(
                            '<div class="landing-note-label">Rannacher initialization:' \
                            ' fixing oscilations in Crank-Nicolson</div>',
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
                            '<div class="landing-note-label">Hull-White Model: binomial trees for asian payoff</div>',
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
        'Developed by Gabriel Meisler · Pricing engine: Python and NumPy, built around quantitative finance and numerical methods '
        '(Monte Carlo, binomial trees and finite differences) · '
        'Interactive charts: Plotly · Data handling: pandas · '
        'Streamlit frontend built with significant help from AI'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.current_page == "Interactive Lab":
    render_pricing_lab()
elif st.session_state.current_page == "Theoretical Blog":
    is_dark = st.session_state.theme_mode == "Dark"
    theory_ink = "#e6e7e9" if is_dark else "#183247"
    theory_muted = "#9da7b2" if is_dark else "#61727f"
    theory_rule = "#363c46" if is_dark else "#dfe5e9"
    theory_accent = "#91cfc2" if is_dark else "#176b68"
    st.markdown(
        f"""
        <section style="max-width:760px;margin:clamp(72px,14vh,150px) auto;
                        padding:32px 24px;text-align:center;
                        border-top:1px solid {theory_rule};
                        border-bottom:1px solid {theory_rule};">
          <div style="margin-bottom:14px;color:{theory_accent};
                      font:700 0.68rem/1.4 Inter,system-ui,sans-serif;
                      letter-spacing:0.14em;text-transform:uppercase;">
            Applied theory
          </div>
          <h1 style="margin:0;color:{theory_ink};
                     font:400 clamp(2rem,4vw,3rem)/1.12
                     Georgia,'Times New Roman',serif;letter-spacing:-0.025em;">
            This page is being built
          </h1>
          <p style="max-width:460px;margin:14px auto 0;color:{theory_muted};
                    font:400 0.95rem/1.7 Inter,system-ui,sans-serif;">
            New applied theory content is on its way.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.stop()
elif st.session_state.current_page == "Dev & Architecture":
    render_engine_design()
