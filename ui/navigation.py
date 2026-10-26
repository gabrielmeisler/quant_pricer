"""Shared page navigation and theme controls."""
from __future__ import annotations

import streamlit as st

PAGE_HOME = "Landing Page"
PAGE_PRICING_LAB = "Interactive Lab"
PAGE_ENGINE_DESIGN = "Dev & Architecture"
PAGE_APPLIED_THEORY = "Theoretical Blog"


def initialize_app_state() -> None:
    """Set defaults for state shared by the app shell and page renderers."""
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "Light"
    if "current_page" not in st.session_state:
        st.session_state.current_page = PAGE_HOME


def _set_page(page_name: str) -> None:
    st.session_state.current_page = page_name


def _set_theme_mode() -> None:
    st.session_state.theme_mode = st.session_state["app-nav-theme"]


def _open_engine_design() -> None:
    _set_page(PAGE_ENGINE_DESIGN)


def render_navigation() -> None:
    """Render the shared brand, navigation links, and theme selector."""
    with st.container(key="app-top-nav"):
        brand, home, lab, engine, notes, theme = st.columns(
            [2.5, 0.72, 1.0, 1.2, 1.1, 1.15],
            vertical_alignment="center",
            gap="small",
        )
        with brand:
            st.markdown(
                '<div class="app-nav-brand">'
                '<span class="app-nav-brand-initials">QP</span>'
                '<span class="app-nav-brand-divider">/</span>'
                '<span>QuantPricer</span></div>',
                unsafe_allow_html=True,
            )
        destinations = (
            (home, "Home", PAGE_HOME, "app-nav-home"),
            (lab, "Pricing lab", PAGE_PRICING_LAB, "app-nav-lab"),
            (engine, "Engine design", PAGE_ENGINE_DESIGN, "app-nav-engine"),
            (notes, "Applied theory", PAGE_APPLIED_THEORY, "app-nav-notes"),
        )
        for column, label, route, key in destinations:
            with column:
                st.button(
                    label,
                    key=key,
                    type=(
                        "primary"
                        if st.session_state.current_page == route
                        else "secondary"
                    ),
                    use_container_width=True,
                    on_click=_set_page,
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
                on_change=_set_theme_mode,
            )
