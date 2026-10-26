"""Route requests to the shared app shell and page renderers."""
from __future__ import annotations

import streamlit as st

from ui.applied_theory import render_applied_theory
from ui.engine_design import render_engine_design
from ui.home import render_home
from ui.navigation import (
    PAGE_APPLIED_THEORY,
    PAGE_ENGINE_DESIGN,
    PAGE_HOME,
    PAGE_PRICING_LAB,
    render_navigation,
)
from ui.pricing_lab import render_pricing_lab


def render_application() -> None:
    """Render shared navigation and the page selected in session state."""
    render_navigation()
    page_renderers = {
        PAGE_HOME: render_home,
        PAGE_PRICING_LAB: render_pricing_lab,
        PAGE_ENGINE_DESIGN: render_engine_design,
        PAGE_APPLIED_THEORY: render_applied_theory,
    }
    page_renderer = page_renderers.get(st.session_state.current_page)
    if page_renderer is None:
        raise ValueError(
            f"Unknown app page: {st.session_state.current_page!r}"
        )
    page_renderer()
