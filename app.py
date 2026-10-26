"""Streamlit entry point for the QuantPricer application."""
from __future__ import annotations

import streamlit as st

from ui.application import render_application
from ui.navigation import initialize_app_state
from ui.theme import apply_theme


st.set_page_config(
    page_title="QuantPricer | Financial Derivatives Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

initialize_app_state()
apply_theme()
render_application()
