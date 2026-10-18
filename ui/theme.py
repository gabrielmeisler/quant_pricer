"""Apply the shared app stylesheet for the active light or dark theme."""
from __future__ import annotations

from pathlib import Path

import streamlit as st


def apply_theme() -> None:
    """Inject shared layout, navigation, and component styles."""
    # Keep large stylesheets as CSS assets instead of embedding them in Python.
    dark_theme_css = Path(__file__).with_name("dark_theme.css").read_text(
        encoding="utf-8"
    )
    # The small toggle rules depend on the current session theme.
    theme_toggle_css = f"""
       .stApp .st-key-app-nav-theme [data-testid="stButtonGroup"] {{
           background: {"#1b1e25" if st.session_state.theme_mode == "Dark" else "#ffffff"} !important;
           border-color: {"#393d47" if st.session_state.theme_mode == "Dark" else "#dfe5e9"} !important;
       }}
    
       .stApp .st-key-app-nav-theme [data-testid="stButtonGroup"] button {{
           background: transparent !important;
           color: {"#aeb3bb" if st.session_state.theme_mode == "Dark" else "#526574"} !important;
       }}
    
       .stApp .st-key-app-nav-theme [data-testid="stButtonGroup"] button[aria-checked="true"] {{
           background: {"#1d302c" if st.session_state.theme_mode == "Dark" else "#e8f2f1"} !important;
           color: {"#91cfc2" if st.session_state.theme_mode == "Dark" else "#176b68"} !important;
           box-shadow: inset 0 0 0 1px {"#31564e" if st.session_state.theme_mode == "Dark" else "#c5d9d5"} !important;
       }}
    
       .stApp .st-key-app-nav-theme [data-testid="stButtonGroup"] button p {{
           color: inherit !important;
       }}
    """
    
    # Replace placeholders so only the active theme's overrides are injected.
    style_markup = (
        Path(__file__).with_name("theme.css").read_text(encoding="utf-8")
        .replace(
            "__DARK_THEME_CSS__",
            dark_theme_css if st.session_state.theme_mode == "Dark" else "",
        )
        .replace("__THEME_TOGGLE_CSS__", theme_toggle_css)
    )
    st.markdown(style_markup, unsafe_allow_html=True)
