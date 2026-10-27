"""Interactive pricing workflow: contract builder, runs, and comparisons."""
from __future__ import annotations

import colorsys
import html
from typing import Any
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from pricing.state import UnderlyingState, PathDependentState, PricingState
from pricing.markets import MarketData
from pricing.payoff import VanillaPayoff, DigitalPayoff, AsianPayoff, BarrierPayoff
from pricing.exercise_rights import EuropeanExercise, AmericanExercise, BermudanExercise
from pricing.contract import DerivativeContract
from pricing.stochastic_process import GeometricBrownianMotion
from pricing.engines.analytical import AnalyticalEngine
from pricing.engines.base import PricingResult
from pricing.engines.binomial import BinomialEngine
from pricing.engines.finte_differences import FiniteDifferenceEngine
from pricing.engines.monte_carlo import MonteCarloEngine

# Shared helpers used by the interactive builder and pricing workflow.

BERMUDAN_EXERCISE_INTERVAL_MONTHS = {
    "Quarterly": 3,
    "Semi-Annually": 6,
    "Monthly": 1,
}


def bermudan_exercise_times(maturity: float, interval_months: int) -> np.ndarray:
    interval_years = interval_months / 12
    exercise_times = np.arange(interval_years, maturity + 1e-9, interval_years)
    if not len(exercise_times) or not np.isclose(
        exercise_times[-1], maturity, rtol=0.0, atol=1e-9
    ):
        exercise_times = np.append(exercise_times, maturity)
    else:
        exercise_times[-1] = maturity
    return exercise_times


def render_builder_plot(
    full_figure: go.Figure,
    compact_figure: go.Figure,
    tiny_figure: go.Figure,
    key: str,
) -> None:
    config = {"responsive": True, "displaylogo": False}
    for size, figure in (
        ("full", full_figure),
        ("compact", compact_figure),
        ("tiny", tiny_figure),
    ):
        with st.container(key=f"{key}-{size}"):
            st.plotly_chart(
                figure,
                width="stretch",
                config=config,
                key=f"{key}_{size}",
            )



def save_builder_settings() -> None:
    stage_settings = {
        0: (
            "builder_payoff_type",
            "builder_opt_kind",
            "builder_spot",
            "builder_strike",
            "builder_barrier",
            "builder_cash",
        ),
        1: (
            "builder_ex_style",
            "builder_maturity",
            "builder_berm_freq",
        ),
        2: ("builder_vol", "builder_r", "builder_q"),
    }
    stage = st.session_state.get("builder_stage", 0)
    for key in stage_settings.get(stage, ()):
        input_key = f"{key}_input"
        if input_key in st.session_state:
            st.session_state[key] = st.session_state[input_key]


def set_lab_workflow_step(step: int) -> None:
    if step not in (1, 2, 3, 4):
        raise ValueError(f"Unsupported pricing workflow step: {step}")

    save_builder_settings()
    builder_stage = min(step - 1, 2)
    st.session_state.builder_stage = builder_stage
    st.session_state.builder_max_stage = max(
        st.session_state.get("builder_max_stage", 0),
        builder_stage,
    )
    st.session_state.interactive_lab_section = (
        "compare" if step == 4 else "portfolio"
    )


def open_valuation_tab():
    set_lab_workflow_step(4)



def render_pricing_lab() -> None:
    """Render the builder and pricing results for the active contract."""
    if "interactive_lab_section" not in st.session_state:
        st.session_state.interactive_lab_section = "portfolio"
    active_lab_section = st.session_state.interactive_lab_section
    st.markdown(
        """
        <style>
            .st-key-lab-builder-content { display: none !important; }
        </style>
        """
        if active_lab_section == "compare"
        else """
        <style>
            .st-key-lab-engine-content { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    workflow_step = (
        4
        if active_lab_section == "compare"
        else min(st.session_state.get("builder_stage", 0) + 1, 3)
    )
    workflow_labels = (
        "Construct payoff",
        "Select exercise rights",
        "Select market parameters",
        "Price the contract",
    )
    with st.container(key="lab-section-switch"):
        step_columns = st.columns(4, vertical_alignment="center", gap="small")
        for step, (column, label) in enumerate(
            zip(step_columns, workflow_labels),
            start=1,
        ):
            step_status = (
                "current"
                if step == workflow_step
                else "complete"
                if step < workflow_step
                else "upcoming"
            )
            with column, st.container(key=f"lab-step-{step}-{step_status}"):
                st.button(
                    label,
                    key=f"lab-workflow-step-{step}",
                    type="tertiary",
                    use_container_width=True,
                    on_click=set_lab_workflow_step,
                    args=(step,),
                )

    with st.container(key="lab-builder-content"):
        builder_defaults = {
            "builder_payoff_type": "Vanilla Payoff",
            "builder_opt_kind": "Call",
            "builder_spot": 100.0,
            "builder_strike": 100.0,
            "builder_barrier": 120.0,
            "builder_cash": 10.0,
            "builder_ex_style": "European",
            "builder_maturity": 1.0,
            "builder_berm_freq": "Quarterly",
            "builder_vol": 0.02,
            "builder_r": 0.05,
            "builder_q": 0.01,
            "builder_stage": 0,
        }
        for key, value in builder_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = st.session_state.get(
                    f"{key}_saved", value
                )
            input_key = f"{key}_input"
            if key != "builder_stage" and input_key not in st.session_state:
                st.session_state[input_key] = st.session_state[key]

        def save_builder_setting(key: str) -> None:
            st.session_state[key] = st.session_state[f"{key}_input"]

        builder_stage = st.session_state.builder_stage
        if "builder_max_stage" not in st.session_state:
            st.session_state.builder_max_stage = builder_stage
        else:
            st.session_state.builder_max_stage = max(
                st.session_state.builder_max_stage, builder_stage
            )
        builder_max_stage = st.session_state.builder_max_stage
        required_drawers = []
        if builder_max_stage < 1:
            required_drawers.append("Payoff")
        if builder_max_stage < 2:
            required_drawers.extend(["Exercise", "Market"])
        missing_drawers_label = " · ".join(required_drawers)
        pricing_ready = not required_drawers
        is_dark = st.session_state.theme_mode == "Dark"
        chart_text = "#c2c5ca" if is_dark else "#526574"
        chart_grid = "#343943" if is_dark else "#dfe5e9"
        chart_primary = "#82c5b8" if is_dark else "#176b68"
        chart_red = "#e08a83" if is_dark else "#a95350"
        chart_gold = "#d2b574" if is_dark else "#a8874a"
        chart_purple = "#b59bc0" if is_dark else "#765a7d"
        payoff_drawer_type = {
            "Vanilla Payoff": "Vanilla",
            "Digital Cash-or-Nothing": "Digital",
            "Asian (Running Average)": "Asian average",
            "Barrier Knock-Out": "Barrier KO",
        }[st.session_state.builder_payoff_type]
        payoff_drawer_terms = [
            f"**Payoff type**  \n{payoff_drawer_type}",
            f"**Option right**  \n{st.session_state.builder_opt_kind}",
            f"**Spot**  \n{st.session_state.builder_spot:g}",
            f"**Strike**  \n{st.session_state.builder_strike:g}",
        ]
        if st.session_state.builder_payoff_type == "Barrier Knock-Out":
            payoff_drawer_terms.append(
                f"**Barrier**  \n{st.session_state.builder_barrier:g}"
            )
        elif st.session_state.builder_payoff_type == "Digital Cash-or-Nothing":
            payoff_drawer_terms.append(
                f"**Cash payout**  \n{st.session_state.builder_cash:g}"
            )
        payoff_drawer_summary = "\n\n".join(payoff_drawer_terms)
        exercise_drawer_terms = [
            f"**Exercise style**  \n{st.session_state.builder_ex_style}",
            f"**Maturity**  \n{st.session_state.builder_maturity:.2f}y",
        ]
        if st.session_state.builder_ex_style == "Bermudan":
            exercise_drawer_terms.append(
                f"**Exercise frequency**  \n{st.session_state.builder_berm_freq}"
            )
        exercise_drawer_summary = "\n\n".join(exercise_drawer_terms)
        market_drawer_terms = [
            f"**Volatility**  \n{st.session_state.builder_vol:.0%}",
            f"**Risk-free rate**  \n{st.session_state.builder_r:.1%}",
            f"**Dividend yield**  \n{st.session_state.builder_q:.1%}",
        ]
        market_drawer_summary = "\n\n".join(market_drawer_terms)

        payoff_type = st.session_state.builder_payoff_type
        option_kind = st.session_state.builder_opt_kind
        is_call = option_kind == "Call"
        spot_price = st.session_state.builder_spot
        strike_price = st.session_state.builder_strike
        cash_payout = st.session_state.builder_cash
        barrier_level = st.session_state.builder_barrier
        barrier_dir = "up"
        exercise_type = st.session_state.builder_ex_style
        maturity = st.session_state.builder_maturity
        bermudan_freq = (
            st.session_state.builder_berm_freq
            if exercise_type == "Bermudan"
            else ("Continuous" if exercise_type == "American" else "At Expiry")
        )
        months_per_exercise = BERMUDAN_EXERCISE_INTERVAL_MONTHS.get(
            bermudan_freq, 3
        )
        volatility = st.session_state.builder_vol
        risk_free = st.session_state.builder_r
        dividend_yield = st.session_state.builder_q

        def make_payoff_profile():
            if payoff_type == "Vanilla Payoff":
                payoff = VanillaPayoff(strike=strike_price, is_call=is_call)
            elif payoff_type == "Digital Cash-or-Nothing":
                payoff = DigitalPayoff(
                    strike=strike_price, is_call=is_call, cash_payout=cash_payout
                )
            elif payoff_type == "Asian (Running Average)":
                payoff = AsianPayoff(strike=strike_price, is_call=is_call)
            else:
                payoff = BarrierPayoff(
                    strike=strike_price,
                    is_call=is_call,
                    barrier_level=barrier_level,
                    barrier_direction=barrier_dir,
                )

            upper = max(
                spot_price,
                strike_price,
                barrier_level if payoff_type == "Barrier Knock-Out" else 0.0,
            ) * 2.0
            spots = np.linspace(0.0, upper, 260)
            states = [
                PricingState(
                    UnderlyingState(time=1.0, spot=float(spot)),
                    PathDependentState(
                        {"running_average": float(spot), "knock_out": False}
                    ),
                )
                for spot in spots
            ]
            values = [float(payoff(state)) for state in states]
            return payoff, spots, values, upper

        if builder_stage != 0:
            dummy_payoff, s_grid, payoff_vals, chart_upper = make_payoff_profile()

        column_weights = (
            [8, 1.8, 1.8] if builder_stage == 0
            else [1.8, 8, 1.8] if builder_stage == 1
            else [1.8, 1.8, 8]
        )
        payoff_column, exercise_column, market_column = st.columns(
            column_weights, gap="small"
        )

        if builder_stage == 0:
            for key in (
                "builder_payoff_type",
                "builder_opt_kind",
                "builder_spot",
                "builder_strike",
                "builder_barrier",
                "builder_cash",
            ):
                st.session_state[f"{key}_input"] = st.session_state[key]
            with payoff_column, st.container(key="builder-active-payoff"):
                st.markdown("### 01  Payoff structure")
                payoff_type = st.selectbox(
                    "Payoff structure",
                    [
                        "Vanilla Payoff",
                        "Digital Cash-or-Nothing",
                        "Asian (Running Average)",
                        "Barrier Knock-Out",
                    ],
                    key="builder_payoff_type_input",
                    on_change=save_builder_setting,
                    args=("builder_payoff_type",),
                )
                option_kind = st.segmented_control(
                    "Option right",
                    ["Call", "Put"],
                    key="builder_opt_kind_input",
                    width="stretch",
                    on_change=save_builder_setting,
                    args=("builder_opt_kind",),
                )
                has_extra_payoff_input = payoff_type in {
                    "Barrier Knock-Out",
                    "Digital Cash-or-Nothing",
                }
                is_call = option_kind == "Call"

                payoff_reference_controls = st.columns(
                    [1.0, 1.0] + ([1.0] if has_extra_payoff_input else []),
                    gap="small",
                )
                with payoff_reference_controls[0]:
                    spot_price = st.number_input(
                    "Spot ($S_0$)", min_value=1.0, step=1.0,
                        key="builder_spot_input",
                        on_change=save_builder_setting,
                        args=("builder_spot",),
                    )
                with payoff_reference_controls[1]:
                    strike_price = st.number_input(
                    "Strike ($K$)", min_value=1.0, step=1.0,
                        key="builder_strike_input",
                        on_change=save_builder_setting,
                        args=("builder_strike",),
                    )
                if payoff_type == "Barrier Knock-Out":
                    with payoff_reference_controls[2]:
                        barrier_level = st.number_input(
                        "Barrier ($B$)", min_value=1.0, step=1.0,
                            key="builder_barrier_input",
                            on_change=save_builder_setting,
                            args=("builder_barrier",),
                        )
                elif payoff_type == "Digital Cash-or-Nothing":
                    with payoff_reference_controls[2]:
                        cash_payout = st.number_input(
                        "Payout ($)", min_value=0.1, step=1.0,
                            key="builder_cash_input",
                            on_change=save_builder_setting,
                            args=("builder_cash",),
                        )

                dummy_payoff, s_grid, payoff_vals, chart_upper = make_payoff_profile()
                payoff_y_top = float(np.max(payoff_vals)) * 1.08
                if payoff_y_top <= 0:
                    payoff_y_top = 1.0
                fig_payoff = go.Figure()
                fig_payoff.add_trace(
                    go.Scatter(
                        x=s_grid,
                        y=payoff_vals,
                        mode="lines",
                        name="Expiry payoff",
                        line={"color": chart_primary, "width": 2.5},
                        fill="tozeroy",
                        fillcolor=(
                            "rgba(23,107,104,0.075)" if not is_dark
                            else "rgba(130,197,184,0.085)"
                        ),
                        hovertemplate="Sₜ = %{x:.2f}<br>Payoff = %{y:.2f}<extra></extra>",
                        showlegend=False,
                    )
                )
                for value, label, color, dash in (
                    (strike_price, f"Strike K = {strike_price:g}", chart_red, "dash"),
                    (spot_price, f"Spot S₀ = {spot_price:g}", chart_gold, "dot"),
                ):
                    fig_payoff.add_trace(
                        go.Scatter(
                            x=[value, value],
                            y=[0, payoff_y_top],
                            mode="lines",
                            name=label,
                            line={"color": color, "width": 1.2, "dash": dash},
                            hoverinfo="skip",
                        )
                    )
                if payoff_type == "Barrier Knock-Out":
                    fig_payoff.add_trace(
                        go.Scatter(
                            x=[barrier_level, barrier_level],
                            y=[0, payoff_y_top],
                            mode="lines",
                            name=f"Barrier B = {barrier_level:g}",
                            line={
                                "color": chart_purple,
                                "width": 1.2,
                                "dash": "dashdot",
                            },
                            hoverinfo="skip",
                        )
                    )
                fig_payoff.update_layout(
                    height=250,
                    autosize=True,
                    margin={"l": 55, "r": 20, "t": 35, "b": 55},
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={
                        "color": chart_text,
                        "size": 11,
                        "family": "Inter, system-ui, sans-serif",
                    },
                    hovermode="x",
                    showlegend=True,
                    legend={
                        "orientation": "h",
                        "x": 0,
                        "y": 1.08,
                        "xanchor": "left",
                        "yanchor": "bottom",
                        "font": {"color": chart_text, "size": 9},
                        "bgcolor": "rgba(0,0,0,0)",
                        "itemsizing": "constant",
                    },
                    hoverlabel={
                        "bgcolor": "#20252a" if is_dark else "#ffffff",
                        "bordercolor": chart_primary,
                        "font": {"color": chart_text, "size": 11},
                    },
                    modebar={
                        "bgcolor": "rgba(0,0,0,0)",
                        "color": chart_text,
                        "activecolor": chart_primary,
                    },
                    xaxis={
                        "title": {
                            "text": "Underlying at expiry, Sₜ",
                            "font": {"size": 11, "color": chart_text},
                            "standoff": 8,
                        },
                        "range": [0, chart_upper],
                        "gridcolor": chart_grid,
                        "gridwidth": 0.5,
                        "griddash": "dot",
                        "zeroline": False,
                        "showline": False,
                        "ticks": "",
                        "tickfont": {"size": 10, "color": chart_text},
                        "automargin": True,
                    },
                    yaxis={
                        "title": {
                            "text": "Contract payoff",
                            "font": {"size": 11, "color": chart_text},
                            "standoff": 8,
                        },
                        "range": [0, payoff_y_top],
                        "gridcolor": chart_grid,
                        "gridwidth": 0.5,
                        "griddash": "dot",
                        "zeroline": False,
                        "showline": False,
                        "ticks": "",
                        "tickfont": {"size": 10, "color": chart_text},
                        "automargin": True,
                    },
                )
                compact_payoff = go.Figure(fig_payoff)
                compact_payoff.update_layout(
                    height=165,
                    margin={"l": 38, "r": 8, "t": 30, "b": 42},
                    font={"color": chart_text, "size": 10},
                    hoverlabel={
                        "bgcolor": "#20252a" if is_dark else "#ffffff",
                        "bordercolor": chart_primary,
                        "font": {"color": chart_text, "size": 10},
                    },
                )
                tiny_payoff = go.Figure(compact_payoff)
                tiny_payoff.update_layout(
                    height=76,
                    margin={"l": 30, "r": 5, "t": 8, "b": 26},
                    font={"color": chart_text, "size": 9},
                    showlegend=False,
                    xaxis_title=None,
                    yaxis_title=None,
                )
                render_builder_plot(
                    fig_payoff,
                    compact_payoff,
                    tiny_payoff,
                    "builder-payoff-plot",
                )

                payoff_name = {
                    "Vanilla Payoff": "Vanilla",
                    "Digital Cash-or-Nothing": "Digital",
                    "Asian (Running Average)": "Asian average",
                    "Barrier Knock-Out": "Barrier knock-out",
                }[payoff_type]
                payoff_action_summary = (
                    f"{payoff_name} {option_kind} · "
                    f"S₀ {spot_price:g} · K {strike_price:g}"
                )
                if payoff_type == "Barrier Knock-Out":
                    payoff_action_summary += f" · B {barrier_level:g}"
                elif payoff_type == "Digital Cash-or-Nothing":
                    payoff_action_summary += f" · payout {cash_payout:g}"

                with st.container(key="builder-next-payoff"):
                    if st.button(
                        f"Select {payoff_action_summary}  →",
                        key="builder_continue_payoff",
                        type="primary",
                        use_container_width=True,
                    ):
                        save_builder_settings()
                        st.session_state.builder_stage = 1
                        st.rerun()
        else:
            drawer_label = (
                f"**01**\n\n**PAYOFF**\n\n{payoff_drawer_summary}"
            )
            with payoff_column, st.container(key="builder-open-payoff"):
                if st.button(
                    drawer_label,
                    key="builder_open_payoff",
                    use_container_width=True,
                ):
                    save_builder_settings()
                    st.session_state.builder_stage = 0
                    st.rerun()

        if builder_stage == 1:
            for key in (
                "builder_ex_style",
                "builder_maturity",
                "builder_berm_freq",
            ):
                st.session_state[f"{key}_input"] = st.session_state[key]
            with exercise_column, st.container(key="builder-active-exercise"):
                st.markdown("### 02  Exercise rights")
                exercise_type = st.selectbox(
                    "Style",
                    ["European", "American", "Bermudan"],
                    key="builder_ex_style_input",
                    on_change=save_builder_setting,
                    args=("builder_ex_style",),
                )
                exercise_controls = st.columns(2, gap="small")
                with exercise_controls[0]:
                    maturity = st.number_input(
                        "Maturity (years)", min_value=0.1, max_value=10.0,
                        step=0.25, key="builder_maturity_input",
                        on_change=save_builder_setting,
                        args=("builder_maturity",),
                    )
                if exercise_type == "European":
                    bermudan_freq = "At Expiry"
                    with exercise_controls[1]:
                        st.selectbox(
                            "Frequency",
                            ["At Expiry Only (t = T)"],
                            disabled=True,
                            key="builder_ex_freq_eu",
                        )
                elif exercise_type == "American":
                    bermudan_freq = "Continuous"
                    with exercise_controls[1]:
                        st.selectbox(
                            "Frequency",
                            ["Continuous Window [0, T]"],
                            disabled=True,
                            key="builder_ex_freq_am",
                        )
                else:
                    with exercise_controls[1]:
                        bermudan_freq = st.selectbox(
                            "Frequency",
                            ["Quarterly", "Semi-Annually", "Monthly"],
                            key="builder_berm_freq_input",
                            on_change=save_builder_setting,
                            args=("builder_berm_freq",),
                        )
                    months_per_exercise = BERMUDAN_EXERCISE_INTERVAL_MONTHS[
                        bermudan_freq
                    ]

                if exercise_type == "Bermudan":
                    exercise_dates = bermudan_exercise_times(
                        maturity, months_per_exercise
                    )
                else:
                    exercise_dates = np.array([maturity]) if exercise_type == "European" else np.array([])

                exercise_axis_padding = maturity * 0.06
                fig_exercise = go.Figure()
                fig_exercise.add_trace(
                    go.Scatter(
                        x=[0, maturity], y=[0, 0],
                        mode="lines",
                        line={"color": chart_grid, "width": 1},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                if exercise_type == "American":
                    fig_exercise.add_trace(
                        go.Scatter(
                            x=[0, maturity], y=[0, 0],
                            mode="lines",
                            line={"color": chart_primary, "width": 4},
                            name="Exercise any time",
                            hovertemplate="Exercise permitted throughout [0, T]<extra></extra>",
                        )
                    )
                fig_exercise.add_trace(
                    go.Scatter(
                        x=[0], y=[0],
                        mode="markers",
                        marker={"color": chart_text, "size": 8},
                        name="Inception",
                        hovertemplate="t = %{x:.2f} years<extra></extra>",
                    )
                )
                fig_exercise.add_trace(
                    go.Scatter(
                        x=[maturity], y=[0],
                        mode="markers",
                        marker={
                            "color": chart_primary,
                            "size": 8,
                            "symbol": "circle",
                        },
                        name="Maturity",
                        hovertemplate="t = %{x:.2f} years<extra></extra>",
                    )
                )
                fig_exercise.add_annotation(
                    x=0,
                    y=0,
                    text="Inception",
                    showarrow=False,
                    xanchor="left",
                    xshift=8,
                    yshift=14,
                    font={"color": chart_text, "size": 10},
                )
                fig_exercise.add_annotation(
                    x=maturity,
                    y=0,
                    text="Maturity",
                    showarrow=False,
                    xanchor="right",
                    xshift=-8,
                    yshift=14,
                    font={"color": chart_text, "size": 10},
                )
                if exercise_type == "Bermudan" and len(exercise_dates):
                    displayed_exercise_dates = (
                        exercise_dates[
                            ~np.isclose(
                                exercise_dates,
                                maturity,
                                rtol=0.0,
                                atol=1e-9,
                            )
                        ]
                        if exercise_type == "Bermudan"
                        else np.array([])
                    )
                    if len(displayed_exercise_dates):
                        fig_exercise.add_trace(
                            go.Scatter(
                                x=displayed_exercise_dates,
                                y=np.zeros(len(displayed_exercise_dates)),
                                mode="markers",
                                marker={
                                    "color": chart_gold,
                                    "size": 10,
                                    "symbol": "star",
                                },
                                name="Permitted exercise dates",
                                hovertemplate="Exercise at t = %{x:.2f} years<extra></extra>",
                            )
                        )
                fig_exercise.update_layout(
                    height=190,
                    autosize=True,
                    margin={"l": 16, "r": 16, "t": 40, "b": 52},
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={
                        "color": chart_text,
                        "size": 11,
                        "family": "Inter, system-ui, sans-serif",
                    },
                    showlegend=True,
                    legend={
                        "orientation": "h",
                        "yanchor": "bottom",
                        "y": 1.08,
                        "xanchor": "left",
                        "x": 0,
                        "font": {"color": chart_text, "size": 9},
                        "bgcolor": "rgba(0,0,0,0)",
                    },
                    hoverlabel={
                        "bgcolor": "#20252a" if is_dark else "#ffffff",
                        "bordercolor": chart_primary,
                        "font": {"color": chart_text, "size": 10},
                    },
                    modebar={
                        "bgcolor": "rgba(0,0,0,0)",
                        "color": chart_text,
                        "activecolor": chart_primary,
                    },
                    xaxis={
                        "title": {
                            "text": "Time to maturity (years)",
                            "font": {"size": 11, "color": chart_text},
                            "standoff": 7,
                        },
                        "range": [
                            -exercise_axis_padding,
                            maturity + exercise_axis_padding,
                        ],
                        "tickvals": np.linspace(0, maturity, 5),
                        "tickformat": ".2g",
                        "showgrid": True,
                        "gridcolor": chart_grid,
                        "gridwidth": 0.5,
                        "griddash": "dot",
                        "showline": False,
                        "ticks": "",
                        "tickfont": {"size": 10, "color": chart_text},
                        "zeroline": False,
                        "automargin": True,
                    },
                    yaxis={
                        "range": [-0.18, 0.32],
                        "showgrid": False,
                        "showticklabels": False,
                        "showline": False,
                        "ticks": "",
                        "zeroline": False,
                    },
                )
                compact_exercise = go.Figure(fig_exercise)
                compact_exercise.update_layout(
                    height=198,
                    margin={"l": 10, "r": 10, "t": 34, "b": 38},
                    font={"color": chart_text, "size": 10},
                    hoverlabel={
                        "bgcolor": "#20252a" if is_dark else "#ffffff",
                        "bordercolor": chart_primary,
                        "font": {"color": chart_text, "size": 10},
                    },
                )
                tiny_exercise = go.Figure(compact_exercise)
                tiny_exercise.update_layout(
                    height=65,
                    margin={"l": 5, "r": 5, "t": 5, "b": 20},
                    font={"color": chart_text, "size": 9},
                    showlegend=False,
                )
                tiny_exercise.update_layout(annotations=[])
                tiny_exercise.update_xaxes(title=None)
                tiny_exercise.update_xaxes(
                    range=[
                        -exercise_axis_padding,
                        maturity + exercise_axis_padding,
                    ],
                    tickvals=np.linspace(0, maturity, 5),
                )
                render_builder_plot(
                    fig_exercise,
                    compact_exercise,
                    tiny_exercise,
                    "builder-exercise-plot",
                )
                exercise_action_summary = (
                    f"{exercise_type} · T {maturity:.2f}y"
                )
                if exercise_type == "Bermudan":
                    exercise_action_summary += f" · {bermudan_freq}"

                with st.container(key="builder-next-exercise"):
                    if st.button(
                        f"Select {exercise_action_summary}  →",
                        key="builder_continue_exercise",
                        type="primary",
                        use_container_width=True,
                    ):
                        save_builder_settings()
                        st.session_state.builder_stage = 2
                        st.rerun()
        else:
            if builder_max_stage >= 1:
                drawer_label = (
                    f"**02**\n\n**EXERCISE**\n\n{exercise_drawer_summary}"
                )
                with exercise_column, st.container(key="builder-open-exercise"):
                    if st.button(
                        drawer_label,
                        key="builder_open_exercise",
                        use_container_width=True,
                    ):
                        save_builder_settings()
                        st.session_state.builder_stage = 1
                        st.rerun()
            else:
                drawer_label = "**02**\n\n**EXERCISE**"
                with exercise_column, st.container(key="builder-open-exercise"):
                    st.button(
                        drawer_label,
                        key="builder_locked_exercise",
                        use_container_width=True,
                        disabled=True,
                    )

        if builder_stage == 2:
            with market_column, st.container(key="builder-active-market"):
                st.markdown("### 03  Market assumptions")
                st.session_state.builder_vol_input = st.session_state.builder_vol
                st.session_state.builder_r_input = st.session_state.builder_r
                st.session_state.builder_q_input = st.session_state.builder_q
                market_inputs = st.columns(3)
                with market_inputs[0]:
                    volatility = st.number_input(
                        "Volatility (σ)", min_value=0.01, max_value=2.0,
                        step=0.01, key="builder_vol_input",
                        on_change=save_builder_setting,
                        args=("builder_vol",),
                    )
                with market_inputs[1]:
                    risk_free = st.number_input(
                        "Risk-free rate (r)", min_value=-0.10, max_value=0.50,
                        step=0.01, key="builder_r_input",
                        on_change=save_builder_setting,
                        args=("builder_r",),
                    )
                with market_inputs[2]:
                    dividend_yield = st.number_input(
                        "Dividend yield (q)", min_value=0.0, max_value=0.50,
                        step=0.01, key="builder_q_input",
                        on_change=save_builder_setting,
                        args=("builder_q",),
                    )
                st.selectbox(
                    "Stochastic process",
                    ["GBM Diffusion"],
                    disabled=True,
                    key="builder_process_model",
                )
                t_grid = np.linspace(0, maturity, 100)
                log_drift = (risk_free - dividend_yield - 0.5 * volatility**2) * t_grid
                expected_path = spot_price * np.exp(
                    (risk_free - dividend_yield) * t_grid
                )
                lower_1s = spot_price * np.exp(
                    log_drift - volatility * np.sqrt(t_grid)
                )
                upper_1s = spot_price * np.exp(
                    log_drift + volatility * np.sqrt(t_grid)
                )
                lower_2s = spot_price * np.exp(
                    log_drift - 2 * volatility * np.sqrt(t_grid)
                )
                upper_2s = spot_price * np.exp(
                    log_drift + 2 * volatility * np.sqrt(t_grid)
                )
                envelope_min = float(np.min(lower_2s))
                envelope_max = float(np.max(upper_2s))
                envelope_margin = (envelope_max - envelope_min) * 0.04
                market_y_range = [
                    max(0.0, envelope_min - envelope_margin),
                    envelope_max + envelope_margin,
                ]

                fig_market = go.Figure()
                for lower, upper, label, fill_color in (
                    (
                        lower_2s, upper_2s, "±2σ quantiles",
                        "rgba(23,107,104,0.045)" if not is_dark
                        else "rgba(130,197,184,0.055)",
                    ),
                    (
                        lower_1s, upper_1s, "±1σ quantiles",
                        "rgba(23,107,104,0.12)" if not is_dark
                        else "rgba(130,197,184,0.13)",
                    ),
                ):
                    fig_market.add_trace(
                        go.Scatter(
                            x=t_grid, y=upper, mode="lines",
                            line={"width": 0}, hoverinfo="skip",
                            showlegend=False,
                        )
                    )
                    fig_market.add_trace(
                        go.Scatter(
                            x=t_grid, y=lower, mode="lines",
                            line={"width": 0}, fill="tonexty",
                            fillcolor=fill_color, name=label,
                            hoverinfo="skip",
                        )
                    )
                fig_market.add_trace(
                    go.Scatter(
                        x=t_grid, y=expected_path, mode="lines",
                        line={"color": chart_primary, "width": 2.5},
                        name="Risk-neutral expected price",
                        hovertemplate="t = %{x:.2f} years<br>E[Sₜ] = %{y:.2f}<extra></extra>",
                    )
                )
                if payoff_type == "Barrier Knock-Out":
                    fig_market.add_hline(
                        y=barrier_level,
                        line_dash="dash",
                        line_color=chart_purple,
                        annotation_text=f"Barrier B = {barrier_level:g}",
                        annotation_position="top left",
                    )
                fig_market.update_layout(
                    height=310,
                    autosize=True,
                    margin={"l": 60, "r": 25, "t": 25, "b": 55},
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={
                        "color": chart_text,
                        "size": 11,
                        "family": "Inter, system-ui, sans-serif",
                    },
                    hovermode="x",
                    hoverlabel={
                        "bgcolor": "#20252a" if is_dark else "#ffffff",
                        "bordercolor": chart_primary,
                        "font": {"color": chart_text, "size": 11},
                    },
                    modebar={
                        "bgcolor": "rgba(0,0,0,0)",
                        "color": chart_text,
                        "activecolor": chart_primary,
                    },
                    legend={
                        "orientation": "h",
                        "yanchor": "bottom",
                        "y": 1.02,
                        "xanchor": "left",
                        "x": 0,
                        "font": {"color": chart_text, "size": 10},
                        "bgcolor": "rgba(0,0,0,0)",
                    },
                    xaxis={
                        "title": {
                            "text": "Time (years)",
                            "font": {"size": 11, "color": chart_text},
                            "standoff": 8,
                        },
                        "range": [0, maturity],
                        "showgrid": True,
                        "gridcolor": chart_grid,
                        "gridwidth": 0.5,
                        "griddash": "dot",
                        "showline": False,
                        "ticks": "",
                        "tickfont": {"size": 10, "color": chart_text},
                        "zeroline": False,
                        "automargin": True,
                    },
                    yaxis={
                        "title": {
                            "text": "Underlying price",
                            "font": {"size": 11, "color": chart_text},
                            "standoff": 8,
                        },
                        "range": market_y_range,
                        "showgrid": True,
                        "gridcolor": chart_grid,
                        "gridwidth": 0.5,
                        "griddash": "dot",
                        "showline": False,
                        "ticks": "",
                        "tickfont": {"size": 10, "color": chart_text},
                        "zeroline": False,
                        "automargin": True,
                    },
                )
                compact_market = go.Figure(fig_market)
                compact_market.update_layout(
                    height=210,
                    margin={"l": 42, "r": 10, "t": 18, "b": 40},
                    font={"color": chart_text, "size": 10},
                    legend={
                        "orientation": "h",
                        "yanchor": "bottom",
                        "y": 1.0,
                        "xanchor": "left",
                        "x": 0,
                        "font": {"color": chart_text, "size": 9},
                        "bgcolor": "rgba(0,0,0,0)",
                    },
                    hoverlabel={
                        "bgcolor": "#20252a" if is_dark else "#ffffff",
                        "bordercolor": chart_primary,
                        "font": {"color": chart_text, "size": 10},
                    },
                )
                tiny_market = go.Figure(compact_market)
                tiny_market.update_layout(
                    height=90,
                    margin={"l": 30, "r": 5, "t": 8, "b": 25},
                    font={"color": chart_text, "size": 9},
                    showlegend=False,
                )
                tiny_market.update_xaxes(title=None)
                tiny_market.update_yaxes(title=None)
                render_builder_plot(
                    fig_market,
                    compact_market,
                    tiny_market,
                    "builder-market-plot",
                )
                price_button_label = "Price contract"
                st.button(
                    price_button_label,
                    key="builder_price_button",
                    type="primary",
                    use_container_width=True,
                    on_click=open_valuation_tab,
                )
        elif builder_max_stage >= 2:
            drawer_label = (
                f"**03**\n\n**MARKET**\n\n{market_drawer_summary}"
            )
            with market_column, st.container(key="builder-open-market"):
                if st.button(
                    drawer_label,
                    key="builder_open_market",
                    use_container_width=True,
                ):
                    save_builder_settings()
                    st.session_state.builder_stage = 2
                    st.rerun()
        else:
            drawer_label = "**03**\n\n**MARKET**"
            with market_column, st.container(key="builder-open-market"):
                st.button(
                    drawer_label,
                    key="builder_locked_market",
                    use_container_width=True,
                    disabled=True,
                )

    with st.container(key="lab-engine-content"):
        is_path_dep = payoff_type in ["Asian (Running Average)", "Barrier Knock-Out"]
        engine_options = ["Monte Carlo (LSM)"]
        if not is_path_dep:
            engine_options.append("Binomial Tree (CRR)")
        if payoff_type == "Vanilla Payoff":
            engine_options.append("Finite Difference (θ-scheme)")

        slider_track_styles: list[str] = []

        def add_slider_track_style(
            widget_key: str, value: float, minimum: float, maximum: float
        ) -> None:
            fill_percentage = (value - minimum) / (maximum - minimum) * 100
            slider_track_styles.append(
                f"""
                [role="dialog"] .st-key-{widget_key}
                [data-rac][data-orientation="horizontal"]
                > div:not([data-rac]):first-child {{
                    background: linear-gradient(
                        to right,
                        var(--pricing-engine-accent) 0%,
                        var(--pricing-engine-accent) {fill_percentage:.4f}%,
                        rgba(153, 173, 194, 0.25) {fill_percentage:.4f}%,
                        rgba(153, 173, 194, 0.25) 100%
                    ) !important;
                }}
                """
            )

        def render_engine_settings(
            engine_name: str,
            key_prefix: str,
            initial_settings: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            initial_settings = initial_settings or {}
            settings: dict[str, Any] = {}
            if engine_name == "Monte Carlo (LSM)":
                left, right = st.columns(2, gap="small")
                with left:
                    settings["n_paths"] = st.slider(
                        "Simulation paths", 1000, 50000,
                        int(initial_settings.get("n_paths", 10000)), 1000,
                        key=f"{key_prefix}_n_paths",
                        help=(
                            "Number of simulated price paths. More paths can "
                            "improve the estimate but increase computation time."
                        ),
                    )
                    add_slider_track_style(
                        f"{key_prefix}_n_paths", settings["n_paths"], 1000, 50000
                    )
                with right:
                    settings["n_steps"] = st.slider(
                        "Time steps", 10, 252,
                        int(initial_settings.get("n_steps", 50)), 5,
                        key=f"{key_prefix}_n_steps",
                        help=(
                            "Number of time intervals used to simulate each "
                            "path through the contract's life."
                        ),
                    )
                    add_slider_track_style(
                        f"{key_prefix}_n_steps", settings["n_steps"], 10, 252
                    )
                left, right = st.columns(2, gap="small")
                with left:
                    settings["poly_degree"] = st.number_input(
                        "LSM polynomial degree", 1, 5,
                        int(initial_settings.get("poly_degree", 2)),
                        key=f"{key_prefix}_poly_degree",
                        help=(
                            "Polynomial degree used to estimate continuation "
                            "values during the least-squares regression."
                        ),
                    )
                with right:
                    settings["seed"] = st.number_input(
                        "Random seed",
                        min_value=0,
                        value=int(initial_settings.get("seed", 42)),
                        step=1,
                        key=f"{key_prefix}_seed",
                        help=(
                            "Seed for the random number generator. Keep it "
                            "fixed to reproduce the same simulation."
                        ),
                    )
            elif engine_name == "Binomial Tree (CRR)":
                settings["tree_depth"] = st.slider(
                    "Tree depth (steps)", 10, 1000,
                    int(initial_settings.get("tree_depth", 100)), 10,
                    key=f"{key_prefix}_tree_depth",
                    help=(
                        "Number of time steps in the binomial tree. "
                        "A deeper tree can improve accuracy and takes longer."
                    ),
                )
                add_slider_track_style(
                    f"{key_prefix}_tree_depth",
                    settings["tree_depth"],
                    10,
                    1000,
                )
            else:
                left, right = st.columns(2, gap="small")
                with left:
                    settings["n_space"] = st.slider(
                        "Spatial intervals", 40, 800,
                        int(initial_settings.get("n_space", 200)), 20,
                        key=f"{key_prefix}_n_space",
                        help=(
                            "Number of intervals across the underlying-price "
                            "grid used to solve the pricing equation."
                        ),
                    )
                    add_slider_track_style(
                        f"{key_prefix}_n_space", settings["n_space"], 40, 800
                    )
                with right:
                    settings["n_time"] = st.slider(
                        "Time steps", 40, 1200,
                        int(initial_settings.get("n_time", 400)), 40,
                        key=f"{key_prefix}_n_time",
                        help=(
                            "Number of time intervals used to evolve the "
                            "pricing equation from valuation to expiry."
                        ),
                    )
                    add_slider_track_style(
                        f"{key_prefix}_n_time", settings["n_time"], 40, 1200
                    )
                left, right = st.columns(2, gap="small")
                with left:
                    scheme_values = {
                        "Crank–Nicolson": 0.5,
                        "Implicit": 1.0,
                        "Explicit": 0.0,
                    }
                    selected_scheme = next(
                        (
                            label
                            for label, value in scheme_values.items()
                            if value == initial_settings.get("theta", 0.5)
                        ),
                        "Crank–Nicolson",
                    )
                    scheme = st.segmented_control(
                        "Time-stepping scheme",
                        list(scheme_values),
                        default=selected_scheme,
                        key=f"{key_prefix}_theta",
                        width="stretch",
                        help=(
                            "Select the time-stepping method. Crank–Nicolson "
                            "balances the implicit and explicit schemes."
                        ),
                    )
                    settings["theta"] = scheme_values[scheme]
                with right:
                    grid_label = st.segmented_control(
                        "Spatial grid",
                        ["Log-price", "Spot"],
                        default=(
                            "Log-price"
                            if initial_settings.get("grid_type", "log") == "log"
                            else "Spot"
                        ),
                        key=f"{key_prefix}_grid_type",
                        width="stretch",
                        help=(
                            "Choose whether the spatial grid is uniform in "
                            "log-price or in the underlying spot price."
                        ),
                    )
                    settings["grid_type"] = (
                        "log" if grid_label == "Log-price" else "spot"
                    )
                    if settings["grid_type"] == "log":
                        dynamic_setting = "log"
                    else:
                        dynamic_setting = "spot"
                if dynamic_setting == "log":
                    settings["log_width_sigma"] = st.slider(
                        "Log-grid width (σ units)", 3.0, 8.0,
                        float(initial_settings.get("log_width_sigma", 5.0)), 0.5,
                        key=f"{key_prefix}_log_width",
                        help=(
                            "Half-width of the log-price grid around the "
                            "current spot, measured in volatility units."
                        ),
                    )
                    add_slider_track_style(
                        f"{key_prefix}_log_width",
                        settings["log_width_sigma"],
                        3.0,
                        8.0,
                    )
                else:
                    settings["s_max"] = st.number_input(
                        "Spot-grid upper bound", min_value=100.0,
                        value=float(initial_settings.get(
                            "s_max", max(300.0, 3.0 * max(spot_price, strike_price))
                        )),
                        step=25.0, key=f"{key_prefix}_s_max",
                        help=(
                            "Upper spot-price boundary for the numerical "
                            "grid; values above it are outside the domain."
                        ),
                    )
            return settings

        def make_pricing_contract() -> tuple[DerivativeContract, Any, Any]:
            if payoff_type == "Vanilla Payoff":
                payoff_obj = VanillaPayoff(strike=strike_price, is_call=is_call)
            elif payoff_type == "Digital Cash-or-Nothing":
                payoff_obj = DigitalPayoff(
                    strike=strike_price, is_call=is_call, cash_payout=cash_payout
                )
            elif payoff_type == "Asian (Running Average)":
                payoff_obj = AsianPayoff(strike=strike_price, is_call=is_call)
            else:
                payoff_obj = BarrierPayoff(
                    strike=strike_price,
                    is_call=is_call,
                    barrier_level=barrier_level,
                    barrier_direction=barrier_dir,
                )

            if exercise_type == "European":
                rights_obj = EuropeanExercise(maturity=maturity)
            elif exercise_type == "American":
                rights_obj = AmericanExercise(maturity=maturity)
            else:
                exercise_dates = bermudan_exercise_times(
                    maturity, months_per_exercise
                ).tolist()
                rights_obj = BermudanExercise(
                    maturity=maturity, exercise_times=exercise_dates
                )
            return (
                DerivativeContract(payoff=payoff_obj, exercise_rights=rights_obj),
                payoff_obj,
                rights_obj,
            )

        def price_with_settings(
            engine_name: str, settings: dict[str, Any]
        ) -> tuple[Any, PricingResult, Any, Any]:
            contract, payoff_obj, rights_obj = make_pricing_contract()
            market_data = MarketData(
                risk_free_rate=risk_free, dividend_yield=dividend_yield
            )
            process = GeometricBrownianMotion(volatility=volatility)
            if engine_name == "Monte Carlo (LSM)":
                engine = MonteCarloEngine(
                    n_paths=settings["n_paths"],
                    n_steps=settings["n_steps"],
                    seed=settings["seed"],
                    poly_degree=settings["poly_degree"],
                )
            elif engine_name == "Binomial Tree (CRR)":
                engine = BinomialEngine(tree_depth=settings["tree_depth"])
            else:
                engine = FiniteDifferenceEngine(
                    n_space=settings["n_space"],
                    n_time=settings["n_time"],
                    theta=settings["theta"],
                    grid_type=settings["grid_type"],
                    s_max=settings.get("s_max"),
                    log_width_sigma=settings.get("log_width_sigma", 5.0),
                )
            result = engine.price(
                derivative=contract,
                process=process,
                initial_spot=spot_price,
                market_data=market_data,
            )
            return engine, result, payoff_obj, rights_obj

        def describe_engine_settings(
            engine_name: str, settings: dict[str, Any]
        ) -> str:
            if engine_name == "Monte Carlo (LSM)":
                return (
                    f"{settings['n_paths']:,} paths · {settings['n_steps']} steps · "
                    f"degree {settings['poly_degree']} · seed {settings['seed']}"
                )
            if engine_name == "Binomial Tree (CRR)":
                return f"{settings['tree_depth']:,} tree steps"
            theta_labels = {0.0: "Explicit", 0.5: "Crank–Nicolson", 1.0: "Implicit"}
            grid_description = (
                f"log grid ({settings['log_width_sigma']:g}σ)"
                if settings["grid_type"] == "log"
                else f"spot grid (Smax {settings['s_max']:g})"
            )
            return (
                f"{theta_labels[settings['theta']]} · {settings['n_space']:,} "
                f"space · {settings['n_time']:,} time · {grid_description}"
            )

        def engine_color_family(engine_name: str) -> str:
            if engine_name == "Monte Carlo (LSM)":
                return "monte_carlo"
            if engine_name == "Binomial Tree (CRR)":
                return "binomial"
            return "finite_difference"

        def engine_shade(
            engine_name: str, occurrence: int, total: int
        ) -> tuple[str, str, str]:
            family_hues = {
                "monte_carlo": 174 / 360,
                "binomial": 40 / 360,
                "finite_difference": 278 / 360,
            }
            hue = family_hues[engine_color_family(engine_name)]
            shade_fraction = occurrence / max(total - 1, 1)
            background_lightness = (
                0.2 + 0.18 * shade_fraction
                if is_dark
                else 0.95 - 0.15 * shade_fraction
            )
            foreground_lightness = 0.78 if is_dark else 0.34
            accent_lightness = (
                0.72 - 0.12 * shade_fraction
                if is_dark
                else 0.42 - 0.12 * shade_fraction
            )
            background = colorsys.hls_to_rgb(
                hue, background_lightness, 0.38
            )
            foreground = colorsys.hls_to_rgb(
                hue, foreground_lightness, 0.62
            )
            accent = colorsys.hls_to_rgb(
                hue, accent_lightness, 0.58
            )

            def to_hex(rgb: tuple[float, float, float]) -> str:
                return "#{:02X}{:02X}{:02X}".format(
                    *(round(channel * 255) for channel in rgb)
                )

            return to_hex(background), to_hex(foreground), to_hex(accent)

        contract_signature = (
            payoff_type, option_kind, spot_price, strike_price, barrier_level,
            cash_payout, exercise_type, maturity, bermudan_freq, volatility,
            risk_free, dividend_yield,
        )
        if st.session_state.get("valuation_contract_signature") != contract_signature:
            st.session_state.valuation_contract_signature = contract_signature
            st.session_state.valuation_runs = []
            st.session_state.valuation_reference_run_id = None
            st.session_state.valuation_editor_open = False
            st.session_state.valuation_editing_run_id = None
        if not isinstance(st.session_state.get("valuation_runs"), list):
            st.session_state.valuation_runs = []
        for obsolete_key in (
            "valuation_primary_record",
            "valuation_comparisons",
            "valuation_primary_engine",
            "valuation_comparison_engine",
            "last_result",
        ):
            st.session_state.pop(obsolete_key, None)
        st.session_state.pop("valuation_dialog_open", None)
        if "valuation_editor_counter" not in st.session_state:
            st.session_state.valuation_editor_counter = 0

        def open_pricing_editor(editing_run_id: int | None = None) -> None:
            st.session_state.valuation_editor_counter += 1
            st.session_state.valuation_editor_id = (
                st.session_state.valuation_editor_counter
            )
            st.session_state.valuation_editing_run_id = editing_run_id
            st.session_state.valuation_editor_open = True
            st.session_state.valuation_editor_step = (
                "parameters" if editing_run_id is not None else "engine"
            )
            editing_record = next(
                (
                    record
                    for record in st.session_state.valuation_runs
                    if record["id"] == editing_run_id
                ),
                None,
            )
            st.session_state.valuation_selected_engine = (
                editing_record["engine_name"]
                if editing_record
                else engine_options[0]
            )

        all_runs = st.session_state.valuation_runs
        for run_index, record in enumerate(all_runs):
            record.setdefault("run_name", f"Pricing {run_index + 1}")
        black_scholes_result: PricingResult | None = None
        black_scholes_supported = (
            pricing_ready
            and exercise_type == "European"
            and payoff_type in ("Vanilla Payoff", "Digital Cash-or-Nothing")
        )
        if black_scholes_supported:
            contract, _, _ = make_pricing_contract()
            black_scholes_result = AnalyticalEngine().price(
                derivative=contract,
                process=GeometricBrownianMotion(volatility=volatility),
                initial_spot=spot_price,
                market_data=MarketData(
                    risk_free_rate=risk_free,
                    dividend_yield=dividend_yield,
                ),
            )
        reference_run_id = st.session_state.get("valuation_reference_run_id")
        reference_index = next(
            (
                index
                for index, record in enumerate(all_runs)
                if record["id"] == reference_run_id
            ),
            0,
        )
        family_counts: dict[str, int] = {}
        for record in all_runs:
            family = engine_color_family(record["engine_name"])
            family_counts[family] = family_counts.get(family, 0) + 1
        payoff_label = {
            "Vanilla Payoff": "Vanilla",
            "Digital Cash-or-Nothing": "Digital",
            "Asian (Running Average)": "Asian average",
            "Barrier Knock-Out": "Barrier knock-out",
        }[payoff_type]
        overview_ink = "#e6e7e9" if is_dark else "#183247"
        overview_muted = "#9da7b2" if is_dark else "#61727f"
        overview_rule = "#363c46" if is_dark else "#dfe5e9"
        overview_accent = "#91cfc2" if is_dark else "#176b68"
        contract_choices = [
            ("Payoff", f"{payoff_label} {option_kind}"),
            ("Exercise", exercise_type),
            ("Spot", f"${spot_price:g}"),
            ("Strike", f"${strike_price:g}"),
            ("Maturity", f"{maturity:g}Y"),
            ("Volatility", f"{volatility:.1%}"),
            ("Risk-free rate", f"{risk_free:.1%}"),
            ("Dividend yield", f"{dividend_yield:.1%}"),
        ]
        if exercise_type == "Bermudan":
            contract_choices.append(("Frequency", bermudan_freq))
        if payoff_type == "Digital Cash-or-Nothing":
            contract_choices.append(("Cash payout", f"${cash_payout:g}"))
        elif payoff_type == "Barrier Knock-Out":
            contract_choices.append(("Barrier", f"${barrier_level:g}"))
        contract_choice_markup = "".join(
            (
                '<div style="min-width:0;padding:0 10px;'
                f'border-left:1px solid {overview_rule};">'
                f'<div style="color:{overview_muted};font:600 0.64rem/1.35 '
                'Inter,system-ui,sans-serif;letter-spacing:0.07em;'
                f'text-transform:uppercase;">{html.escape(label)}</div>'
                f'<div style="margin-top:3px;color:{overview_ink};'
                'font:500 0.9rem/1.35 Inter,system-ui,sans-serif;'
                f'overflow-wrap:anywhere;">{html.escape(value)}</div>'
                '</div>'
            )
            for label, value in contract_choices
        )
        st.markdown(
            f"""
            <div style="display:grid;
                        grid-template-columns:minmax(220px,0.72fr)
                                             minmax(0,1.9fr);
                        align-items:center;gap:32px;margin:26px 0 32px;
                        padding-left:12px;
                        padding-bottom:20px;border-bottom:1px solid {overview_rule};">
              <div>
                <div style="color:{overview_ink};
                            font:400 clamp(2rem,3vw,2.65rem)/1.08
                            Georgia,'Times New Roman',serif;
                            letter-spacing:-0.035em;">
                  Pricing overview
                </div>
              </div>
              <div style="width:fit-content;max-width:100%;margin-left:auto;">
                <div style="margin-bottom:8px;color:{overview_accent};
                            font:700 0.62rem/1.4 Inter,system-ui,sans-serif;
                            letter-spacing:0.12em;text-align:left;
                            text-transform:uppercase;">
                  Contract parameters
                </div>
                <div style="display:flex;width:100%;flex-wrap:wrap;
                            align-items:center;justify-content:flex-start;
                            column-gap:4px;row-gap:8px;">
                  {contract_choice_markup}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if all_runs or black_scholes_result is not None:
            reference_record = all_runs[reference_index] if all_runs else None
            reference_price = (
                black_scholes_result.price
                if black_scholes_result is not None
                else reference_record["result"].price
            )
            run_prices = [record["result"].price for record in all_runs]
            if black_scholes_result is not None:
                run_prices.append(black_scholes_result.price)
            lowest_price = min(run_prices)
            highest_price = max(run_prices)
            price_spread = highest_price - lowest_price
            spread_percent = (
                price_spread / abs(reference_price) * 100
                if reference_price != 0
                else None
            )
            fastest_record = (
                min(all_runs, key=lambda record: record["result"].runtime)
                if all_runs
                else None
            )
            if black_scholes_result is not None:
                spread_detail = (
                    "Black–Scholes reference only"
                    if not all_runs
                    else (
                        f"Black–Scholes + {len(all_runs)} numerical run(s) · "
                        f"difference ${price_spread:.4f}"
                        + (
                            f" ({spread_percent:.2f}%)"
                            if spread_percent is not None
                            else ""
                        )
                    )
                )
                reference_detail = "European Black–Scholes analytical value"
            else:
                spread_detail = (
                    f"Difference ${price_spread:.4f} "
                    f"({spread_percent:.2f}%) · {len(all_runs)} pricing runs"
                    if spread_percent is not None
                    else f"Difference ${price_spread:.4f} · "
                    f"{len(all_runs)} pricing runs"
                )
                reference_detail = (
                    f"{reference_record['run_name']} · "
                    f"{reference_record['engine_name']}"
                )
            price_range_label = (
                f"${lowest_price:.4f}"
                if lowest_price == highest_price
                else f"${lowest_price:.4f} – ${highest_price:.4f}"
            )
            overview_columns = st.columns(3, gap="small")
            overview_values = (
                (
                    "Reference price",
                    f"${reference_price:.4f}",
                    reference_detail,
                ),
                (
                    "Price range",
                    price_range_label,
                    spread_detail,
                ),
                (
                    "Fastest runtime",
                    (
                        f"{fastest_record['result'].runtime * 1000:.2f} ms"
                        if fastest_record is not None
                        else "—"
                    ),
                    (
                        f"{fastest_record['run_name']} · "
                        f"{fastest_record['engine_name']}"
                        if fastest_record is not None
                        else "Run a numerical method to compare"
                    ),
                ),
            )
            for column_index, (column, (label, value, detail)) in enumerate(
                zip(overview_columns, overview_values)
            ):
                with column:
                    right_rule = (
                        f"border-right:1px solid {overview_rule};"
                        if column_index < len(overview_values) - 1
                        else ""
                    )
                    st.markdown(
                        f"""
                        <div style="min-height:112px;padding:22px 14px 12px;
                                    display:flex;flex-direction:column;
                                    align-items:center;justify-content:center;
                                    text-align:center;{right_rule}">
                          <div style="color:{overview_muted};
                                      font:700 0.62rem/1.4 Inter,system-ui,sans-serif;
                                      letter-spacing:0.12em;text-transform:uppercase;">
                            {label}
                          </div>
                          <div style="margin-top:8px;color:{overview_ink};
                                      font:400 clamp(1.25rem,2vw,1.8rem)/1.2
                                      Georgia,'Times New Roman',serif;">
                            {value}
                          </div>
                          <div style="margin-top:6px;color:{overview_muted};
                                      font-size:0.76rem;line-height:1.35;">
                            {detail}
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        if not all_runs:
            prompt_column, action_column = st.columns(
                [5, 1], vertical_alignment="center", gap="small"
            )
            with prompt_column:
                st.caption(
                    "Run a pricing method to establish a reference result."
                    if pricing_ready
                    else f"Complete {missing_drawers_label} to enable pricing."
                )
            with st.container(horizontal=True, horizontal_alignment="center"):
                if st.button(
                    "Add pricing",
                    type="tertiary",
                    width="content",
                    disabled=not pricing_ready,
                    key="valuation_add_pricing",
                ):
                    open_pricing_editor()

        def dismiss_pricing_editor() -> None:
            st.session_state.valuation_editor_open = False
            st.session_state.valuation_editing_run_id = None

        if st.session_state.get("valuation_editor_open", False):
            engine_descriptions = {
                "Monte Carlo (LSM)": (
                    "Simulation-based valuation for early-exercise contracts.",
                    "Monte Carlo",
                ),
                "Binomial Tree (CRR)": (
                    "Tree-based valuation with configurable time steps.",
                    "Binomial tree",
                ),
                "Finite Difference (θ-scheme)": (
                    "PDE-based valuation with configurable numerical grids.",
                    "Finite difference",
                ),
            }

            @st.dialog(
                "Configure a pricing run",
                width="large",
                on_dismiss=dismiss_pricing_editor,
            )
            def render_pricing_editor() -> None:
                editing_run_id = st.session_state.valuation_editing_run_id
                editing_record = next(
                    (
                        record
                        for record in all_runs
                        if record["id"] == editing_run_id
                    ),
                    None,
                )
                if (
                    editing_record
                    and editing_record["engine_name"] not in engine_options
                ):
                    st.session_state.valuation_editing_run_id = None
                    editing_record = None

                selected_engine = st.session_state.get(
                    "valuation_selected_engine", engine_options[0]
                )
                if selected_engine not in engine_options:
                    selected_engine = engine_options[0]
                    st.session_state.valuation_selected_engine = selected_engine

                editor_id = st.session_state.valuation_editor_id
                editor_step = st.session_state.get(
                    "valuation_editor_step", "engine"
                )
                selected_family = engine_color_family(selected_engine)

                step_number = "01 / 02" if editor_step == "engine" else "02 / 02"
                step_description = (
                    "Step 1 of 2 · Select a valuation method"
                    if editor_step == "engine"
                    else "Step 2 of 2 · Configure parameters"
                )
                card_surface = "#172126" if is_dark else "#FFFFFF"
                card_border = "#3A454B" if is_dark else "#D9DEE1"
                engine_accents = {
                    "monte_carlo": "#55B8A8" if is_dark else "#177E78",
                    "binomial": "#E0AB54" if is_dark else "#A66B18",
                    "finite_difference": "#A99AE3" if is_dark else "#6256A4",
                }
                dialog_surface = "#171A20" if is_dark else "#FFFFFF"
                dialog_text = "#E6E7E9" if is_dark else "#253746"
                dialog_muted_text = "#AEB3BB" if is_dark else "#64748B"
                engine_card_styles = []
                for engine_name in engine_options:
                    family = engine_color_family(engine_name)
                    accent = engine_accents[family]
                    engine_card_styles.append(
                        f"""
                        [role="dialog"] .st-key-valuation-engine-card-{family} {{
                            background-color: {card_surface};
                            border-color: {card_border} !important;
                            border-radius: 10px;
                            gap: 8px !important;
                            padding: 12px !important;
                        }}
                        [role="dialog"] .st-key-valuation-engine-card-{family}
                        .valuation-engine-card-title strong {{
                            color: {accent};
                        }}
                        [role="dialog"] .st-key-valuation-engine-card-{family}
                        [data-testid="stCaptionContainer"] {{
                            margin-top: 15px !important;
                        }}
                        [role="dialog"] .st-key-valuation_select_engine_{editor_id}_{family}
                        button {{
                            width: 100%;
                            justify-content: flex-start;
                            padding: 0.2rem 0 !important;
                            color: {accent} !important;
                            background: transparent !important;
                            border: 0 !important;
                            box-shadow: none !important;
                            font-weight: 550;
                        }}
                        [role="dialog"] .st-key-valuation_select_engine_{editor_id}_{family}
                        button > div,
                        [role="dialog"] .st-key-valuation_select_engine_{editor_id}_{family}
                        button > div > span {{
                            width: 100%;
                            justify-content: flex-start !important;
                            text-align: left !important;
                        }}
                        [role="dialog"] .st-key-valuation_select_engine_{editor_id}_{family}
                        button p,
                        [role="dialog"] .st-key-valuation_select_engine_{editor_id}_{family}
                        button span {{
                            color: {accent} !important;
                        }}
                        [role="dialog"] .st-key-valuation_select_engine_{editor_id}_{family}
                        button:hover {{
                            color: {accent} !important;
                            background: transparent !important;
                            text-decoration: underline;
                        }}
                        """
                    )
                st.markdown(
                    f"""
                    <style>
                    div.stDialog:has([role="dialog"]) {{
                        align-items: center !important;
                        background: {
                            "rgba(0, 0, 0, 0.68)" if is_dark
                            else "rgba(153, 173, 194, 0.25)"
                        } !important;
                    }}
                    div:has(> [role="dialog"]) {{
                        width: min(760px, calc(100vw - 24px)) !important;
                        max-width: calc(100vw - 24px) !important;
                        margin-inline: auto !important;
                        background: {dialog_surface} !important;
                        color: {dialog_text} !important;
                    }}
                    [role="dialog"] {{
                        --pricing-engine-accent: {engine_accents[selected_family]};
                        background: {dialog_surface} !important;
                        color: {dialog_text} !important;
                        width: 100% !important;
                        max-width: calc(100vw - 24px) !important;
                        box-sizing: border-box;
                        margin-inline: auto !important;
                    }}
                    [role="dialog"] button[aria-label="Close"] {{
                        color: {dialog_text} !important;
                    }}
                    [role="dialog"] > h2[slot="title"] {{
                        padding: 12px 54px 0 16px !important;
                        color: {dialog_text} !important;
                        font-size: 1.2rem !important;
                        font-weight: 600 !important;
                        letter-spacing: -0.025em;
                        line-height: 1.3;
                    }}
                    [role="dialog"] > div {{
                        padding: 4px 16px 8px !important;
                    }}
                    [role="dialog"] h3 {{
                        margin-top: 0.6rem;
                        margin-bottom: 0.55rem;
                        color: {dialog_text} !important;
                        font-size: 1rem !important;
                        font-weight: 600 !important;
                        letter-spacing: -0.015em;
                        line-height: 1.35;
                    }}
                    [role="dialog"] [data-testid="stSlider"]
                    [data-rac][data-orientation="horizontal"]
                    [data-rac]:not([data-orientation]) {{
                        background-color: {engine_accents[selected_family]} !important;
                        border-color: {engine_accents[selected_family]} !important;
                    }}
                    [role="dialog"] .st-key-valuation-engine-settings-{editor_id}
                    [data-testid="stHorizontalBlock"] {{
                        min-height: 68px;
                    }}
                    [role="dialog"] hr {{
                        margin: 6px 0 !important;
                    }}
                    [role="dialog"] .st-key-valuation-slider-style-{editor_id} {{
                        display: none !important;
                    }}
                    [role="dialog"] [data-testid="stLayoutWrapper"]:has(
                        > .stVerticalBlock.st-key-valuation-slider-style-{editor_id}
                    ) {{
                        display: none !important;
                    }}
                    [role="dialog"] [data-testid="stWidgetLabel"],
                    [role="dialog"] [data-testid="stWidgetLabel"] p {{
                        color: {dialog_muted_text} !important;
                    }}
                    [role="dialog"] [role="radiogroup"]
                    button[data-selected="true"] {{
                        background-color: {engine_accents[selected_family]} !important;
                        border-color: {engine_accents[selected_family]} !important;
                        color: #FFFFFF !important;
                    }}
                    [role="dialog"] [role="radiogroup"]
                    button[data-selected="true"] * {{
                        color: #FFFFFF !important;
                    }}
                    [role="dialog"] [role="radiogroup"]
                    button[role="radio"][aria-checked="false"] {{
                        background-color: {"#252A33" if is_dark else "#F5F7F8"} !important;
                        border-color: {"#3A454B" if is_dark else "#D9DEE1"} !important;
                        color: {"#D6DEE3" if is_dark else "#253746"} !important;
                    }}
                    [role="dialog"] [role="radiogroup"]
                    button[role="radio"][aria-checked="false"] * {{
                        color: {"#D6DEE3" if is_dark else "#253746"} !important;
                    }}
                    [role="dialog"] [role="radiogroup"]
                    button[role="radio"][aria-checked="false"]:hover {{
                        background-color: {"#303741" if is_dark else "#EDF1F3"} !important;
                    }}
                    [role="dialog"] [data-testid="stNumberInput"]
                    [data-baseweb="input"]:has(input:focus) {{
                        border-color: {engine_accents[selected_family]} !important;
                        box-shadow: 0 0 0 1px {engine_accents[selected_family]} !important;
                    }}
                    [role="dialog"] [data-testid="stTooltipHoverTarget"] button {{
                        color: {engine_accents[selected_family]} !important;
                    }}
                    [role="dialog"] .valuation-dialog-step {{
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        gap: 12px;
                        margin: 0 0 12px;
                        padding: 0 0 10px;
                        border-bottom: 1px solid {card_border};
                        color: #64748B;
                        font-size: 0.9rem;
                    }}
                    [role="dialog"] .valuation-dialog-step-number {{
                        flex: 0 0 auto;
                        padding: 2px 9px;
                        border-radius: 999px;
                        background: {"#29343B" if is_dark else "#F1F3F4"};
                        color: {"#D6DEE3" if is_dark else "#334155"};
                        font-size: 0.78rem;
                        font-weight: 600;
                    }}
                    [role="dialog"] .valuation-selected-engine {{
                        color: var(--engine-accent);
                        font-size: 1rem;
                        font-weight: 600;
                        letter-spacing: -0.015em;
                    }}
                    [role="dialog"] .st-key-valuation_back_to_engines_{editor_id}
                    button {{
                        padding: 0.2rem 0 !important;
                        color: {engine_accents[selected_family]} !important;
                        background: transparent !important;
                        border: 0 !important;
                        box-shadow: none !important;
                        font-size: 0.85rem;
                    }}
                    [role="dialog"] .st-key-valuation_back_to_engines_{editor_id}
                    button p,
                    [role="dialog"] .st-key-valuation_back_to_engines_{editor_id}
                    button span {{
                        color: {engine_accents[selected_family]} !important;
                    }}
                    [role="dialog"] .st-key-valuation_back_to_engines_{editor_id}
                    button:hover {{
                        background: transparent !important;
                        text-decoration: underline;
                    }}
                    [role="dialog"] .st-key-valuation-engine-header-{editor_id} {{
                        align-items: center !important;
                        gap: 10px !important;
                    }}
                    [role="dialog"] .st-key-valuation-engine-header-{editor_id}
                    [data-testid="stElementContainer"]:has(.valuation-selected-engine) {{
                        align-self: center;
                        margin: 0 !important;
                    }}
                    [role="dialog"] .st-key-valuation-engine-header-{editor_id}
                    .stMarkdown > div > div {{
                        margin-bottom: 0 !important;
                    }}
                    [role="dialog"] .st-key-valuation_cancel_editor_{editor_id}
                    button {{
                        padding: 0.25rem 0 !important;
                        color: {dialog_muted_text} !important;
                        background: transparent !important;
                        border: 0 !important;
                        box-shadow: none !important;
                        font-weight: 500;
                    }}
                    [role="dialog"] .st-key-valuation_cancel_editor_{editor_id}
                    button p,
                    [role="dialog"] .st-key-valuation_cancel_editor_{editor_id}
                    button span {{
                        color: {dialog_muted_text} !important;
                    }}
                    [role="dialog"] .st-key-valuation_cancel_editor_{editor_id}
                    button:hover {{
                        background: transparent !important;
                        text-decoration: underline;
                    }}
                    [role="dialog"] .st-key-valuation_run_editor_{editor_id}
                    button {{
                        padding: 0.25rem 0 !important;
                        color: {engine_accents[selected_family]} !important;
                        background: transparent !important;
                        border: 0 !important;
                        box-shadow: none !important;
                        font-weight: 600;
                    }}
                    [role="dialog"] .st-key-valuation_run_editor_{editor_id}
                    button p,
                    [role="dialog"] .st-key-valuation_run_editor_{editor_id}
                    button span {{
                        color: {engine_accents[selected_family]} !important;
                    }}
                    [role="dialog"] .st-key-valuation_run_editor_{editor_id}
                    button:hover {{
                        background: transparent !important;
                        text-decoration: underline;
                    }}
                    [role="dialog"] .valuation-engine-card-title {{
                        display: flex;
                        align-items: center;
                        gap: 10px;
                    }}
                    [role="dialog"]
                    .st-key-valuation_cancel_engine_selection_{editor_id} button {{
                        padding: 0.25rem 0 !important;
                        color: {dialog_muted_text} !important;
                        background: transparent !important;
                        border: 0 !important;
                        box-shadow: none !important;
                    }}
                    [role="dialog"]
                    .st-key-valuation_cancel_engine_selection_{editor_id} button:hover {{
                        color: #334155 !important;
                        background: transparent !important;
                        text-decoration: underline;
                    }}
                    {"".join(engine_card_styles)}
                    @media (max-width: 600px) {{
                        [role="dialog"] > h2[slot="title"] {{
                            padding: 12px 68px 0 14px !important;
                            font-size: 1.1rem !important;
                        }}
                        [role="dialog"] > div {{
                            padding: 4px 12px 8px !important;
                        }}
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="valuation-dialog-step">
                        <span>{step_description}</span>
                        <span class="valuation-dialog-step-number">
                            {step_number}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if editor_step == "engine":
                    st.subheader("Choose a pricing engine")
                    engine_columns = st.columns(
                        len(engine_options), gap="small"
                    )
                    for column, engine_name in zip(
                        engine_columns, engine_options
                    ):
                        description, short_label = engine_descriptions[
                            engine_name
                        ]
                        family = engine_color_family(engine_name)
                        with column:
                            with st.container(
                                key=f"valuation-engine-card-{family}",
                                border=True,
                            ):
                                icon_svg = {
                                    "monte_carlo": (
                                        '<svg viewBox="0 0 20 20" aria-hidden="true">'
                                        '<path d="M3 16.5h14M3.5 16V4" fill="none" '
                                        'stroke="currentColor" stroke-width="1.5"/>'
                                        '<path d="m4.5 13 3-3.2 2.4 1.6 4.6-6 2.1 1.4" '
                                        'fill="none" stroke="currentColor" '
                                        'stroke-width="1.8" stroke-linecap="round" '
                                        'stroke-linejoin="round"/></svg>'
                                    ),
                                    "binomial": (
                                        '<svg viewBox="0 0 20 20" aria-hidden="true">'
                                        '<path d="M5 4v6l5 3 5-6" fill="none" '
                                        'stroke="currentColor" stroke-width="1.8"/>'
                                        '<circle cx="5" cy="4" r="1.7" fill="none" '
                                        'stroke="currentColor" stroke-width="1.5"/>'
                                        '<circle cx="10" cy="13" r="1.7" fill="none" '
                                        'stroke="currentColor" stroke-width="1.5"/>'
                                        '<circle cx="15" cy="7" r="1.7" fill="none" '
                                        'stroke="currentColor" stroke-width="1.5"/></svg>'
                                    ),
                                    "finite_difference": (
                                        '<svg viewBox="0 0 20 20" aria-hidden="true">'
                                        '<rect x="3" y="3" width="14" height="14" '
                                        'rx="2" fill="none" stroke="currentColor" '
                                        'stroke-width="1.8"/><path d="M3 8h14M8 3v14" '
                                        'fill="none" stroke="currentColor" '
                                        'stroke-width="1.5"/></svg>'
                                    ),
                                }[family]
                                st.markdown(
                                    f'<div class="valuation-engine-card-title">'
                                    f'<span style="width:20px;height:20px;'
                                    f'color:{engine_accents[family]};display:inline-flex">'
                                    f'{icon_svg}</span><strong>{short_label}</strong></div>',
                                    unsafe_allow_html=True,
                                )
                                st.caption(description)
                                engine_selected = st.button(
                                    "Select engine →",
                                    key=(
                                        f"valuation_select_engine_{editor_id}_"
                                        f"{family}"
                                    ),
                                    type="primary",
                                    width="stretch",
                                )
                                if engine_selected:
                                    st.session_state.valuation_selected_engine = (
                                        engine_name
                                    )
                                    st.session_state.valuation_editor_step = (
                                        "parameters"
                                    )
                                    st.rerun()

                    with st.container(
                        horizontal=True,
                        horizontal_alignment="distribute",
                        vertical_alignment="center",
                        gap="small",
                        key=f"valuation-editor-footer-{editor_id}",
                    ):
                        cancel_selection = st.button(
                            "Cancel",
                            key=f"valuation_cancel_engine_selection_{editor_id}",
                            width="content",
                        )
                    if cancel_selection:
                        st.session_state.valuation_editor_open = False
                        st.session_state.valuation_editing_run_id = None
                        st.rerun()
                    return

                family = selected_family
                selected_accent = engine_accents[family]
                with st.container(
                    horizontal=True,
                    horizontal_alignment="left",
                    vertical_alignment="center",
                    gap="small",
                    key=f"valuation-engine-header-{editor_id}",
                ):
                    st.markdown(
                        f'<div class="valuation-selected-engine" '
                        f'style="--engine-accent:{selected_accent}">'
                        f'{selected_engine}</div>',
                        unsafe_allow_html=True,
                    )

                settings_initial = (
                    editing_record["settings"]
                    if editing_record
                    and selected_engine == editing_record["engine_name"]
                    else None
                )
                settings_prefix = (
                    f"valuation_editor_{editor_id}_{family}"
                )
                with st.container(
                    key=f"valuation-engine-settings-{editor_id}"
                ):
                    settings = render_engine_settings(
                        selected_engine, settings_prefix, settings_initial
                    )
                with st.container(key=f"valuation-slider-style-{editor_id}"):
                    st.markdown(
                        f"<style>{''.join(slider_track_styles)}</style>",
                        unsafe_allow_html=True,
                    )

                st.divider()
                with st.container(
                    horizontal=True,
                    horizontal_alignment="distribute",
                    vertical_alignment="center",
                    gap="small",
                    key=f"valuation-editor-footer-{editor_id}",
                ):
                    cancel_editor = st.button(
                        "Cancel",
                        key=f"valuation_cancel_editor_{editor_id}",
                        width="content",
                    )
                    run_label = (
                        "Update pricing" if editing_record else "Run pricing"
                    )
                    with st.container(
                        horizontal=True,
                        horizontal_alignment="right",
                        vertical_alignment="center",
                        gap="small",
                        key=f"valuation-editor-footer-actions-{editor_id}",
                    ):
                        change_engine = st.button(
                            "Change engine →",
                            key=f"valuation_back_to_engines_{editor_id}",
                            width="content",
                        )
                        run_selected = st.button(
                            run_label,
                            key=f"valuation_run_editor_{editor_id}",
                            type="primary",
                            width="content",
                            disabled=not pricing_ready,
                        )
                if change_engine:
                    st.session_state.valuation_editor_step = "engine"
                    st.rerun()
                if cancel_editor:
                    st.session_state.valuation_editor_open = False
                    st.session_state.valuation_editing_run_id = None
                    st.rerun()
                if run_selected:
                    try:
                        _, result, _, _ = price_with_settings(
                            selected_engine, settings
                        )
                        record = {
                            "id": (
                                editing_record["id"]
                                if editing_record
                                else editor_id
                            ),
                            "run_name": (
                                editing_record["run_name"]
                                if editing_record
                                else f"Pricing {len(all_runs) + 1}"
                            ),
                            "engine_name": selected_engine,
                            "settings": settings.copy(),
                            "settings_label": describe_engine_settings(
                                selected_engine, settings
                            ),
                            "result": result,
                        }
                        if editing_record:
                            run_index = next(
                                index
                                for index, existing in enumerate(all_runs)
                                if existing["id"] == editing_record["id"]
                            )
                            all_runs[run_index] = record
                        else:
                            all_runs.append(record)
                        st.session_state.valuation_runs = all_runs
                        st.session_state.valuation_editor_open = False
                        st.session_state.valuation_editing_run_id = None
                        st.rerun()
                    except Exception as error:
                        st.error(f"{selected_engine} valuation failed: {error}")

            render_pricing_editor()

        if all_runs:
            st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
            comparison_heading, add_column = st.columns(
                [5, 1], vertical_alignment="center", gap="small"
            )
            with comparison_heading:
                st.markdown(
                    '<div style="margin:0 0 10px;">'
                    f'<div style="color:{overview_ink};'
                    'font:500 1.05rem/1.3 Inter,system-ui,sans-serif;'
                    'letter-spacing:-0.01em;">Pricing comparison</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            with add_column:
                if st.button(
                    "Add pricing",
                    type="tertiary",
                    width="content",
                    disabled=not pricing_ready,
                    key="valuation_add_pricing",
                ):
                    open_pricing_editor()
                    st.rerun()
            comparison_rows = []
            comparison_engine_styles = []
            family_occurrences = {}
            reference_price = (
                black_scholes_result.price
                if black_scholes_result is not None
                else all_runs[reference_index]["result"].price
            )
            if black_scholes_result is not None:
                comparison_rows.append(
                    {
                        "Run": "Black–Scholes",
                        "Role": "Reference",
                        "Engine": "Black–Scholes",
                        "Configuration": "Closed-form European value",
                        "Price": f"${black_scholes_result.price:.4f}",
                        "Δ vs reference": "—",
                        "Runtime": (
                            f"{black_scholes_result.runtime * 1000:.2f} ms"
                        ),
                    }
                )
                comparison_engine_styles.append(
                    (overview_accent, overview_accent)
                )
            reference_row_index = (
                0 if black_scholes_result is not None else reference_index
            )
            for run_index, record in enumerate(all_runs):
                family = engine_color_family(record["engine_name"])
                occurrence = family_occurrences.get(family, 0)
                family_occurrences[family] = occurrence + 1
                _, text_color, point_color = engine_shade(
                    record["engine_name"], occurrence, family_counts[family]
                )
                result = record["result"]
                reference_delta = result.price - reference_price
                reference_delta_percent = (
                    reference_delta / abs(reference_price) * 100
                    if reference_price != 0
                    else None
                )
                comparison_engine_styles.append((text_color, point_color))
                comparison_rows.append(
                    {
                        "Run": record["run_name"],
                        "Role": (
                            "Comparison"
                            if black_scholes_result is not None
                            else (
                                "Reference"
                                if run_index == reference_index
                                else "Comparison"
                            )
                        ),
                        "Engine": record["engine_name"],
                        "Configuration": record["settings_label"],
                        "Price": f"${result.price:.4f}",
                        "Δ vs reference": (
                            "—"
                            if (
                                black_scholes_result is None
                                and run_index == reference_index
                            )
                            else (
                                f"${reference_delta:+.4f} "
                                f"({reference_delta_percent:+.2f}%)"
                                if reference_delta_percent is not None
                                else f"${reference_delta:+.4f}"
                            )
                        ),
                        "Runtime": f"{result.runtime * 1000:.2f} ms",
                    }
                )
            table_surface = "#1b1e25" if is_dark else "#ffffff"
            table_header = "#242932" if is_dark else "#f3f6f7"
            table_ink = "#e6e7e9" if is_dark else "#183247"
            table_muted = "#9da7b2" if is_dark else "#61727f"
            table_rule = "#363c46" if is_dark else "#dfe5e9"
            selected_surface = "#263c39" if is_dark else "#e8f2f1"
            table_html = [
                '<div class="valuation-comparison-table-scroll" '
                'style="overflow:hidden;'
                f'border:1px solid {table_rule};border-radius:10px;'
                f'background:{table_surface};">',
                '<table style="width:100%;table-layout:fixed;'
                'border-collapse:collapse;margin:0;font-size:0.82rem;line-height:1.4;">',
                '<colgroup><col style="width:12%"><col style="width:13%">'
                '<col style="width:19%"><col style="width:25%">'
                '<col style="width:10%"><col style="width:13%">'
                '<col style="width:8%"></colgroup>',
                '<thead><tr>',
            ]
            for column_index, label in enumerate(comparison_rows[0]):
                alignment = "left" if column_index < 4 else "center"
                table_html.append(
                    f'<th style="padding:12px 14px;text-align:{alignment};'
                    f'background:{table_header};color:{table_muted};'
                    f'border-bottom:1px solid {table_rule};'
                    'font:600 0.68rem ui-monospace,monospace;'
                    'letter-spacing:0.04em;text-transform:uppercase;'
                    f'{"white-space:nowrap;" if column_index >= 4 else ""}">'
                    f'{html.escape(label)}</th>'
                )
            table_html.append("</tr></thead><tbody>")
            for row_index, row in enumerate(comparison_rows):
                row_surface = (
                    selected_surface if row_index == reference_row_index
                    else table_surface
                )
                row_weight = "600" if row_index == reference_row_index else "400"
                table_html.append("<tr>")
                for column_index, value in enumerate(row.values()):
                    cell_color = table_ink
                    cell_border = ""
                    if column_index == 2:
                        cell_color, engine_accent = (
                            comparison_engine_styles[row_index]
                        )
                        cell_border = (
                            f"border-left:3px solid {engine_accent};"
                        )
                    alignment = "left" if column_index < 4 else "center"
                    table_html.append(
                        f'<td style="padding:11px 14px;text-align:{alignment};'
                        f'background:{row_surface};color:{cell_color};'
                        f'border-bottom:1px solid {table_rule};'
                        f'font-weight:{row_weight};{cell_border}'
                        f'{"white-space:nowrap;" if column_index >= 4 else ""}'
                        f'overflow:hidden;text-overflow:ellipsis;">'
                        f'{html.escape(str(value))}</td>'
                    )
                table_html.append("</tr>")
            table_html.append("</tbody></table></div>")
            if black_scholes_result is None:
                for row_index in range(len(all_runs)):
                    table_html.append(
                        f"<style>.st-key-valuation-comparison-surface "
                        f".st-key-valuation_reference_row_{row_index} {{"
                        f"top:{39.5 + 41.375 * row_index}px !important;"
                        "left:0 !important;width:100% !important;"
                        "height:41.375px !important;}</style>"
                    )
            with st.container(key="valuation-comparison-surface"):
                st.markdown("".join(table_html), unsafe_allow_html=True)
                if black_scholes_result is None:
                    for row_index, record in enumerate(all_runs):
                        st.button(
                            f"Set {record['run_name']} as reference",
                            key=f"valuation_reference_row_{row_index}",
                            on_click=lambda run_id=record["id"]: st.session_state.update(
                                {"valuation_reference_run_id": run_id}
                            ),
                        )
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

        st.stop()
