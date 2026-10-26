"""Render the interactive engine-design and source-browser page."""
from __future__ import annotations

import ast
import builtins
import html
import io
import keyword
from pathlib import Path
from textwrap import dedent
import tokenize
from typing import Any

import streamlit as st



def render_engine_design() -> None:
    """Build the architecture overview, runnable example, and source browser."""
    section_margin = "24px 0 12px"
    hero_margin = "18px auto 0"

    intro_accent = (
        "#91cfc2" if st.session_state.theme_mode == "Dark" else "#176b68"
    )
    intro_text = (
        "#e6e7e9" if st.session_state.theme_mode == "Dark" else "#183247"
    )
    intro_muted = (
        "#aeb3bb" if st.session_state.theme_mode == "Dark" else "#526574"
    )
    
    dark_pipeline = st.session_state.theme_mode == "Dark"
    pipeline_colors = {
        "surface": "#1b1e25" if dark_pipeline else "#ffffff",
        "subtle": "#242932" if dark_pipeline else "#f3f6f7",
        "ink": "#e6e7e9" if dark_pipeline else "#183247",
        "muted": "#9da7b2" if dark_pipeline else "#61727f",
        "rule": "#363c46" if dark_pipeline else "#dfe5e9",
        "accent": "#91cfc2" if dark_pipeline else "#176b68",
    }
    section_break_html = '<div style="height:44px;"></div>'

    def render_engine_section_heading(label: str, target: Any = st) -> None:
        target.markdown(
            f"""
            <div style="margin:{section_margin};padding:0 0 11px;
                        border-bottom:1px solid {pipeline_colors["rule"]};">
              <div role="heading" aria-level="2" style="display:flex;
                   align-items:center;gap:9px;margin:0;color:{pipeline_colors["ink"]};
                   font:600 0.9rem/1.3 Inter,system-ui,sans-serif;">
                <span aria-hidden="true" style="width:3px;height:15px;
                     flex:0 0 3px;border-radius:2px;
                     background:{pipeline_colors["accent"]};"></span>
                <span>{label}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    hero_html = f"""
    <style>
      div[data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"] {{
        gap:12px;
      }}
    </style>
    <div style="max-width:1160px;margin:{hero_margin};">
      <h1 style="max-width:780px;margin:0;color:{intro_text};
                 font:400 clamp(2.2rem,2.9vw,2.85rem)/1.1
                 Georgia,'Times New Roman',serif;
                 letter-spacing:-0.035em;font-kerning:normal;
                 text-wrap:balance;">
        Pricing engine architecture
      </h1>
      <p style="max-width:760px;margin:16px 0 0;color:{intro_muted};
                font:400 1.02rem/1.65 Inter,system-ui,sans-serif;">
        This section introduces the pricing lab's implementation, 
        outlining its core design principles, a quick-start example 
        that illustrates the engine’s modular architecture, and a 
        compatibility table summarizing the supported inputs for each engine.
      </p>
    </div>
    """
    engine_summary_html = f"""
    <div style="margin:18px 0 0;padding:18px 20px;
                border:1px solid {pipeline_colors["rule"]};
                border-radius:14px;background:{pipeline_colors["surface"]};">
      <div style="margin-bottom:13px;color:{intro_accent};
                  font:700 0.68rem/1.4 Inter,system-ui,sans-serif;
                  letter-spacing:0.12em;text-transform:uppercase;">
        Engine coverage
      </div>
      <div style="display:grid;grid-template-columns:1fr;gap:10px;">
        <div style="padding-bottom:9px;border-bottom:1px solid {pipeline_colors["rule"]};">
          <strong style="color:{intro_text};">03</strong>
          <span style="margin-left:8px;color:{intro_muted};">numerical method families</span>
        </div>
        <div style="padding-bottom:9px;border-bottom:1px solid {pipeline_colors["rule"]};">
          <strong style="color:{intro_text};">04</strong>
          <span style="margin-left:8px;color:{intro_muted};">payoff families, including path-dependent ones</span>
        </div>
        <div>
          <strong style="color:{intro_text};">03</strong>
          <span style="margin-left:8px;color:{intro_muted};">exercise-rights types</span>
        </div>
      </div>
    </div>
    """
    hero_copy, hero_summary = st.columns(
        [1.25, 0.75], gap="large", vertical_alignment="center"
    )
    with hero_copy:
        st.markdown(hero_html, unsafe_allow_html=True)
    with hero_summary:
        st.markdown(engine_summary_html, unsafe_allow_html=True)

    st.markdown(section_break_html, unsafe_allow_html=True)
    render_engine_section_heading("Architecture")
    st.markdown(
        """
        QuantPricer achieves modular flexibility through well-defined separation
        of concerns and component isolation. Rather than constructing rigid,
        hyper-specific option classes (e.g., _EuropeanVanillaCall_), a
        **Contract** is independently assembled as a wrapper combining any
        arbitrary **Payoff** with any **Exercise Rights**.

        To compute valuation, the numerical **Engine** consumes this Contract
        (which defines cashflows and exercise boundaries), a **Process**
        (modeling underlying asset dynamics), and **Market** parameters. The
        engine verifies model support across the inputs and evaluates the
        expected present value.
        """
    )
    graph_common_style = f"""
    <style>
      .engine-graph {{ margin:8px 0 0; color:{pipeline_colors["ink"]}; }}
      .engine-graph * {{ box-sizing:border-box; }}
      .engine-graph .node-title {{
        color:{pipeline_colors["ink"]}; font-weight:650; line-height:1.4;
      }}
      .engine-graph .node-note {{
        margin-top:5px; color:{pipeline_colors["muted"]};
        font-size:0.82rem; line-height:1.45;
      }}
      .engine-graph .small-label {{
        color:{pipeline_colors["accent"]}; font-size:0.66rem;
        font-weight:700; letter-spacing:0.1em; text-transform:uppercase;
      }}
      .eg-arrow {{ color:{pipeline_colors["accent"]}; font-size:1.25rem; }}
      .eg-open {{
        display:grid; grid-template-columns:1.2fr 28px 1fr 28px 1fr;
        gap:16px; align-items:center; padding:22px 0;
      }}
      .eg-open-stage {{ min-height:116px; padding-top:12px;
        border-top:2px solid {pipeline_colors["accent"]}; }}
      @media (max-width:760px) {{
        .eg-open {{ grid-template-columns:1fr; gap:10px; }}
        .eg-open-arrow {{ transform:rotate(90deg); text-align:center; }}
        .eg-open-stage {{ min-height:unset; }}
      }}
    </style>
    """
    graph_content = """
    <div class="engine-graph eg-open">
      <div class="eg-open-stage">
        <div class="small-label">Contract specification</div>
        <div class="node-title">Payoff + Exercise</div>
        <div class="node-note">A payoff defines cashflows; exercise rights define when they may be claimed.</div>
      </div>
      <div class="eg-open-arrow eg-arrow" aria-hidden="true">+</div>
      <div class="eg-open-stage">
        <div class="small-label">Assumptions</div>
        <div class="node-title">Process + market</div>
        <div class="node-note">Underlying dynamics, rates, and dividends.</div>
      </div>
      <div class="eg-open-arrow eg-arrow" aria-hidden="true">→</div>
      <div class="eg-open-stage">
        <div class="small-label">Computation</div>
        <div class="node-title">Pricing engine</div>
        <div class="node-note">Support validation followed by present-value estimation.</div>
      </div>
    </div>
    """
    st.markdown(
        dedent(graph_common_style) + dedent(graph_content),
        unsafe_allow_html=True,
    )

    dark_matrix = st.session_state.theme_mode == "Dark"
    matrix_colors = {
        "surface": "#1b1e25" if dark_matrix else "#ffffff",
        "header": "#242932" if dark_matrix else "#f3f6f7",
        "ink": "#e6e7e9" if dark_matrix else "#183247",
        "muted": "#9da7b2" if dark_matrix else "#61727f",
        "rule": "#363c46" if dark_matrix else "#dfe5e9",
        "accent": "#91cfc2" if dark_matrix else "#176b68",
        "empty": "#626c78" if dark_matrix else "#aab5be",
    }
    matrix_rows = (
        ("Vanilla · Put / Call", True, True, True),
        ("Digital · Put / Call", True, True, False),
        ("Asian (arithmetic) · Put / Call", True, False, False),
        ("Knock-out barrier · Put / Call", True, False, False),
        ("European exercise", True, True, True),
        ("Bermudan exercise", True, True, True),
        ("American exercise", True, True, True),
    )
    matrix_label_width = "52%"
    matrix_method_width = "16%"
    matrix_html = [
        '<style>'
        '.qp-product-matrix table,'
        '.qp-product-matrix th,'
        '.qp-product-matrix td {'
        'border:0 !important;'
        'box-shadow:none !important;'
        '}'
        '.qp-product-matrix table {'
        'margin:0 !important;'
        'border:0 !important;'
        'border-collapse:separate !important;'
        'border-spacing:0 !important;'
        '}'
        '.qp-product-matrix tr.qp-matrix-section > th,'
        '.qp-product-matrix tr.qp-matrix-spacer > td {'
        'border:0 !important;'
        'box-shadow:none !important;'
        '}'
        '.qp-product-matrix tr.qp-matrix-spacer > td {'
        'height:5px !important;'
        'padding:0 !important;'
        '}'
        f'.qp-product-matrix tr.qp-matrix-rule > th,'
        f'.qp-product-matrix tr.qp-matrix-rule > td {{'
        f'border-bottom:1px solid {matrix_colors["rule"]} !important;'
        '}'
        '</style>',
        f'<div class="qp-product-matrix" style="overflow:hidden;'
        f'border:1px solid {matrix_colors["rule"]};'
        f'border-radius:12px;background:{matrix_colors["surface"]};">',
        '<table style="width:100%;table-layout:fixed;border-collapse:collapse;'
        'font-size:0.88rem;line-height:1.4;">',
        f'<colgroup><col style="width:{matrix_label_width}">'
        f'<col style="width:{matrix_method_width}">'
        f'<col style="width:{matrix_method_width}">'
        f'<col style="width:{matrix_method_width}"></colgroup>',
        '<thead><tr class="qp-matrix-rule">',
        f'<th style="padding:13px 20px;text-align:left;'
        f'background:{matrix_colors["header"]};color:{matrix_colors["muted"]};'
        f'font:600 0.72rem ui-monospace,monospace;letter-spacing:0.06em;'
        f'text-transform:uppercase;">'
        "Contract component</th>",
    ]
    for method in ("Monte Carlo", "Binomial", "Finite differences"):
        matrix_html.append(
            f'<th style="padding:13px 10px;text-align:center;'
            f'background:{matrix_colors["header"]};color:{matrix_colors["muted"]};'
            f'font:600 0.7rem ui-monospace,monospace;letter-spacing:0.05em;'
            f'text-transform:uppercase;white-space:nowrap;">{method}</th>'
        )
    matrix_html.append("</tr></thead><tbody>")
    for row_index, row in enumerate(matrix_rows[:4]):
        if row_index == 0:
            matrix_html.append(
                f'<tr class="qp-matrix-section"><th colspan="4" style="padding:11px 20px 7px;text-align:left;'
                f'color:{matrix_colors["accent"]};font-size:0.68rem;'
                f'font-weight:700;letter-spacing:0.1em;text-transform:uppercase;'
                f'background:{matrix_colors["surface"]};">Payoff</th></tr>'
            )
        label, *support = row
        matrix_html.append('<tr class="qp-matrix-rule">')
        matrix_html.append(
            f'<td style="padding:11px 20px;color:{matrix_colors["ink"]};">{label}</td>'
        )
        for is_supported in support:
            symbol = "●" if is_supported else "—"
            color = matrix_colors["accent"] if is_supported else matrix_colors["empty"]
            matrix_html.append(
                f'<td aria-label="{"Supported" if is_supported else "Not supported"}" '
                f'style="padding:11px 10px;text-align:center;color:{color};'
                f'font-size:0.82rem;">{symbol}</td>'
            )
        matrix_html.append("</tr>")
    matrix_html.append(
        f'<tr class="qp-matrix-spacer" aria-hidden="true"><td colspan="4" '
        f'style="height:5px;padding:0;background:{matrix_colors["surface"]};"></td></tr>'
        f'<tr class="qp-matrix-section"><th colspan="4" style="padding:11px 20px 7px;text-align:left;'
        f'color:{matrix_colors["accent"]};font-size:0.68rem;'
        f'font-weight:700;letter-spacing:0.1em;text-transform:uppercase;'
        f'background:{matrix_colors["surface"]};">Exercise rights</th></tr>'
    )
    for row_index, row in enumerate(matrix_rows[4:]):
        label, *support = row
        row_class = ' class="qp-matrix-rule"' if row_index < 2 else ""
        matrix_html.append(f"<tr{row_class}>")
        matrix_html.append(
            f'<td style="padding:11px 20px;color:{matrix_colors["ink"]};">{label}</td>'
        )
        for is_supported in support:
            symbol = "●" if is_supported else "—"
            color = matrix_colors["accent"] if is_supported else matrix_colors["empty"]
            matrix_html.append(
                f'<td aria-label="{"Supported" if is_supported else "Not supported"}" '
                f'style="padding:11px 10px;text-align:center;color:{color};'
                f'font-size:0.82rem;">{symbol}</td>'
            )
        matrix_html.append("</tr>")
    matrix_html.append("</tbody></table></div>")
  
    payoff_options = {
        "Vanilla put": (
            "VanillaPayoff",
            "VanillaPayoff(strike=100.0, is_call=False)",
            "all",
        ),
        "Vanilla call": (
            "VanillaPayoff",
            "VanillaPayoff(strike=100.0, is_call=True)",
            "all",
        ),
        "Digital put": (
            "DigitalPayoff",
            "DigitalPayoff(strike=100.0, is_call=False, cash_payout=1.0)",
            "digital",
        ),
        "Digital call": (
            "DigitalPayoff",
            "DigitalPayoff(strike=100.0, is_call=True, cash_payout=1.0)",
            "digital",
        ),
        "Asian put": (
            "AsianPayoff",
            "AsianPayoff(strike=100.0, is_call=False)",
            "path",
        ),
        "Asian call": (
            "AsianPayoff",
            "AsianPayoff(strike=100.0, is_call=True)",
            "path",
        ),
        "Knock-out put": (
            "BarrierPayoff",
            'BarrierPayoff(strike=100.0, is_call=False, barrier_level=130.0, barrier_direction="up")',
            "path",
        ),
        "Knock-out call": (
            "BarrierPayoff",
            'BarrierPayoff(strike=100.0, is_call=True, barrier_level=130.0, barrier_direction="up")',
            "path",
        ),
    }
    exercise_options = {
        "American": (
            "AmericanExercise",
            "AmericanExercise(maturity=1.0)",
        ),
        "European": (
            "EuropeanExercise",
            "EuropeanExercise(maturity=1.0)",
        ),
        "Bermudan · quarterly": (
            "BermudanExercise",
            "BermudanExercise(maturity=1.0, exercise_times=[0.25, 0.5, 0.75])",
        ),
    }
    engine_options = {
        "Monte Carlo · Longstaff–Schwartz": (
            "MonteCarloEngine",
            "MonteCarloEngine(n_paths=10000, n_steps=100, seed=42)",
        ),
        "Binomial · CRR": (
            "BinomialEngine",
            "BinomialEngine(tree_depth=500)",
        ),
        "Finite differences · Crank–Nicolson": (
            "FiniteDifferenceEngine",
            'FiniteDifferenceEngine(n_space=200, n_time=200, theta=0.5, grid_type="log")',
        ),
    }
    engine_modules = {
        "MonteCarloEngine": "monte_carlo",
        "BinomialEngine": "binomial",
        "FiniteDifferenceEngine": "finte_differences",
    }

    original_quick_start = """from pricing.contract import DerivativeContract
from pricing.exercise_rights import AmericanExercise
from pricing.markets import MarketData
from pricing.payoff import VanillaPayoff
from pricing.stochastic_process import GeometricBrownianMotion
from pricing.engines.monte_carlo import MonteCarloEngine

payoff = VanillaPayoff(strike=100.0, is_call=False)
exercise = AmericanExercise(maturity=1.0)
contract = DerivativeContract(payoff, exercise)

market = MarketData(risk_free_rate=0.05, dividend_yield=0.02)
process = GeometricBrownianMotion(volatility=0.2)

engine = MonteCarloEngine(n_paths=10000, n_steps=100, seed=42)
result = engine.price(contract, process, initial_spot=100.0, market_data=market)

print(f"Price: {result.price:.4f} (±{result.std_error:.4f})")"""
    payoff_state_key = "engine_design_payoff_choice"
    exercise_state_key = "engine_design_exercise_choice"
    process_state_key = "engine_design_process_choice"
    payoff_choice = st.session_state.get(payoff_state_key, "Vanilla put")
    exercise_choice = st.session_state.get(exercise_state_key, "American")
    current_payoff = payoff_options[payoff_choice]
    if current_payoff[2] == "path":
        available_engines = ["Monte Carlo · Longstaff–Schwartz"]
    elif current_payoff[2] == "digital":
        available_engines = [
            "Monte Carlo · Longstaff–Schwartz",
            "Binomial · CRR",
        ]
    else:
        available_engines = list(engine_options)
    engine_state_key = "engine_design_engine_choice"
    if st.session_state.get(engine_state_key) not in available_engines:
        st.session_state[engine_state_key] = available_engines[0]
    engine_choice = st.session_state[engine_state_key]

    payoff_class, payoff_expression, _ = current_payoff
    exercise_class, exercise_expression = exercise_options[exercise_choice]
    engine_class, engine_expression = engine_options[engine_choice]
    original_components = {
        "Payoff": (
            "from pricing.payoff import VanillaPayoff",
            "payoff = VanillaPayoff(strike=100.0, is_call=False)",
        ),
        "Exercise rights": (
            "from pricing.exercise_rights import AmericanExercise",
            "exercise = AmericanExercise(maturity=1.0)",
        ),
        "Pricing engine": (
            "from pricing.engines.monte_carlo import MonteCarloEngine",
            "engine = MonteCarloEngine(n_paths=10000, n_steps=100, seed=42)",
        ),
    }
    selected_components = {
        "Payoff": (
            f"from pricing.payoff import {payoff_class}",
            f"payoff = {payoff_expression}",
        ),
        "Exercise rights": (
            f"from pricing.exercise_rights import {exercise_class}",
            f"exercise = {exercise_expression}",
        ),
        "Pricing engine": (
            f"from pricing.engines.{engine_modules[engine_class]} import {engine_class}",
            f"engine = {engine_expression}",
        ),
    }
    original_result_line = (
        'print(f"Price: {result.price:.4f} (±{result.std_error:.4f})")'
    )
    original_engine_comment = (
        "# Price using Monte Carlo with Longstaff-Schwartz"
    )
    selected_engine_comment = {
        "Monte Carlo · Longstaff–Schwartz": original_engine_comment,
        "Binomial · CRR": "# Price using the CRR binomial tree",
        "Finite differences · Crank–Nicolson": (
            "# Price using Crank-Nicolson finite differences"
        ),
    }[engine_choice]
    selected_result_line = (
        original_result_line
        if engine_class == "MonteCarloEngine"
        else 'print(f"Price: {result.price:.4f}") # deterministic engine'
    )
    replacements = {
        original_line: selected_line
        for original_pair, selected_pair in zip(
            original_components.values(), selected_components.values()
        )
        for original_line, selected_line in zip(original_pair, selected_pair)
    }
    replacements[original_engine_comment] = selected_engine_comment
    replacements[original_result_line] = selected_result_line
    selected_quick_start = "\n".join(
        replacements.get(line, line) for line in original_quick_start.splitlines()
    )
    component_highlights = {
        "Payoff": (
            "rgba(92, 190, 174, 0.25)"
            if st.session_state.theme_mode == "Dark"
            else "rgba(23, 107, 104, 0.17)"
        ),
        "Exercise rights": (
            "rgba(226, 184, 105, 0.25)"
            if st.session_state.theme_mode == "Dark"
            else "rgba(138, 103, 39, 0.18)"
        ),
        "Pricing engine": (
            "rgba(128, 160, 220, 0.25)"
            if st.session_state.theme_mode == "Dark"
            else "rgba(56, 92, 145, 0.17)"
        ),
    }
    component_markers = {
        "Payoff": "#176b68" if st.session_state.theme_mode != "Dark" else "#5cbeae",
        "Exercise rights": "#8a6727" if st.session_state.theme_mode != "Dark" else "#e2b869",
        "Pricing engine": "#385c91" if st.session_state.theme_mode != "Dark" else "#80a0dc",
    }
    component_styles = f"""
    <style>
      .st-key-engine_design_payoff_choice label {{
        color: {component_markers["Payoff"]} !important;
      }}
      .st-key-engine_design_payoff_choice [role="group"] {{
        border-color: {component_markers["Payoff"]} !important;
        box-shadow: inset 3px 0 0 {component_markers["Payoff"]} !important;
        background-color: {component_highlights["Payoff"]} !important;
      }}
      .st-key-engine_design_exercise_choice label {{
        color: {component_markers["Exercise rights"]} !important;
      }}
      .st-key-engine_design_exercise_choice [role="group"] {{
        border-color: {component_markers["Exercise rights"]} !important;
        box-shadow: inset 3px 0 0 {component_markers["Exercise rights"]} !important;
        background-color: {component_highlights["Exercise rights"]} !important;
      }}
      .st-key-engine_design_engine_choice label {{
        color: {component_markers["Pricing engine"]} !important;
      }}
      .st-key-engine_design_engine_choice [role="group"] {{
        border-color: {component_markers["Pricing engine"]} !important;
        box-shadow: inset 3px 0 0 {component_markers["Pricing engine"]} !important;
        background-color: {component_highlights["Pricing engine"]} !important;
      }}
    </style>
    """
    line_components = {
        original_pair[1]: component
        for component, original_pair in original_components.items()
    }
    line_components[original_engine_comment] = "Pricing engine"
    line_components[original_result_line] = "Pricing engine"
    code_color = "#e6e7e9" if st.session_state.theme_mode == "Dark" else "#263746"
    line_number_color = (
        "#747d89" if st.session_state.theme_mode == "Dark" else "#9aa4ad"
    )
    python_token_colors = {
        "keyword": "#91cfc2" if st.session_state.theme_mode == "Dark" else "#176b68",
        "string": "#e6bd78" if st.session_state.theme_mode == "Dark" else "#9a6413",
        "number": "#e6bd78" if st.session_state.theme_mode == "Dark" else "#9a6413",
        "comment": "#929aa6" if st.session_state.theme_mode == "Dark" else "#71808c",
        "builtin": "#9fc5ff" if st.session_state.theme_mode == "Dark" else "#385c91",
        "operator": "#c6a0f6" if st.session_state.theme_mode == "Dark" else "#7952a0",
    }
    fstring_token_types = {
        token_type
        for token_type in (
            getattr(tokenize, "FSTRING_START", None),
            getattr(tokenize, "FSTRING_MIDDLE", None),
            getattr(tokenize, "FSTRING_END", None),
        )
        if token_type is not None
    }
    code_lines = selected_quick_start.splitlines()
    line_token_spans: list[list[tuple[int, int, str | None]]] = [
        [] for _ in code_lines
    ]
    for code_token in tokenize.generate_tokens(
        io.StringIO(selected_quick_start).readline
    ):
        token_color = None
        if code_token.type == tokenize.COMMENT:
            token_color = python_token_colors["comment"]
        elif (
            code_token.type == tokenize.STRING
            or code_token.type in fstring_token_types
        ):
            token_color = python_token_colors["string"]
        elif code_token.type == tokenize.NUMBER:
            token_color = python_token_colors["number"]
        elif code_token.type == tokenize.OP:
            token_color = python_token_colors["operator"]
        elif code_token.type == tokenize.NAME:
            if keyword.iskeyword(code_token.string):
                token_color = python_token_colors["keyword"]
            elif code_token.string in vars(builtins):
                token_color = python_token_colors["builtin"]
        start_row, start_column = code_token.start
        end_row, end_column = code_token.end
        for row in range(start_row - 1, end_row):
            span_start = start_column if row == start_row - 1 else 0
            span_end = (
                end_column
                if row == end_row - 1
                else len(code_lines[row])
            )
            if span_end > span_start:
                line_token_spans[row].append(
                    (span_start, span_end, token_color)
                )

    rendered_lines = []
    for line_number, (original_line, selected_line) in enumerate(
        zip(original_quick_start.splitlines(), selected_quick_start.splitlines()),
        start=1,
    ):
        changed_component = (
            line_components.get(original_line)
            if original_line != selected_line
            else None
        )
        background = (
            component_highlights[changed_component]
            if changed_component is not None
            else "transparent"
        )
        line_marker = (
            f"border-left:2px solid {component_markers[changed_component]};"
            if changed_component is not None
            else ""
        )
        highlighted_line = []
        cursor = 0
        for span_start, span_end, token_color in sorted(
            line_token_spans[line_number - 1],
            key=lambda span: (span[0], span[1]),
        ):
            span_start = max(cursor, span_start)
            if span_end <= span_start:
                continue
            if span_start > cursor:
                highlighted_line.append(
                    html.escape(selected_line[cursor:span_start])
                )
            token_text = html.escape(selected_line[span_start:span_end])
            if token_color:
                token_text = (
                    f'<span style="color:{token_color}">{token_text}</span>'
                )
            highlighted_line.append(token_text)
            cursor = span_end
        highlighted_line.append(html.escape(selected_line[cursor:]))
        rendered_lines.append(
            f'<div style="display:grid;grid-template-columns:2.6em minmax(0,1fr);'
            f'background:{background};{line_marker}">'
            f'<span style="color:{line_number_color};text-align:right;'
            f'padding-right:1em;user-select:none;">{line_number}</span>'
            f'<span style="white-space:pre-wrap;overflow-wrap:anywhere;">'
            f'{"".join(highlighted_line) or "&nbsp;"}</span></div>'
        )

    def render_engine_code_panel() -> None:
        st.markdown(
            component_styles
            + f'<div style="overflow-x:auto;color:{code_color};'
            f'padding:0.35rem 0;font-family:ui-monospace,SFMono-Regular,'
            f'Menlo,Consolas,monospace;font-size:0.78rem;line-height:1.75;">'
            + "".join(rendered_lines)
            + "</div>",
            unsafe_allow_html=True,
        )

    def render_engine_controls(horizontal: bool = False) -> None:
        st.markdown(
            f'<p style="margin:0 0 0.35rem;color:{pipeline_colors["accent"]};'
            f'font-size:0.7rem;font-weight:700;letter-spacing:0.1em;'
            f'text-transform:uppercase;">'
            "Explore the separation</p>"
            f'<p style="margin:0 0 1rem;color:{pipeline_colors["muted"]};'
            f'font-size:0.9rem;line-height:1.55;">'
            "Change an input to see its component update in the code, while "
            "the other parts stay independent.</p>",
            unsafe_allow_html=True,
        )
        if horizontal:
            control_columns = st.columns(4, gap="small")
            with control_columns[0]:
                st.selectbox(
                    "Payoff",
                    list(payoff_options),
                    key=payoff_state_key,
                )
            with control_columns[1]:
                st.selectbox(
                    "Exercise rights",
                    list(exercise_options),
                    key=exercise_state_key,
                )
            with control_columns[2]:
                st.selectbox(
                    "Pricing engine",
                    available_engines,
                    key=engine_state_key,
                )
            with control_columns[3]:
                st.selectbox(
                    "Process",
                    ("Geometric Brownian motion (GBM)",),
                    key=process_state_key,
                    disabled=True,
                    help="GBM is the only implemented stochastic process thus far.",
                )
        else:
            st.selectbox(
                "Payoff",
                list(payoff_options),
                key=payoff_state_key,
            )
            st.selectbox(
                "Exercise rights",
                list(exercise_options),
                key=exercise_state_key,
            )
            st.selectbox(
                "Pricing engine",
                available_engines,
                key=engine_state_key,
            )
            st.selectbox(
                "Process",
                ("Geometric Brownian motion (GBM)",),
                key=process_state_key,
                disabled=True,
                help="GBM is the only implemented stochastic process thus far.",
            )

    render_engine_section_heading("Quick code implementation")
    st.markdown(
        "The code below verifies the architecture above: **Payoff** and "
        "**Exercise Rights** are defined independently, assembled into a "
        "**Contract**, then passed with the **Process** and **Market** inputs "
        "to the **Pricing Engine**. Change a supported input to see the "
        "corresponding code component update without rewriting the others."
    )
    st.markdown('<div style="height:0.75rem"></div>', unsafe_allow_html=True)
    code_column, controls_column = st.columns(
        [2, 1], gap="medium", vertical_alignment="top"
    )
    with code_column:
        render_engine_code_panel()
    with controls_column:
        render_engine_controls()

    st.markdown(section_break_html, unsafe_allow_html=True)
    render_engine_section_heading("Engine Compatibility")
    st.markdown("".join(matrix_html), unsafe_allow_html=True)

    st.markdown(section_break_html, unsafe_allow_html=True)
    render_engine_section_heading("Explore the source")
    st.markdown(
        "Browse the modules behind the architecture, select a class or "
        "function to inspect its implementation, or choose the full file."
    )
    package_root = Path(__file__).resolve().parent.parent
    source_modules = (
        "ui/application.py",
        "ui/home.py",
        "ui/navigation.py",
        "ui/theme.py",
        "ui/pricing_lab.py",
        "ui/engine_design.py",
        "ui/applied_theory.py",
        "pricing/contract.py",
        "pricing/payoff.py",
        "pricing/exercise_rights.py",
        "pricing/markets.py",
        "pricing/stochastic_process.py",
        "pricing/state.py",
        "pricing/path_tracker.py",
        "pricing/engines/base.py",
        "pricing/engines/analytical.py",
        "pricing/engines/monte_carlo.py",
        "pricing/engines/binomial.py",
        "pricing/engines/finte_differences.py",
    )
    source_file_column, source_component_column = st.columns(
        [1, 1], gap="medium", vertical_alignment="bottom"
    )
    with source_file_column:
        source_file = st.selectbox(
            "Source file",
            source_modules,
            key="engine_design_source_file_v2",
        )
    source_text = (package_root / source_file).read_text(encoding="utf-8")
    source_tree = ast.parse(source_text, filename=source_file)
    source_components = [
        node
        for node in source_tree.body
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        )
    ]
    if not source_components:
        raise RuntimeError(f"No browsable code components found in {source_file}.")

    full_source_option = "Full file"
    component_nodes = {node.name: node for node in source_components}
    component_options = [full_source_option, *component_nodes]
    with source_component_column:
        component_name = st.selectbox(
            "Class or function",
            component_options,
            format_func=lambda name: (
                f"{name}()"
                if name in component_nodes
                and isinstance(
                    component_nodes[name],
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                )
                else name
            ),
            key=f"engine_design_source_component_{source_file.replace('/', '_')}",
        )
    if component_name == full_source_option:
        component_source = source_text
    else:
        selected_component = component_nodes[component_name]
        component_source = ast.get_source_segment(
            source_text, selected_component
        )
        if component_source is None:
            raise RuntimeError(
                f"Unable to read {component_name} from {source_file}."
            )
    with st.container(key="engine-design-source-code"):
        st.code(component_source, language="python", line_numbers=True)

    st.stop()
