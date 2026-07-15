"""
utils.py — Shared constants, theme helpers, and UI utility functions.
All colours use CSS custom properties so they adapt to light AND dark themes.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Colour palette (used in Plotly traces — these are fixed accent colours
#    that look good on both light and dark chart backgrounds) ─────────────────
PALETTE = {
    "primary":    "#6366F1",   # indigo-500  — vivid on both themes
    "secondary":  "#22D3EE",   # cyan-400    — vivid on both themes
    "success":    "#34D399",   # emerald-400 — vivid on both themes
    "warning":    "#FBBF24",   # amber-400   — vivid on both themes
    "danger":     "#F87171",   # red-400     — vivid on both themes
    "neutral":    "#94A3B8",   # slate-400   — mid-tone, visible both themes
    "text_muted": "#94A3B8",
}

SEQUENTIAL_BLUE   = px.colors.sequential.Blues
SEQUENTIAL_TEAL   = px.colors.sequential.Teal
DIVERGING_RDG     = px.colors.diverging.RdYlGn
QUALITATIVE_SAFE  = px.colors.qualitative.Safe

# Plotly layout — transparent backgrounds so the chart inherits the page theme
PLOTLY_LAYOUT = dict(
    font_family="Inter, sans-serif",
    font_color="#E2E8F0",          # light default; overridden per-chart where needed
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=54, b=40),
    hoverlabel=dict(
        bgcolor="#1E293B",         # slate-800 — readable on both themes
        font_color="#F1F5F9",
        font_size=13,
        bordercolor="#334155",
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def apply_layout(fig: go.Figure, title: str = "", **kwargs) -> go.Figure:
    """Apply the standard adaptive layout to any Plotly figure."""
    layout = {**PLOTLY_LAYOUT, **kwargs}
    if title:
        layout["title"] = dict(
            text=title,
            font=dict(size=15, color="#C7D2FE"),   # indigo-200 — visible on both themes
            x=0.02,
        )
    fig.update_layout(**layout)
    # Make axis lines/ticks theme-neutral
    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.15)",
        linecolor="rgba(148,163,184,0.3)",
        tickfont=dict(color="#94A3B8"),
        title_font=dict(color="#94A3B8"),
        zerolinecolor="rgba(148,163,184,0.2)",
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.15)",
        linecolor="rgba(148,163,184,0.3)",
        tickfont=dict(color="#94A3B8"),
        title_font=dict(color="#94A3B8"),
        zerolinecolor="rgba(148,163,184,0.2)",
    )
    return fig


# ── CSS custom-property block injected once ───────────────────────────────────
# All custom UI components read from these variables, so they flip automatically
# when Streamlit switches between light and dark themes.
_CSS_VARS = """
<style>
/* ── Design tokens ── */
:root {
  --c-accent:       #6366F1;
  --c-accent-dim:   rgba(99,102,241,0.15);
  --c-surface:      rgba(255,255,255,0.06);
  --c-border:       rgba(148,163,184,0.2);
  --c-text:         inherit;
  --c-text-muted:   rgba(148,163,184,0.9);
  --c-success:      #34D399;
  --c-success-dim:  rgba(52,211,153,0.12);
  --c-warning:      #FBBF24;
  --c-warning-dim:  rgba(251,191,36,0.12);
  --c-danger:       #F87171;
  --c-danger-dim:   rgba(248,113,113,0.12);
  --c-cyan:         #22D3EE;
  --c-cyan-dim:     rgba(34,211,238,0.12);
  --radius-card:    10px;
  --radius-sm:      6px;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background  : var(--c-surface) !important;
    border      : 1px solid var(--c-border) !important;
    border-radius: var(--radius-card) !important;
    padding     : 14px 18px !important;
    box-shadow  : 0 1px 6px rgba(0,0,0,.12) !important;
    backdrop-filter: blur(4px);
}
[data-testid="metric-container"] label {
    font-size   : 11px !important;
    font-weight : 700 !important;
    color       : var(--c-text-muted) !important;
    text-transform: uppercase;
    letter-spacing: .06em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size   : 24px !important;
    font-weight : 700 !important;
    color       : var(--c-accent) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E1B4B 100%) !important;
}
section[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 { color: #E2E8F0 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stSlider label {
    color: #94A3B8 !important;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .05em;
}

/* ── Tabs ── */
button[data-baseweb="tab"] { font-weight: 600; font-size: 13px; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--c-accent) !important;
    border-bottom: 2px solid var(--c-accent) !important;
}

/* ── Dividers ── */
hr { border-color: var(--c-border) !important; margin: 10px 0; }

/* ── Layout ── */
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; }

/* ── DataFrame ── */
[data-testid="stDataFrame"] { border-radius: var(--radius-card); overflow: hidden; }
</style>
"""


def inject_css() -> None:
    """Inject global CSS variables and component styles. Call once at app startup."""
    st.markdown(_CSS_VARS, unsafe_allow_html=True)


# ── Streamlit helpers ─────────────────────────────────────────────────────────

def metric_card(label: str, value: str, delta: str | None = None,
                help_text: str | None = None, icon: str = "") -> None:
    display_label = f"{icon} {label}" if icon else label
    st.metric(label=display_label, value=value, delta=delta, help=help_text)


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div style="margin-bottom:4px;">
          <span style="font-size:20px;font-weight:700;color:var(--c-accent);">{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(
            f'<p style="font-size:13px;color:var(--c-text-muted);margin-top:2px;'
            f'margin-bottom:10px;">{subtitle}</p>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<hr style="border:none;border-top:1px solid var(--c-border);margin:6px 0 18px 0;">',
        unsafe_allow_html=True,
    )


# Colour tokens for insight boxes — (border, background)
_INSIGHT_TOKENS: dict[str, tuple[str, str]] = {
    "#6366F1": ("var(--c-accent)",   "var(--c-accent-dim)"),
    "#22D3EE": ("var(--c-cyan)",     "var(--c-cyan-dim)"),
    "#34D399": ("var(--c-success)",  "var(--c-success-dim)"),
    "#FBBF24": ("var(--c-warning)",  "var(--c-warning-dim)"),
    "#F87171": ("var(--c-danger)",   "var(--c-danger-dim)"),
    # legacy hex keys that may be passed from app.py
    "#4F46E5": ("var(--c-accent)",   "var(--c-accent-dim)"),
    "#06B6D4": ("var(--c-cyan)",     "var(--c-cyan-dim)"),
    "#10B981": ("var(--c-success)",  "var(--c-success-dim)"),
    "#F59E0B": ("var(--c-warning)",  "var(--c-warning-dim)"),
    "#EF4444": ("var(--c-danger)",   "var(--c-danger-dim)"),
}


def insight_box(text: str, colour: str = "#6366F1") -> None:
    border_var, bg_var = _INSIGHT_TOKENS.get(colour, ("var(--c-accent)", "var(--c-accent-dim)"))
    st.markdown(
        f"""
        <div style="border-left:3px solid {border_var};
                    padding:11px 16px;
                    background:{bg_var};
                    border-radius:0 var(--radius-sm) var(--radius-sm) 0;
                    margin:6px 0;
                    line-height:1.6;">
            <span style="font-size:14px;">{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def recommendation_card(target: str, campaign: str, timing: str, impact: str) -> None:
    st.markdown(
        f"""
        <div style="border:1px solid var(--c-border);
                    border-radius:var(--radius-card);
                    padding:22px 24px;
                    background:var(--c-accent-dim);
                    margin-bottom:16px;
                    backdrop-filter:blur(4px);">

          <p style="margin:0 0 4px 0;font-size:11px;color:var(--c-text-muted);
                    font-weight:700;text-transform:uppercase;letter-spacing:.07em;">
            🎯 Target Segment</p>
          <p style="margin:0 0 16px 0;font-size:15px;font-weight:500;">{target}</p>

          <p style="margin:0 0 4px 0;font-size:11px;color:var(--c-text-muted);
                    font-weight:700;text-transform:uppercase;letter-spacing:.07em;">
            📢 Suggested Campaign</p>
          <p style="margin:0 0 16px 0;font-size:14px;">{campaign}</p>

          <p style="margin:0 0 4px 0;font-size:11px;color:var(--c-text-muted);
                    font-weight:700;text-transform:uppercase;letter-spacing:.07em;">
            ⏱ Recommended Timing</p>
          <p style="margin:0 0 16px 0;font-size:14px;">{timing}</p>

          <p style="margin:0 0 4px 0;font-size:11px;color:var(--c-text-muted);
                    font-weight:700;text-transform:uppercase;letter-spacing:.07em;">
            📈 Expected Impact</p>
          <p style="margin:0;font-size:14px;">{impact}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt_pct(v: float, decimals: int = 1) -> str:
    return f"{v:.{decimals}f}%"


def fmt_days(v: float) -> str:
    return f"{v:.0f} days"
