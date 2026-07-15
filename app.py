"""
app.py — E-Commerce Retention Analytics Dashboard
Run: streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    build_basket_discount_analysis,
    build_basket_segment_detail,
    build_campaign_duration,
    build_cohort_matrix,
    build_discount_type_summary,
    build_first_orders,
    build_recommendations,
    build_rpr_table,
    build_segment_summary,
    build_t2o,
    generate_insights,
    load_and_clean,
    _campaign_insights,
    _basket_discount_insights,
)
from utils import (
    PALETTE,
    QUALITATIVE_SAFE,
    SEQUENTIAL_BLUE,
    apply_layout,
    fmt_days,
    fmt_pct,
    inject_css,
    insight_box,
    metric_card,
    recommendation_card,
    section_header,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retention Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Global CSS (theme-adaptive via CSS custom properties) ─────────────────────
inject_css()


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_base_data():
    """
    Load, clean and build first-orders table.
    Returns only the lightweight metadata needed for the sidebar,
    plus the two core DataFrames for downstream computation.
    The heavy df is NOT returned — callers should use get_all_filtered_data().
    """
    df           = load_and_clean()
    first_orders = build_first_orders(df)
    # Sidebar metadata — tiny objects
    date_min      = df["created_at"].min()
    date_max      = df["created_at"].max()
    total_orders  = len(df)
    total_cust    = first_orders["customer_id"].nunique()
    disc_titles   = sorted(df["first_discount_title"].dropna().unique().tolist())
    return date_min, date_max, total_orders, total_cust, disc_titles


@st.cache_data(show_spinner=False)
def get_all_filtered_data():
    """
    Compute ALL derived metric tables from the full dataset.
    Returns only small summary DataFrames — NOT the raw orders df.
    """
    df           = load_and_clean()
    first_orders = build_first_orders(df)

    # Build all derived tables
    t2o_f                         = build_t2o(df, first_orders)
    rpr_f                         = build_rpr_table(df, first_orders)
    cohort_counts_f, cohort_pct_f = build_cohort_matrix(df, first_orders)
    disc_sum_f                    = build_segment_summary(
        t2o_f, first_orders, df, "first_order_discount",
        label_map={True: "Used Discount", False: "No Discount"},
    )
    basket_sum_f    = build_segment_summary(t2o_f, first_orders, df, "basket_segment")
    disc_type_f     = build_discount_type_summary(t2o_f, first_orders, df)
    bd_f            = build_basket_discount_analysis(first_orders, df, t2o_f)
    basket_detail_f = build_basket_segment_detail(first_orders, df, t2o_f)
    median_basket   = first_orders["first_basket_size"].median()

    # Only return small derived tables — drop the large raw frames
    return (
        t2o_f, rpr_f,
        cohort_counts_f, cohort_pct_f,
        disc_sum_f, basket_sum_f, disc_type_f, bd_f, basket_detail_f,
        median_basket,
    )


try:
    loading_placeholder = st.empty()
    loading_placeholder.markdown(
        """
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;height:60vh;gap:16px;">
            <div style="font-size:48px;">📊</div>
            <div style="font-size:20px;font-weight:600;">
                Loading data…
            </div>
            <div style="font-size:14px;color:var(--c-text-muted,#94A3B8);">
                Cleaning orders and computing retention metrics
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    date_min, date_max, total_orders, total_cust, disc_titles = get_base_data()
    loading_placeholder.empty()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='color:#C7D2FE;font-size:17px;font-weight:700;margin-bottom:2px;'>"
        "📊 Retention Analytics</h2>"
        "<p style='color:#64748B;font-size:12px;margin-top:0;letter-spacing:.03em;'>"
        "E-Commerce Growth Dashboard</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if data_loaded:
        n_months = (date_max - date_min).days / 30.44

        st.markdown("#### 📁 Dataset Overview")
        st.markdown(
            f"""
            <div style='font-size:13px;line-height:2;color:var(--c-text-muted);'>
            📅 <b style='color:#C7D2FE;'>Date range:</b> {date_min.date()} → {date_max.date()}<br>
            👥 <b style='color:#C7D2FE;'>Customers:</b> {total_cust:,}<br>
            🛒 <b style='color:#C7D2FE;'>Total orders:</b> {total_orders:,}<br>
            📆 <b style='color:#C7D2FE;'>Span:</b> ~{n_months:.0f} months
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(
            "<p style='font-size:11px;color:#475569;text-align:center;margin-top:8px;'>"
            "Streamlit · Plotly · Pandas</p>",
            unsafe_allow_html=True,
        )


# ── Error state ───────────────────────────────────────────────────────────────
if not data_loaded:
    st.error("### ⚠️ Data file not found")
    st.markdown(
        """
        Please place **`ecommerce_orders.csv`** in the same directory as `app.py` and refresh.

        The CSV should be a Shopify-style order export with at minimum:
        - `created_at` — ISO-8601 order timestamp
        - `customer.id` — numeric customer identifier
        - `line_items[*].quantity` — per-item quantities
        - `discount_applications[*].title` — discount titles (optional)
        """
    )
    st.stop()


# ── Resolve all derived data from cache (single call) ─────────────────────────
(t2o_f, rpr_f_base,
 cohort_counts_f_base, cohort_pct_f_base,
 disc_sum_f_base, basket_sum_f_base, disc_type_f_base, bd_f_base,
 basket_detail_f_base, median_basket_f) = get_all_filtered_data()

# Convenience aliases used throughout tabs
df_f           = None   # not held in memory — all metrics pre-computed above
first_orders_f = None   # same

cohort_months = sorted(cohort_pct_f_base.index.astype(str).tolist())
with st.sidebar:
    if data_loaded:
        if len(cohort_months) >= 2:
            cohort_range = st.select_slider(
                "Cohort Month Range",
                options=cohort_months,
                value=(cohort_months[0], cohort_months[-1]),
            )
        else:
            cohort_range = (cohort_months[0], cohort_months[-1]) if cohort_months else (None, None)

# ── Tabs ──────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-size:26px;font-weight:700;color:var(--c-accent);margin-bottom:2px;'>"
    "📊 Retention Analytics Dashboard</h1>"
    "<p style='color:var(--c-text-muted);font-size:13px;margin-top:0;'>Customer retention, "
    "repeat purchase &amp; lifecycle analysis</p>",
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "🏠 Executive Summary",
    "🔵 Cohort Retention",
    "🔁 Repeat Purchase",
    "⏱ Time to 2nd Order",
    "🏷 Discount Analysis",
    "📋 Discount Types",
    "🛒 Basket Analysis",
    # "💡 Insights",
    # "🎯 Recommendations",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — Executive Summary
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    section_header("Executive Summary", "Top-line KPIs at a glance")

    rpr_f    = rpr_f_base
    rpr_dict = rpr_f.set_index("Window (days)")["RPR (%)"].to_dict()

    med_t2o = t2o_f["days_to_second_order"].median() if len(t2o_f) else 0
    p25_t2o = t2o_f["days_to_second_order"].quantile(0.25) if len(t2o_f) else 0
    p75_t2o = t2o_f["days_to_second_order"].quantile(0.75) if len(t2o_f) else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        metric_card("Total Orders", f"{total_orders:,}", icon="🛒")
    with c2:
        metric_card("Unique Customers", f"{total_cust:,}", icon="👥")
    with c3:
        metric_card("30-day RPR", fmt_pct(rpr_dict.get(30, 0)), icon="🔁")
    with c4:
        metric_card("60-day RPR", fmt_pct(rpr_dict.get(60, 0)), icon="🔁")
    with c5:
        metric_card("90-day RPR", fmt_pct(rpr_dict.get(90, 0)), icon="🔁")
    with c6:
        metric_card("Median → 2nd Order", fmt_days(med_t2o), icon="⏱")

    st.markdown("<br>", unsafe_allow_html=True)

    # Mini RPR trend chart
    col_left, col_right = st.columns([1, 1])
    with col_left:
        fig_rpr = go.Figure()
        colors_rpr = [PALETTE["primary"], PALETTE["secondary"], PALETTE["success"]]
        for i, row in rpr_f.iterrows():
            fig_rpr.add_bar(
                x=[f"{int(row['Window (days)'])}d"],
                y=[row["RPR (%)"]],
                name=f"{int(row['Window (days)'])} days",
                marker_color=colors_rpr[i],
                text=[f"{row['RPR (%)']:.1f}%"],
                textposition="outside",
            )
        apply_layout(fig_rpr, "Repeat Purchase Rate by Window")
        fig_rpr.update_layout(showlegend=False, yaxis_title="RPR (%)", height=320)
        st.plotly_chart(fig_rpr, use_container_width=True, key="exec_rpr_bar")

    with col_right:
        # Funnel: new → repeat30 → repeat60 → repeat90
        total_c = total_cust
        funnel_vals = [
            total_c,
            int(rpr_dict.get(30, 0) / 100 * total_c),
            int(rpr_dict.get(60, 0) / 100 * total_c),
            int(rpr_dict.get(90, 0) / 100 * total_c),
        ]
        fig_funnel = go.Figure(go.Funnel(
            y=["All Customers", "Repeat in 30d", "Repeat in 60d", "Repeat in 90d"],
            x=funnel_vals,
            textinfo="value+percent initial",
            marker_color=[PALETTE["primary"], PALETTE["secondary"],
                          PALETTE["success"], PALETTE["warning"]],
        ))
        apply_layout(fig_funnel, "Customer Repeat Purchase Funnel")
        fig_funnel.update_layout(height=320)
        st.plotly_chart(fig_funnel, use_container_width=True, key="exec_funnel")

    # Summary paragraph
    st.markdown("---")
    n_months_f = (date_max - date_min).days / 30.44
    bd_exec = bd_f_base
    st.info(
        f"**Dataset:** {total_orders:,} orders from {total_cust:,} "
        f"customers over ~{n_months_f:.0f} months "
        f"({date_min.date()} → {date_max.date()}).  "
        f"**Median time to second order:** {med_t2o:.0f} days "
        f"(P25={p25_t2o:.0f}d, P75={p75_t2o:.0f}d).  "
        f"**Share with discounted first order:** "
        f"{bd_exec['lb_disc_pct'] * bd_exec['lb_total'] / (bd_exec['lb_total'] + bd_exec['sb_total']):.1f}%.  "
        f"**Large Basket discount rate:** {bd_exec['lb_disc_pct']:.1f}% vs "
        f"{bd_exec['sb_disc_pct']:.1f}% for Small Basket — "
        f"{'large-basket customers are more discount-dependent' if bd_exec['lb_disc_pct'] > bd_exec['sb_disc_pct'] else 'discount usage is similar across basket sizes'}."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Cohort Retention
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    section_header("Cohort Retention Analysis", "How many customers return each month after their first purchase?")

    cohort_counts_f = cohort_counts_f_base.copy()
    cohort_pct_f    = cohort_pct_f_base.copy()

    # Filter cohort range
    if cohort_range[0] and cohort_range[1]:
        try:
            cohort_pct_f   = cohort_pct_f.loc[cohort_range[0]:cohort_range[1]]
            cohort_counts_f = cohort_counts_f.loc[cohort_range[0]:cohort_range[1]]
        except Exception:
            pass

    view_mode = st.radio("View as", ["Percentage (%)", "Customer Counts"], horizontal=True)
    matrix    = cohort_pct_f if "Percentage" in view_mode else cohort_counts_f

    # Plotly heatmap
    z_vals   = matrix.values.astype(float)
    y_labels = [str(r) for r in matrix.index]
    x_labels = [f"Month {c}" for c in matrix.columns]

    fmt_text = [[
        f"{v:.0f}%" if "Percentage" in view_mode else f"{int(v):,}" if not np.isnan(v) else ""
        for v in row
    ] for row in z_vals]

    fig_heatmap = go.Figure(go.Heatmap(
        z=z_vals,
        x=x_labels,
        y=y_labels,
        text=fmt_text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
        colorbar=dict(title="%" if "Percentage" in view_mode else "Customers"),
        hoverongaps=False,
        zmin=0,
    ))
    apply_layout(
        fig_heatmap,
        "Monthly Cohort Retention Heatmap",
        xaxis_title="Months Since First Purchase",
        yaxis_title="Cohort (First Purchase Month)",
    )
    fig_heatmap.update_layout(height=max(400, len(y_labels) * 22 + 120))
    st.plotly_chart(fig_heatmap, use_container_width=True, key="cohort_heatmap")

    # Average retention curve — mature cohorts only
    mature = cohort_pct_f.dropna(subset=[6]) if 6 in cohort_pct_f.columns else cohort_pct_f.dropna()
    if len(mature) > 0:
        avg_curve = mature.mean()
        fig_curve = go.Figure()
        fig_curve.add_scatter(
            x=avg_curve.index.tolist(),
            y=avg_curve.values.tolist(),
            mode="lines+markers+text",
            line=dict(color=PALETTE["primary"], width=3),
            marker=dict(size=9),
            text=[f"{v:.1f}%" for v in avg_curve.values],
            textposition="top center",
            name="Avg Retention",
        )
        apply_layout(fig_curve, f"Average Retention Curve (Mature Cohorts: n={len(mature)})",
                     xaxis_title="Months Since First Purchase",
                     yaxis_title="Retention (%)", yaxis_ticksuffix="%")
        fig_curve.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig_curve, use_container_width=True, key="cohort_avg_curve")

    # Auto insights
    st.markdown("#### 💡 Key Observations")
    if 1 in cohort_pct_f.columns:
        avg_m1 = cohort_pct_f[1].dropna().mean()
        insight_box(
            f"<b>Month 0→1 churn:</b> on average <b>{avg_m1:.1f}%</b> of customers return "
            f"in the month after acquisition — this is the single highest-leverage retention point.",
            PALETTE["primary"],
        )
    if 6 in cohort_pct_f.columns and len(mature) > 0:
        avg_m6 = mature[6].mean()
        avg_m1_m = mature[1].mean() if 1 in mature.columns else 0
        insight_box(
            f"<b>Retention stabilises after Month 1:</b> Month-6 retention averages "
            f"<b>{avg_m6:.1f}%</b> vs Month-1 at {avg_m1_m:.1f}% — customers who survive the "
            f"first month form a durable loyal core.",
            PALETTE["success"],
        )   


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Repeat Purchase Rate
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    section_header("Repeat Purchase Rate (RPR)", "% of customers who placed ≥2 orders within N days of first purchase")

    rpr_f        = rpr_f_base
    rpr_dict_f   = rpr_f.set_index("Window (days)")["RPR (%)"].to_dict()

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("30-Day RPR", fmt_pct(rpr_dict_f.get(30, 0)), icon="🔁",
                    help_text="% of customers with a second order within 30 days")
    with c2:
        metric_card("60-Day RPR", fmt_pct(rpr_dict_f.get(60, 0)), icon="🔁",
                    help_text="% of customers with a second order within 60 days")
    with c3:
        metric_card("90-Day RPR", fmt_pct(rpr_dict_f.get(90, 0)), icon="🔁",
                    help_text="% of customers with a second order within 90 days")

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])
    with col_l:
        fig_bar = go.Figure()
        bar_colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["success"]]
        for i, row in rpr_f.iterrows():
            fig_bar.add_bar(
                x=[f"{int(row['Window (days)'])} days"],
                y=[row["RPR (%)"]],
                marker_color=bar_colors[i],
                text=[f"{row['RPR (%)']:.1f}%"],
                textposition="outside",
                name=f"{int(row['Window (days)'])}d",
            )
        apply_layout(fig_bar, "Repeat Purchase Rate by Window",
                     yaxis_title="RPR (%)", yaxis_ticksuffix="%")
        fig_bar.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_bar, use_container_width=True, key="rpr_bar")

    with col_r:
        # Waterfall: incremental repeaters between windows
        r30 = int(rpr_dict_f.get(30, 0) / 100 * total_cust)
        r60 = int(rpr_dict_f.get(60, 0) / 100 * total_cust)
        r90 = int(rpr_dict_f.get(90, 0) / 100 * total_cust)
        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Day 0–30", "Day 31–60", "Day 61–90", "Total 90d"],
            y=[r30, r60 - r30, r90 - r60, 0],
            connector={"line": {"color": "#9CA3AF"}},
            increasing={"marker": {"color": PALETTE["success"]}},
            totals={"marker": {"color": PALETTE["primary"]}},
            text=[f"{r30:,}", f"+{r60-r30:,}", f"+{r90-r60:,}", f"{r90:,}"],
            textposition="outside",
        ))
        apply_layout(fig_wf, "Incremental Repeat Buyers by Time Window",
                     yaxis_title="Customers")
        fig_wf.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_wf, use_container_width=True, key="rpr_waterfall")

    st.markdown("#### 💡 Business Implications")
    gap_30_60 = rpr_dict_f.get(60, 0) - rpr_dict_f.get(30, 0)
    gap_60_90 = rpr_dict_f.get(90, 0) - rpr_dict_f.get(60, 0)
    insight_box(
        f"<b>30→60 day gap is {gap_30_60:.1f} pp</b> and 60→90 is {gap_60_90:.1f} pp — "
        f"the largest conversion window is in the 31–60 day range, making it the optimal "
        f"moment for a first win-back nudge.",
        PALETTE["primary"],
    )
    insight_box(
        "Most loyal customers reveal themselves <b>after</b> 30 days — a 7–14 day "
        "welcome series alone misses the majority of eventual repeat buyers.",
        PALETTE["secondary"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Time to Second Order
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    section_header("Time to Second Order", "Days between first and second purchase for returning customers")

    if len(t2o_f) == 0:
        st.warning("No customers with 2+ orders found under the current filter selection.")
    else:
        med_f  = t2o_f["days_to_second_order"].median()
        p25_f  = t2o_f["days_to_second_order"].quantile(0.25)
        p75_f  = t2o_f["days_to_second_order"].quantile(0.75)
        repeat_share = len(t2o_f) / total_cust * 100

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Repeat Customers", f"{len(t2o_f):,}", icon="👥")
        with c2:
            metric_card("Repeat Rate", fmt_pct(repeat_share), icon="🔄")
        with c3:
            metric_card("Median Days", fmt_days(med_f), icon="📅",
                        help_text="50th percentile time to second purchase")
        with c4:
            metric_card("P25 / P75", f"{p25_f:.0f}d / {p75_f:.0f}d", icon="📊")

        st.markdown("<br>", unsafe_allow_html=True)
        clip_val = t2o_f["days_to_second_order"].quantile(0.99)
        t2o_clipped = t2o_f[t2o_f["days_to_second_order"] <= clip_val]["days_to_second_order"]

        fig_hist = px.histogram(
            t2o_clipped,
            nbins=60,
            labels={"value": "Days to 2nd Order"},
            color_discrete_sequence=[PALETTE["primary"]],
        )
        fig_hist.add_vline(x=med_f, line_dash="dash", line_color=PALETTE["danger"],
                           annotation_text=f"Median: {med_f:.0f}d",
                           annotation_position="top right")
        fig_hist.add_vline(x=p25_f, line_dash="dot", line_color=PALETTE["warning"],
                           annotation_text=f"P25: {p25_f:.0f}d",
                           annotation_position="top left")
        fig_hist.add_vline(x=p75_f, line_dash="dot", line_color=PALETTE["warning"],
                           annotation_text=f"P75: {p75_f:.0f}d",
                           annotation_position="top right")
        apply_layout(fig_hist, "Distribution: Days to 2nd Order (clipped at P99)",
                     xaxis_title="Days", yaxis_title="Frequency")
        fig_hist.update_layout(showlegend=False, height=460)
        st.plotly_chart(fig_hist, use_container_width=True, key="t2o_hist")

        st.markdown("#### 💡 Interpretation")
        insight_box(
            f"<b>Optimal trigger window:</b> the median repeat buyer returns after "
            f"<b>{med_f:.0f} days</b>. A campaign triggered at day "
            f"<b>{max(20, int(med_f * 0.85)):.0f}–{int(med_f * 0.95):.0f}</b> "
            f"catches most potential repeaters just before they would naturally convert.",
            PALETTE["primary"],
        )
        insight_box(
            f"<b>Long tail opportunity:</b> P75 is {p75_f:.0f} days — a second, longer-horizon "
            f"re-engagement touch (day 60–75) can recover high-value customers who take longer "
            f"to return.",
            PALETTE["secondary"],
        )
        insight_box(
            f"<b>Fast repeaters (< P25 = {p25_f:.0f} days):</b> this cluster buys consumables "
            f"or commodities. They are ideal for subscription or auto-replenishment up-sell if "
            f"one doesn't already exist.",
            PALETTE["success"],
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Discount Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    section_header("Discount vs No Discount", "Do customers acquired with a discount behave differently?")

    disc_sum_f = disc_sum_f_base

    if disc_sum_f.empty:
        st.warning("Not enough data under the current filter selection.")
    else:
        # KPI cards per segment
        for _, row in disc_sum_f.iterrows():
            st.markdown(f"**{row['Segment']}** — {row['N Customers']:,} customers")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("30-day RPR", fmt_pct(row["30-day RPR (%)"]), icon="🔁")
            with c2:
                metric_card("60-day RPR", fmt_pct(row["60-day RPR (%)"]), icon="🔁")
            with c3:
                metric_card("Median Days → 2nd", fmt_days(row["Median Days to 2nd Order"]), icon="⏱")
            with c4:
                metric_card("P25 / P75", f"{row['P25']:.0f}d / {row['P75']:.0f}d", icon="📊")
            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("---")

        fig_disc_t2o = go.Figure()
        colors_seg = [PALETTE["primary"], PALETTE["secondary"]]
        for i, row in disc_sum_f.iterrows():
            fig_disc_t2o.add_trace(go.Bar(
                name=row["Segment"],
                x=[row["Segment"]],
                y=[row["Median Days to 2nd Order"]],
                marker_color=colors_seg[i % 2],
                text=[f"{row['Median Days to 2nd Order']:.0f}d"],
                textposition="outside",
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[row["P75"] - row["Median Days to 2nd Order"]],
                    arrayminus=[row["Median Days to 2nd Order"] - row["P25"]],
                ),
            ))
        apply_layout(fig_disc_t2o, "Median Days to 2nd Order (with P25–P75 range)",
                     yaxis_title="Days")
        fig_disc_t2o.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig_disc_t2o, use_container_width=True, key="disc_t2o_bar")

        # Multi-metric grouped bar
        melted = disc_sum_f.melt(
            id_vars="Segment",
            value_vars=["30-day RPR (%)", "60-day RPR (%)", "90-day RPR (%)"],
            var_name="Window", value_name="RPR",
        )
        fig_grouped = px.bar(
            melted, x="Window", y="RPR", color="Segment", barmode="group",
            color_discrete_sequence=[PALETTE["primary"], PALETTE["secondary"]],
            text=melted["RPR"].apply(lambda v: fmt_pct(v)),
        )
        apply_layout(fig_grouped, "RPR Comparison across All Windows",
                     yaxis_title="RPR (%)", yaxis_ticksuffix="%")
        fig_grouped.update_traces(textposition="outside")
        fig_grouped.update_layout(height=380)
        st.plotly_chart(fig_grouped, use_container_width=True, key="disc_grouped_rpr")

        # Interpretation
        if len(disc_sum_f) == 2:
            disc_row   = disc_sum_f[disc_sum_f["Segment"] == "Used Discount"]
            nodisc_row = disc_sum_f[disc_sum_f["Segment"] == "No Discount"]
            if not disc_row.empty and not nodisc_row.empty:
                d_rpr  = disc_row.iloc[0]["30-day RPR (%)"]
                nd_rpr = nodisc_row.iloc[0]["30-day RPR (%)"]
                if d_rpr > nd_rpr:
                    insight_box(
                        f"<b>Discounts appear to build habit:</b> discount-acquired customers "
                        f"show a <b>higher</b> 30-day RPR ({d_rpr:.1f}% vs {nd_rpr:.1f}%), "
                        f"suggesting the promotion successfully converts first-time buyers "
                        f"into repeat customers.",
                        PALETTE["success"],
                    )
                else:
                    insight_box(
                        f"<b>Discount-acquired customers are more price-sensitive:</b> their "
                        f"30-day RPR ({d_rpr:.1f}%) is <b>lower</b> than full-price customers "
                        f"({nd_rpr:.1f}%), a classic deal-seeker pattern. Focus discount "
                        f"budgets on mechanic-based types that show stronger retention "
                        f"(see Discount Types tab).",
                        PALETTE["warning"],
                    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Discount Type Performance
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    section_header(
        "Discount Type Performance",
        "Year-wise cohort analysis — every row is one (Discount × Year) customer group",
    )

    disc_type_f = disc_type_f_base.copy()

    if disc_type_f.empty:
        st.warning("No discount cohorts with ≥10 customers found under the current filter selection.")
    else:
        # ── Year filter ──────────────────────────────────────────────────────
        available_years = ["All Years"] + sorted(disc_type_f["Year"].unique().tolist(), reverse=True)
        selected_year   = st.selectbox("📅 Filter by Year", available_years, index=0)
        if selected_year != "All Years":
            disc_type_f = disc_type_f[disc_type_f["Year"] == int(selected_year)]

        # ── Summary KPIs ─────────────────────────────────────────────────────
        total_campaigns  = len(disc_type_f)
        avg_duration     = disc_type_f["Duration (days)"].mean()
        best_row         = disc_type_f.loc[disc_type_f["30-day RPR (%)"].idxmax()]
        longest_row      = disc_type_f.loc[disc_type_f["Duration (days)"].idxmax()]

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            metric_card("Total Campaigns", f"{total_campaigns:,}", icon="📋",
                        help_text="Unique Discount × Year combinations")
        with k2:
            metric_card("Avg Campaign Duration", f"{avg_duration:.0f} days", icon="📆")
        with k3:
            metric_card(
                "Best 30d RPR",
                f"{best_row['30-day RPR (%)']:.1f}%",
                icon="🏆",
                help_text=f"{best_row['Discount Type']} ({int(best_row['Year'])})",
            )
        with k4:
            metric_card(
                "Longest Campaign",
                f"{longest_row['Duration (days)']:,}d",
                icon="⏳",
                help_text=f"{longest_row['Discount Type']} ({int(longest_row['Year'])})",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Search + Sort ─────────────────────────────────────────────────────
        col_search, col_sort = st.columns([2, 1])
        with col_search:
            search_term = st.text_input("🔍 Search discount type", placeholder="Type to filter…")
        with col_sort:
            sort_col = st.selectbox(
                "Sort by",
                ["30-day RPR (%)", "N Customers", "Median Days to 2nd Order",
                 "Avg Orders in 90d", "Duration (days)", "Year"],
                index=0,
            )

        if search_term:
            disc_type_f = disc_type_f[
                disc_type_f["Discount Type"].str.contains(search_term, case=False, na=False)
            ]

        disc_type_sorted = disc_type_f.sort_values(
            sort_col,
            ascending=(sort_col in ("Median Days to 2nd Order",)),
        ).reset_index(drop=True)

        # ── Formatted display table ───────────────────────────────────────────
        display_df = disc_type_sorted.copy()
        display_df["30-day RPR (%)"]            = display_df["30-day RPR (%)"].apply(lambda v: f"{v:.1f}%")
        display_df["Median Days to 2nd Order"]  = display_df["Median Days to 2nd Order"].apply(
            lambda v: f"{v:.0f}d" if pd.notna(v) else "—"
        )
        display_df["Avg Orders in 90d"]  = display_df["Avg Orders in 90d"].apply(lambda v: f"{v:.2f}")
        display_df["N Customers"]        = display_df["N Customers"].apply(lambda v: f"{v:,}")
        display_df["Duration (days)"]    = display_df["Duration (days)"].apply(lambda v: f"{v}d")
        display_df["Campaign Start"]     = display_df["Campaign Start"].apply(
            lambda v: v.strftime("%d %b %Y") if pd.notna(v) else "—"
        )
        display_df["Campaign End"]       = display_df["Campaign End"].apply(
            lambda v: v.strftime("%d %b %Y") if pd.notna(v) else "—"
        )
        # Reorder columns for readability
        display_df = display_df[[
            "Discount Type", "Year", "N Customers",
            "30-day RPR (%)", "Median Days to 2nd Order", "Avg Orders in 90d",
            "Campaign Start", "Campaign End", "Duration (days)",
        ]]
        st.dataframe(display_df, use_container_width=True, height=400)

        st.markdown("---")

        # ── Bar charts (use numeric disc_type_sorted, not display_df) ─────────
        top_n_count = min(15, len(disc_type_sorted))
        # Label = "Discount (Year)" for charts
        disc_type_sorted = disc_type_sorted.copy()
        disc_type_sorted["Label"] = (
            disc_type_sorted["Discount Type"] + " ("
            + disc_type_sorted["Year"].astype(str) + ")"
        )
        top_types = disc_type_sorted.nlargest(top_n_count, "30-day RPR (%)")
        bot_types = disc_type_sorted.nsmallest(top_n_count, "30-day RPR (%)").sort_values("30-day RPR (%)")

        col_l, col_r = st.columns(2)
        with col_l:
            fig_top = px.bar(
                top_types.sort_values("30-day RPR (%)"),
                y="Label", x="30-day RPR (%)",
                orientation="h",
                color="30-day RPR (%)",
                color_continuous_scale="Blues",
                text=top_types.sort_values("30-day RPR (%)")["30-day RPR (%)"].apply(fmt_pct),
            )
            apply_layout(fig_top, f"Top {top_n_count} by 30-Day RPR",
                         xaxis_title="30-day RPR (%)", xaxis_ticksuffix="%")
            fig_top.update_traces(textposition="outside")
            fig_top.update_layout(
                coloraxis_showscale=False, showlegend=False,
                height=max(350, top_n_count * 30 + 100),
            )
            st.plotly_chart(fig_top, use_container_width=True, key="disctype_top_bar")

        with col_r:
            fig_bot = px.bar(
                bot_types,
                y="Label", x="30-day RPR (%)",
                orientation="h",
                color="30-day RPR (%)",
                color_continuous_scale="Reds_r",
                text=bot_types["30-day RPR (%)"].apply(fmt_pct),
            )
            apply_layout(fig_bot, f"Bottom {top_n_count} by 30-Day RPR",
                         xaxis_title="30-day RPR (%)", xaxis_ticksuffix="%")
            fig_bot.update_traces(textposition="outside")
            fig_bot.update_layout(
                coloraxis_showscale=False, showlegend=False,
                height=max(350, top_n_count * 30 + 100),
            )
            st.plotly_chart(fig_bot, use_container_width=True, key="disctype_bot_bar")

        # ── Campaign duration overview charts ─────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📆 Campaign Duration Overview")

        def _dur_bucket_inline(d: int) -> str:
            if d >= 300: return "Always-on (≥300d)"
            if d >= 90:  return "Long campaign (90–299d)"
            if d >= 30:  return "Medium (30–89d)"
            return "Short / Flash (<30d)"

        colour_map_inline = {
            "Always-on (≥300d)":      PALETTE["success"],
            "Long campaign (90–299d)": PALETTE["primary"],
            "Medium (30–89d)":         PALETTE["warning"],
            "Short / Flash (<30d)":    PALETTE["danger"],
        }
        disc_type_sorted["Duration Type"] = disc_type_sorted["Duration (days)"].apply(
            lambda v: _dur_bucket_inline(int(str(v).replace("d", "")))
            if isinstance(v, str) else _dur_bucket_inline(int(v))
        )

        col_l, col_r = st.columns(2)
        with col_l:
            fig_dur_hist = px.histogram(
                disc_type_sorted, x="Duration (days)",
                nbins=30,
                color="Duration Type",
                color_discrete_map=colour_map_inline,
                labels={"Duration (days)": "Campaign Duration (days)"},
            )
            apply_layout(fig_dur_hist, "Distribution of Campaign Durations",
                         xaxis_title="Days", yaxis_title="Count")
            fig_dur_hist.update_layout(
                showlegend=True, height=340,
                legend=dict(
                    orientation="h", yanchor="top", y=-0.22,
                    xanchor="center", x=0.5, title=None, font=dict(size=11),
                ),
                margin=dict(b=80),
            )
            st.plotly_chart(fig_dur_hist, use_container_width=True, key="disctype_dur_hist")

        with col_r:
            cpy = (
                disc_type_sorted.groupby("Year")["Discount Type"]
                .nunique().reset_index(name="Campaigns")
            )
            fig_cpy = go.Figure()
            fig_cpy.add_bar(
                x=cpy["Year"].astype(str), y=cpy["Campaigns"],
                marker_color=PALETTE["primary"],
                text=cpy["Campaigns"], textposition="outside",
            )
            apply_layout(fig_cpy, "Unique Discount Campaigns per Year",
                         xaxis_title="Year", yaxis_title="# Campaigns",
                         xaxis_type="category")
            fig_cpy.update_layout(showlegend=False, height=340)
            st.plotly_chart(fig_cpy, use_container_width=True, key="disctype_cpy_bar")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Basket Size Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    section_header("Basket Size Analysis", "Does first-order quantity predict future retention?")

    basket_sum_f    = basket_sum_f_base
    basket_detail_f = basket_detail_f_base
    bd_basket       = bd_f_base

    if basket_sum_f.empty:
        st.warning("Not enough data under the current filter selection.")
    else:
        st.info(f"**Threshold:** Large Basket = first-order quantity > {median_basket_f:.0f} units (median)")

        # ── Per-segment KPI cards (Large then Small) with discount sub-rows ──
        _seg_totals = {
            "Large Basket": (bd_basket["lb_total"], bd_basket["lb_disc_pct"]),
            "Small Basket": (bd_basket["sb_total"], bd_basket["sb_disc_pct"]),
        }
        _seg_disc_help = {
            "Large Basket": "% of large-basket first orders that used a discount",
            "Small Basket": "% of small-basket first orders that used a discount",
        }
        for seg_label in ["Large Basket", "Small Basket"]:
            seg_row = basket_sum_f[basket_sum_f["Segment"] == seg_label]
            if seg_row.empty:
                continue
            row = seg_row.iloc[0]
            seg_total, seg_disc_pct = _seg_totals[seg_label]

            st.markdown(f"**{seg_label}**")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                metric_card(f"{seg_label.split()[0]} Basket (total)", f"{seg_total:,}",
                            icon="🛒" if seg_label == "Large Basket" else "🛍")
            with c2:
                metric_card(f"{seg_label.split()[0]} + Discount", fmt_pct(seg_disc_pct),
                            icon="🏷", help_text=_seg_disc_help[seg_label])
            with c3:
                metric_card("30-day RPR", fmt_pct(row["30-day RPR (%)"]), icon="🔁")
            with c4:
                metric_card("60-day RPR", fmt_pct(row["60-day RPR (%)"]), icon="🔁")
            with c5:
                metric_card("Median Days → 2nd", fmt_days(row["Median Days to 2nd Order"]), icon="⏱")
            with c6:
                metric_card("P25 / P75", f"{row['P25']:.0f}d / {row['P75']:.0f}d", icon="📊")

            # Sub-segment breakdown: With Discount vs No Discount
            sub = basket_detail_f[basket_detail_f["Basket Segment"] == seg_label]
            if not sub.empty:
                disc_colors = {
                    "With Discount": PALETTE["primary"],
                    "No Discount":   PALETTE["neutral"],
                }
                cols = st.columns(len(sub))
                for col, (_, sr) in zip(cols, sub.iterrows()):
                    disc_color = disc_colors.get(sr["Discount"], PALETTE["neutral"])
                    col.markdown(
                        f"""
                        <div style="border-left:3px solid {disc_color};
                                    padding:10px 14px;
                                    background:rgba(99,102,241,0.07);
                                    border-radius:0 6px 6px 0;
                                    margin-top:8px;">
                          <div style="font-size:11px;font-weight:700;
                                      color:{disc_color};text-transform:uppercase;
                                      letter-spacing:.05em;margin-bottom:6px;">
                            🏷 {sr['Discount']} · {sr['N Customers']:,} customers
                          </div>
                          <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 16px;
                                      font-size:13px;line-height:1.8;">
                            <span style="color:#94A3B8;">30d RPR</span>
                            <span style="font-weight:600;">{sr['30-day RPR (%)']:.1f}%</span>
                            <span style="color:#94A3B8;">60d RPR</span>
                            <span style="font-weight:600;">{sr['60-day RPR (%)']:.1f}%</span>
                            <span style="color:#94A3B8;">Median → 2nd</span>
                            <span style="font-weight:600;">{sr['Median Days to 2nd Order']:.0f}d</span>
                            <span style="color:#94A3B8;">P25 / P75</span>
                            <span style="font-weight:600;">{sr['P25']:.0f}d / {sr['P75']:.0f}d</span>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)

        # ── Grouped bar: RPR by segment × discount ────────────────────────────
        if not basket_detail_f.empty:
            detail_melted = basket_detail_f.melt(
                id_vars=["Basket Segment", "Discount", "N Customers"],
                value_vars=["30-day RPR (%)", "60-day RPR (%)", "90-day RPR (%)"],
                var_name="Window", value_name="RPR",
            )
            detail_melted["Group"] = detail_melted["Basket Segment"] + " · " + detail_melted["Discount"]
            fig_grouped_detail = px.bar(
                detail_melted, x="Window", y="RPR", color="Group",
                barmode="group",
                color_discrete_sequence=[
                    PALETTE["primary"], PALETTE["secondary"],
                    PALETTE["success"], PALETTE["warning"],
                ],
                text=detail_melted["RPR"].apply(lambda v: f"{v:.1f}%"),
            )
            apply_layout(fig_grouped_detail,
                         "RPR Across Windows: Basket Size × Discount",
                         yaxis_title="RPR (%)", yaxis_ticksuffix="%")
            fig_grouped_detail.update_traces(textposition="outside")
            fig_grouped_detail.update_layout(height=400)
            st.plotly_chart(fig_grouped_detail, use_container_width=True, key="bask_grouped_detail")

        # ── Heatmap: basket × discount customer count ─────────────────────────
        hm_data = bd_basket["heatmap_data"].pivot(
            index="basket_segment", columns="Discount", values="Count"
        ).fillna(0)
        fig_hm = go.Figure(go.Heatmap(
            z=hm_data.values,
            x=hm_data.columns.tolist(),
            y=hm_data.index.tolist(),
            text=[[f"{int(v):,}" for v in row] for row in hm_data.values],
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Customers"),
        ))
        apply_layout(fig_hm, "Customer Count Heatmap: Basket Size × Discount Usage",
                     xaxis_title="Discount Used", yaxis_title="Basket Segment")
        fig_hm.update_layout(height=300)
        st.plotly_chart(fig_hm, use_container_width=True, key="bask6_heatmap")

        # ── Observations ──────────────────────────────────────────────────────
        st.markdown("#### 💡 Observations")
        for ins in _basket_discount_insights(bd_basket):
            insight_box(ins, PALETTE["secondary"])

        # Extra: discount sub-segment comparison insight
        if not basket_detail_f.empty:
            lb_disc = basket_detail_f[
                (basket_detail_f["Basket Segment"] == "Large Basket") &
                (basket_detail_f["Discount"] == "With Discount")
            ]
            sb_disc = basket_detail_f[
                (basket_detail_f["Basket Segment"] == "Small Basket") &
                (basket_detail_f["Discount"] == "With Discount")
            ]
            if not lb_disc.empty and not sb_disc.empty:
                ld_rpr = lb_disc.iloc[0]["30-day RPR (%)"]
                sd_rpr = sb_disc.iloc[0]["30-day RPR (%)"]
                higher = "Large Basket" if ld_rpr >= sd_rpr else "Small Basket"
                insight_box(
                    f"<b>Among discount-acquired customers:</b> "
                    f"Large Basket + Discount → <b>{ld_rpr:.1f}%</b> 30d RPR vs "
                    f"Small Basket + Discount → <b>{sd_rpr:.1f}%</b>. "
                    f"<b>{higher}</b> discount buyers show stronger early repeat intent.",
                    PALETTE["primary"],
                )

# # ══════════════════════════════════════════════════════════════════════════════
# # TAB 8 — Business Insights
# # ══════════════════════════════════════════════════════════════════════════════
# with tabs[7]:
#     section_header("Business Insights", "Auto-generated from the data — no hardcoded text")

#     rpr_f2        = rpr_f_base
#     disc_sum_f2   = disc_sum_f_base
#     basket_sum_f2 = basket_sum_f_base
#     disc_type_f2  = disc_type_f_base
#     cohort_pct_f2 = cohort_pct_f_base

#     insights = generate_insights(rpr_f2, t2o_f, disc_sum_f2, basket_sum_f2,
#                                   disc_type_f2, cohort_pct_f2)

#     if not insights:
#         st.info("Run with a full dataset to generate insights.")
#     else:
#         insight_colors = [
#             PALETTE["primary"], PALETTE["secondary"], PALETTE["success"],
#             PALETTE["warning"], PALETTE["danger"],
#         ]
#         for idx, ins in enumerate(insights):
#             st.markdown(f"**Insight {idx + 1}**")
#             insight_box(ins, insight_colors[idx % len(insight_colors)])
#             st.markdown("")

#     # Segment leaderboard
#     st.markdown("---")
#     st.markdown("#### 🏆 Segment Performance Leaderboard")

#     all_segments: list[dict] = []

#     # Discount segments
#     for _, row in disc_sum_f2.iterrows():
#         all_segments.append({
#             "Segment": row["Segment"],
#             "Type": "Discount",
#             "N Customers": row["N Customers"],
#             "30d RPR (%)": row["30-day RPR (%)"],
#             "Median Days to 2nd": row["Median Days to 2nd Order"],
#         })

#     # Basket segments
#     for _, row in basket_sum_f2.iterrows():
#         all_segments.append({
#             "Segment": row["Segment"],
#             "Type": "Basket",
#             "N Customers": row["N Customers"],
#             "30d RPR (%)": row["30-day RPR (%)"],
#             "Median Days to 2nd": row["Median Days to 2nd Order"],
#         })

#     if all_segments:
#         df_seg_lb = pd.DataFrame(all_segments).sort_values("30d RPR (%)", ascending=False)
#         fig_lb = px.bar(
#             df_seg_lb, x="Segment", y="30d RPR (%)",
#             color="Type",
#             color_discrete_sequence=[PALETTE["primary"], PALETTE["success"]],
#             text=df_seg_lb["30d RPR (%)"].apply(fmt_pct),
#             hover_data=["N Customers", "Median Days to 2nd"],
#         )
#         apply_layout(fig_lb, "30-Day Repeat Purchase Rate by Segment",
#                      yaxis_title="RPR (%)", yaxis_ticksuffix="%")
#         fig_lb.update_traces(textposition="outside")
#         fig_lb.update_layout(height=400, xaxis_tickangle=-20)
#         st.plotly_chart(fig_lb, use_container_width=True, key="insights_leaderboard")


# # ══════════════════════════════════════════════════════════════════════════════
# # TAB 9 — Recommendations
# # ══════════════════════════════════════════════════════════════════════════════
# with tabs[8]:
#     section_header("Strategic Recommendations", "Data-driven growth actions based on the analysis")

#     rpr_f3        = rpr_f_base
#     disc_sum_f3   = disc_sum_f_base
#     basket_sum_f3 = basket_sum_f_base
#     disc_type_f3  = disc_type_f_base

#     recs = build_recommendations(rpr_f3, t2o_f, disc_sum_f3, basket_sum_f3, disc_type_f3)

#     if not recs:
#         st.info("Not enough data to generate recommendations under the current filters.")
#     else:
#         for i, rec in enumerate(recs):
#             st.markdown(f"#### Recommendation {i + 1}")
#             recommendation_card(
#                 target   = rec["target"],
#                 campaign = rec["campaign"],
#                 timing   = rec["timing"],
#                 impact   = rec["impact"],
#             )

#     # Prioritisation matrix
#     st.markdown("---")
#     st.markdown("#### 📌 Prioritisation: Effort vs Impact")

#     priority_data = pd.DataFrame([
#         {"Action": "Day 25–30 win-back trigger", "Impact": 9, "Effort": 3, "Size": 40},
#         {"Action": "Subscription up-sell (fast repeaters)", "Impact": 8, "Effort": 5, "Size": 30},
#         {"Action": "Best discount mechanic in acquisition", "Impact": 7, "Effort": 4, "Size": 25},
#         {"Action": "Loyalty tier for large-basket customers", "Impact": 6, "Effort": 6, "Size": 20},
#         {"Action": "Win-back (long tail >P75)", "Impact": 5, "Effort": 3, "Size": 15},
#     ])

#     fig_matrix = px.scatter(
#         priority_data,
#         x="Effort", y="Impact",
#         size="Size",
#         text="Action",
#         color="Action",
#         color_discrete_sequence=QUALITATIVE_SAFE,
#     )
#     fig_matrix.add_hline(y=5, line_dash="dot", line_color="#9CA3AF")
#     fig_matrix.add_vline(x=5, line_dash="dot", line_color="#9CA3AF")
#     fig_matrix.add_annotation(x=2, y=9.5, text="🏆 Quick Wins", showarrow=False,
#                                font=dict(color=PALETTE["success"], size=12))
#     fig_matrix.add_annotation(x=8.5, y=9.5, text="📋 Major Projects", showarrow=False,
#                                font=dict(color=PALETTE["primary"], size=12))
#     fig_matrix.add_annotation(x=2, y=0.5, text="🔧 Fill-ins", showarrow=False,
#                                font=dict(color=PALETTE["neutral"], size=12))
#     fig_matrix.add_annotation(x=8.5, y=0.5, text="❌ Reconsider", showarrow=False,
#                                font=dict(color=PALETTE["danger"], size=12))
#     apply_layout(
#         fig_matrix,
#         "Growth Initiative Prioritisation Matrix",
#         xaxis_title="Implementation Effort (1=Low, 10=High)",
#         yaxis_title="Expected Retention Impact (1=Low, 10=High)",
#         xaxis_range=[0, 11], yaxis_range=[0, 11],
#     )
#     fig_matrix.update_traces(textposition="top center")
#     fig_matrix.update_layout(showlegend=False, height=480)
#     st.plotly_chart(fig_matrix, use_container_width=True, key="recs_matrix")

#     # Footer
#     st.markdown("---")
#     st.caption(
#         "All recommendations are auto-generated from the computed metrics. "
#         "No values are hardcoded — re-run the app with any dataset to regenerate. "
#         "Figures should be validated against A/B test results before full rollout."
#     )