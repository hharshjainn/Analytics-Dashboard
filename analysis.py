"""
analysis.py — All data loading, cleaning, and metric calculations.
Pure functions only; no Streamlit calls here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

RAW_PATH = "ecommerce_orders.csv"


# ── 1. Load & Clean ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading & cleaning data…")
def load_and_clean(path: str = RAW_PATH) -> pd.DataFrame:
    # ── Identify columns needed before reading ────────────────────────────────
    all_cols = pd.read_csv(path, nrows=0).columns.tolist()
    qty_cols  = [c for c in all_cols if c.startswith("line_items") and c.endswith(".quantity")]
    disc_cols = [c for c in all_cols if c.startswith("discount_applications") and c.endswith(".title")]
    # Only read the columns the app actually needs — skip all line_items[*].id
    keep_cols = ["created_at", "customer.id"] + qty_cols + disc_cols

    # ── Read with optimal dtypes upfront ─────────────────────────────────────
    # Quantity cols: read as float32 (they're sparse/nullable floats in CSV)
    dtype_map = {c: "float32" for c in qty_cols}
    # Discount title cols: category saves ~80% vs object
    dtype_map.update({c: "category" for c in disc_cols})

    df = pd.read_csv(
        path,
        usecols=keep_cols,
        dtype=dtype_map,
        low_memory=False,
    )

    # ── Parse dates ───────────────────────────────────────────────────────────
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

    # ── Drop rows with missing essential keys ─────────────────────────────────
    df = df.dropna(subset=["created_at", "customer.id"]).copy()
    df = df.drop_duplicates()

    # customer_id as int64 (values like 6509306970261 exceed int32 range)
    df["customer_id"] = df["customer.id"].astype("int64")
    del df["customer.id"]          # redundant after int conversion

    # ── Basket size ───────────────────────────────────────────────────────────
    df["basket_size"] = df[qty_cols].sum(axis=1, skipna=True).astype("float32")
    # Drop quantity columns after basket_size is computed — not used downstream
    df.drop(columns=qty_cols, inplace=True)

    # ── Discount info ─────────────────────────────────────────────────────────
    if disc_cols:
        # bfill across discount title columns to get first non-null
        # Work on string representation to support category dtype
        disc_df = df[disc_cols].apply(lambda s: s.astype(object))
        df["first_discount_title"] = disc_df.bfill(axis=1).iloc[:, 0].astype("category")
    else:
        df["first_discount_title"] = pd.Categorical([np.nan] * len(df))

    df["has_discount"] = df["first_discount_title"].notna()

    # Drop individual discount title columns — only first_discount_title is used
    df.drop(columns=disc_cols, inplace=True)

    # ── Exclude internal/non-promotional discount codes ───────────────────────
    _EXCLUDED_DISCOUNTS = {
        "amazon replacement order",
        "verified reviews order",
    }
    mask_excluded = (
        df["first_discount_title"]
        .astype(str).str.strip().str.lower()
        .isin(_EXCLUDED_DISCOUNTS)
    )
    df.loc[mask_excluded, "first_discount_title"] = np.nan
    df.loc[mask_excluded, "has_discount"] = False

    # Re-cast to category after nulling out excluded values
    df["first_discount_title"] = df["first_discount_title"].astype("category")

    # ── Order rank per customer ───────────────────────────────────────────────
    df = df.sort_values(["customer_id", "created_at"]).reset_index(drop=True)
    df["order_rank"] = (
        df.groupby("customer_id").cumcount() + 1
    ).astype("int32")

    return df


# ── 2. Derived tables ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_first_orders(df: pd.DataFrame) -> pd.DataFrame:
    fo = (
        df[df["order_rank"] == 1]
        [["customer_id", "created_at", "basket_size", "has_discount", "first_discount_title"]]
        .rename(columns={
            "created_at": "first_order_date",
            "basket_size": "first_basket_size",
            "has_discount": "first_order_discount",
            "first_discount_title": "first_order_discount_title",
        })
        .copy()
    )

    median_basket = fo["first_basket_size"].median()
    fo["basket_segment"] = pd.Categorical(
        np.where(fo["first_basket_size"] > median_basket, "Large Basket", "Small Basket")
    )

    return fo


@st.cache_data(show_spinner=False)
def build_t2o(df: pd.DataFrame, first_orders: pd.DataFrame) -> pd.DataFrame:
    second = (
        df[df["order_rank"] == 2][["customer_id", "created_at"]]
        .rename(columns={"created_at": "second_order_date"})
    )
    t2o = first_orders.merge(second, on="customer_id")
    t2o["days_to_second_order"] = (t2o["second_order_date"] - t2o["first_order_date"]).dt.days
    return t2o


# ── 3. Cohort retention ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_cohort_matrix(
    df: pd.DataFrame,
    first_orders: pd.DataFrame,
    max_period: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cohort_counts, retention_pct) pivot tables."""
    co = df.merge(first_orders[["customer_id", "first_order_date"]], on="customer_id")
    co["cohort_month"] = co["first_order_date"].dt.tz_convert(None).dt.to_period("M")
    co["order_month"]  = co["created_at"].dt.tz_convert(None).dt.to_period("M")
    co["period_number"] = (co["order_month"] - co["cohort_month"]).apply(lambda x: x.n)

    co_p = co[co["period_number"].between(0, max_period)]

    counts = (
        co_p.groupby(["cohort_month", "period_number"])["customer_id"]
        .nunique()
        .reset_index()
        .pivot(index="cohort_month", columns="period_number", values="customer_id")
        .reindex(columns=range(0, max_period + 1))
    )
    sizes = counts[0]
    pct   = counts.divide(sizes, axis=0) * 100
    return counts, pct


# ── 4. Repeat Purchase Rate ───────────────────────────────────────────────────

def repeat_purchase_rate(
    df_orders: pd.DataFrame,
    first_orders_tbl: pd.DataFrame,
    window_days: int,
) -> tuple[int, int, float]:
    merged = df_orders.merge(first_orders_tbl[["customer_id", "first_order_date"]], on="customer_id")
    merged["days_since_first"] = (merged["created_at"] - merged["first_order_date"]).dt.days
    repeaters = merged.loc[
        (merged["days_since_first"] > 0) & (merged["days_since_first"] <= window_days),
        "customer_id",
    ].nunique()
    total = first_orders_tbl["customer_id"].nunique()
    pct = repeaters / total * 100 if total > 0 else 0.0
    return int(repeaters), int(total), float(pct)


@st.cache_data(show_spinner=False)
def build_rpr_table(df: pd.DataFrame, first_orders: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w in [30, 60, 90]:
        repeaters, total, pct = repeat_purchase_rate(df, first_orders, w)
        rows.append({"Window (days)": w, "Repeat Customers": repeaters,
                     "Total Customers": total, "RPR (%)": round(pct, 2)})
    return pd.DataFrame(rows)


# ── 5. Segment comparisons ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_segment_summary(
    t2o: pd.DataFrame,
    first_orders: pd.DataFrame,
    df_orders: pd.DataFrame,
    segment_col: str,
    label_map: dict | None = None,
) -> pd.DataFrame:
    # Pre-merge once outside the loop to avoid N redundant merges
    merged_all = df_orders.merge(first_orders[["customer_id", "first_order_date"]], on="customer_id")
    merged_all["days_since_first"] = (merged_all["created_at"] - merged_all["first_order_date"]).dt.days

    rows = []
    for seg_val, grp in first_orders.groupby(segment_col):
        seg_customers = set(grp["customer_id"])
        seg_t2o = t2o[t2o["customer_id"].isin(seg_customers)]["days_to_second_order"]

        # Filter from the pre-merged frame — no re-merge needed
        seg_merged = merged_all[merged_all["customer_id"].isin(seg_customers)]
        total = len(seg_customers)

        def _rpr(window: int) -> float:
            repeaters = seg_merged.loc[
                (seg_merged["days_since_first"] > 0) & (seg_merged["days_since_first"] <= window),
                "customer_id",
            ].nunique()
            return repeaters / total * 100 if total > 0 else 0.0

        label = (label_map or {}).get(seg_val, str(seg_val))
        rows.append({
            "Segment": label,
            "N Customers": total,
            "Median Days to 2nd Order": round(float(seg_t2o.median()), 1) if len(seg_t2o) else np.nan,
            "P25": round(float(seg_t2o.quantile(0.25)), 1) if len(seg_t2o) else np.nan,
            "P75": round(float(seg_t2o.quantile(0.75)), 1) if len(seg_t2o) else np.nan,
            "30-day RPR (%)": round(_rpr(30), 2),
            "60-day RPR (%)": round(_rpr(60), 2),
            "90-day RPR (%)": round(_rpr(90), 2),
        })
    return pd.DataFrame(rows)


# ── 6. Discount type performance ─────────────────────────────────────────────

def _orders_within_nd(df_orders: pd.DataFrame, first_orders_tbl: pd.DataFrame, n: int) -> pd.Series:
    merged = df_orders.merge(first_orders_tbl[["customer_id", "first_order_date"]], on="customer_id")
    merged["days_since_first"] = (merged["created_at"] - merged["first_order_date"]).dt.days
    within = merged[merged["days_since_first"].between(0, n)]
    return within.groupby("customer_id").size()


@st.cache_data(show_spinner=False)
def build_discount_type_summary(
    t2o: pd.DataFrame,
    first_orders: pd.DataFrame,
    df_orders: pd.DataFrame,
    min_customers: int = 10,
) -> pd.DataFrame:
    """
    Unified year-wise discount performance table.
    Each row = one (Discount Type, Year) cohort.
    """
    discounted = first_orders[first_orders["first_order_discount"]].copy()
    discounted["_date"] = discounted["first_order_date"].dt.tz_convert(None)
    discounted["_year"] = discounted["_date"].dt.year

    # Pre-merge once
    merged_all = df_orders.merge(discounted[["customer_id", "first_order_date", "_year"]], on="customer_id")
    merged_all["days_since_first"] = (merged_all["created_at"] - merged_all["first_order_date"]).dt.days

    rows = []
    for (title, year), grp in discounted.groupby(
        ["first_order_discount_title", "_year"], sort=True
    ):
        if len(grp) < min_customers:
            continue

        seg_customers = set(grp["customer_id"])
        seg_t2o   = t2o[t2o["customer_id"].isin(seg_customers)]["days_to_second_order"]
        seg_merged = merged_all[merged_all["customer_id"].isin(seg_customers)]

        total  = len(seg_customers)
        rpr_30 = (
            seg_merged.loc[
                (seg_merged["days_since_first"] > 0) & (seg_merged["days_since_first"] <= 30),
                "customer_id",
            ].nunique() / total * 100
            if total > 0 else 0.0
        )

        within_90   = seg_merged[seg_merged["days_since_first"].between(0, 90)]
        orders_90   = within_90.groupby("customer_id").size()
        avg_orders_90 = orders_90.reindex(list(seg_customers), fill_value=0).mean()

        start = grp["_date"].min().date()
        end   = grp["_date"].max().date()

        rows.append({
            "Discount Type":            title,
            "Year":                     int(year),
            "N Customers":              len(seg_customers),
            "30-day RPR (%)":           round(rpr_30, 2),
            "Median Days to 2nd Order": round(float(seg_t2o.median()), 1) if len(seg_t2o) else np.nan,
            "Avg Orders in 90d":        round(float(avg_orders_90), 2),
            "Campaign Start":           start,
            "Campaign End":             end,
            "Duration (days)":          (end - start).days,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("30-day RPR (%)", ascending=False)
        .reset_index(drop=True)
    )


# ── 9. Campaign Duration Analysis ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_campaign_duration(first_orders: pd.DataFrame, min_customers: int = 10) -> pd.DataFrame:
    """
    For each (Discount Title, Year), compute:
      - Start Date  = earliest first_order_date
      - End Date    = latest  first_order_date
      - Duration    = End - Start (days)
      - N Customers = distinct customers using that discount in that year
    Only considers customers' first orders (acquisition discounts).
    """
    discounted = first_orders[first_orders["first_order_discount"]].copy()
    discounted["_date"] = discounted["first_order_date"].dt.tz_convert(None)
    discounted["_year"] = discounted["_date"].dt.year

    rows = []
    for (title, year), grp in discounted.groupby(["first_order_discount_title", "_year"]):
        n = len(grp)
        if n < min_customers:
            continue
        start = grp["_date"].min().date()
        end   = grp["_date"].max().date()
        rows.append({
            "Discount Type": title,
            "Year":          int(year),
            "Start Date":    start,
            "End Date":      end,
            "Duration (days)": (end - start).days,
            "N Customers":   n,
        })

    df_out = pd.DataFrame(rows).sort_values(
        ["Discount Type", "Year"]
    ).reset_index(drop=True)
    return df_out


def _campaign_insights(df_cd: pd.DataFrame) -> list[str]:
    """Auto-generate 2–3 insights from the campaign duration table."""
    insights = []
    if df_cd.empty:
        return insights

    # Insight 1 — discounts that span most of the year (duration > 300d in any year)
    long_running = (
        df_cd[df_cd["Duration (days)"] >= 300]
        .groupby("Discount Type")["Year"].apply(list)
    )
    if not long_running.empty:
        names = ", ".join(f"<b>{d}</b>" for d in long_running.index[:3])
        insights.append(
            f"📅 <b>Always-on discounts:</b> {names} ran for 300+ days in at least one year, "
            f"suggesting these are evergreen mechanics rather than seasonal campaigns."
        )

    # Insight 2 — discounts reused across multiple years
    multi_year = df_cd.groupby("Discount Type")["Year"].nunique()
    multi = multi_year[multi_year >= 3].index.tolist()
    if multi:
        names = ", ".join(f"<b>{d}</b>" for d in multi[:3])
        suffix = f" (and {len(multi) - 3} more)" if len(multi) > 3 else ""
        insights.append(
            f"🔄 <b>Multi-year discounts:</b> {names}{suffix} appeared in 3+ years, "
            f"indicating recurring promotional strategies worth tracking for trend changes."
        )

    # Insight 3 — short seasonal campaigns (duration < 30d and only 1 year)
    seasonal = df_cd[(df_cd["Duration (days)"] < 30)]
    if not seasonal.empty:
        top_s = seasonal.sort_values("N Customers", ascending=False).head(3)
        names = ", ".join(f"<b>{r['Discount Type']} ({r['Year']})</b>" for _, r in top_s.iterrows())
        insights.append(
            f"🎯 <b>Short seasonal campaigns (&lt;30 days):</b> {names} — "
            f"concentrated windows with high customer volume relative to duration. "
            f"These are likely event-driven (holiday / flash) promotions."
        )

    return insights[:3]


# ── 10. Basket × Discount Analysis ───────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_basket_discount_analysis(
    first_orders: pd.DataFrame,
    df_orders: pd.DataFrame,
    t2o: pd.DataFrame,
) -> dict:
    """
    Returns a dict with all metrics needed for the Basket × Discount tab.
    Reuses basket_segment and first_order_discount columns from first_orders.
    """
    fo = first_orders.copy()

    # ── Overall Large Basket breakdown ──
    lb = fo[fo["basket_segment"] == "Large Basket"]
    sb = fo[fo["basket_segment"] == "Small Basket"]

    lb_disc   = lb["first_order_discount"].sum()
    lb_nodisc = len(lb) - lb_disc
    sb_disc   = sb["first_order_discount"].sum()
    sb_nodisc = len(sb) - sb_disc

    overview = pd.DataFrame([
        {"Category": "Large Basket + Discount",    "Orders": int(lb_disc),
         "%": round(lb_disc / len(lb) * 100, 1) if len(lb) else 0,   "Segment": "Large Basket"},
        {"Category": "Large Basket + No Discount", "Orders": int(lb_nodisc),
         "%": round(lb_nodisc / len(lb) * 100, 1) if len(lb) else 0, "Segment": "Large Basket"},
        {"Category": "Small Basket + Discount",    "Orders": int(sb_disc),
         "%": round(sb_disc / len(sb) * 100, 1) if len(sb) else 0,   "Segment": "Small Basket"},
        {"Category": "Small Basket + No Discount", "Orders": int(sb_nodisc),
         "%": round(sb_nodisc / len(sb) * 100, 1) if len(sb) else 0, "Segment": "Small Basket"},
    ])

    # ── Comparison table: Large vs Small ──
    def _seg_metrics(seg_fo: pd.DataFrame) -> dict:
        cids = set(seg_fo["customer_id"])
        seg_df = df_orders[df_orders["customer_id"].isin(cids)]
        seg_t2o = t2o[t2o["customer_id"].isin(cids)]["days_to_second_order"]
        _, _, rpr30 = repeat_purchase_rate(seg_df, seg_fo, 30)
        return {
            "N Customers":        len(seg_fo),
            "Disc Usage (%)":     round(seg_fo["first_order_discount"].mean() * 100, 1),
            "Avg Basket (units)": round(seg_fo["first_basket_size"].mean(), 2),
            "30-day RPR (%)":     round(rpr30, 2),
            "Median Days to 2nd": round(float(seg_t2o.median()), 1) if len(seg_t2o) else float("nan"),
        }

    comparison = pd.DataFrame([
        {"Segment": "Large Basket", **_seg_metrics(lb)},
        {"Segment": "Small Basket", **_seg_metrics(sb)},
    ])

    # ── Heatmap: basket_segment × has_discount — count matrix ──
    heatmap_data = (
        fo.groupby(["basket_segment", "first_order_discount"])
        .size()
        .reset_index(name="Count")
    )
    heatmap_data["Discount"] = heatmap_data["first_order_discount"].map(
        {True: "With Discount", False: "No Discount"}
    )

    return {
        "overview":      overview,
        "comparison":    comparison,
        "heatmap_data":  heatmap_data,
        "lb_disc_pct":   round(lb_disc / len(lb) * 100, 1) if len(lb) else 0,
        "sb_disc_pct":   round(sb_disc / len(sb) * 100, 1) if len(sb) else 0,
        "lb_total":      len(lb),
        "sb_total":      len(sb),
    }


@st.cache_data(show_spinner=False)
def build_basket_segment_detail(
    first_orders: pd.DataFrame,
    df_orders: pd.DataFrame,
    t2o: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per (basket_segment × discount) combination,
    including customer counts, RPR metrics and time-to-second-order stats.
    Rows: Large+Disc, Large+No Disc, Small+Disc, Small+No Disc
    """
    # Pre-merge once
    merged_all = df_orders.merge(
        first_orders[["customer_id", "first_order_date"]], on="customer_id"
    )
    merged_all["days_since_first"] = (
        merged_all["created_at"] - merged_all["first_order_date"]
    ).dt.days

    rows = []
    for basket_seg in ["Large Basket", "Small Basket"]:
        for disc_used, disc_label in [(True, "With Discount"), (False, "No Discount")]:
            grp = first_orders[
                (first_orders["basket_segment"] == basket_seg)
                & (first_orders["first_order_discount"] == disc_used)
            ]
            if grp.empty:
                continue

            cids       = set(grp["customer_id"])
            seg_merged = merged_all[merged_all["customer_id"].isin(cids)]
            seg_t2o    = t2o[t2o["customer_id"].isin(cids)]["days_to_second_order"]
            total      = len(cids)

            def _rpr(window: int) -> float:
                rep = seg_merged.loc[
                    (seg_merged["days_since_first"] > 0)
                    & (seg_merged["days_since_first"] <= window),
                    "customer_id",
                ].nunique()
                return rep / total * 100 if total > 0 else 0.0

            rows.append({
                "Basket Segment":          basket_seg,
                "Discount":                disc_label,
                "N Customers":             total,
                "30-day RPR (%)":          round(_rpr(30), 2),
                "60-day RPR (%)":          round(_rpr(60), 2),
                "90-day RPR (%)":          round(_rpr(90), 2),
                "Median Days to 2nd Order": round(float(seg_t2o.median()), 1) if len(seg_t2o) else np.nan,
                "P25": round(float(seg_t2o.quantile(0.25)), 1) if len(seg_t2o) else np.nan,
                "P75": round(float(seg_t2o.quantile(0.75)), 1) if len(seg_t2o) else np.nan,
            })

    return pd.DataFrame(rows)


def _basket_discount_insights(bd: dict) -> list[str]:
    insights = []
    lb_pct = bd["lb_disc_pct"]
    sb_pct = bd["sb_disc_pct"]
    diff   = lb_pct - sb_pct

    insights.append(
        f"🛒 <b>{lb_pct:.1f}% of Large Basket first orders used a discount</b>, "
        f"compared to {sb_pct:.1f}% for Small Basket orders "
        f"({'a {:.1f} pp higher rate'.format(diff) if diff > 0 else 'a {:.1f} pp lower rate'.format(abs(diff))})."
    )

    cmp = bd["comparison"]
    lb_row = cmp[cmp["Segment"] == "Large Basket"].iloc[0]
    sb_row = cmp[cmp["Segment"] == "Small Basket"].iloc[0]

    if lb_row["30-day RPR (%)"] > sb_row["30-day RPR (%)"]:
        insights.append(
            f"🔁 <b>Large Basket customers show higher 30-day RPR</b> "
            f"({lb_row['30-day RPR (%)']:.1f}% vs {sb_row['30-day RPR (%)']:.1f}%), "
            f"suggesting higher-quantity first orders correlate with stronger repeat intent — "
            f"regardless of discount usage."
        )
    else:
        insights.append(
            f"🔁 <b>Small Basket customers show higher 30-day RPR</b> "
            f"({sb_row['30-day RPR (%)']:.1f}% vs {lb_row['30-day RPR (%)']:.1f}%), "
            f"suggesting frequent low-quantity buyers are the more habitual segment."
        )

    if abs(diff) >= 5:
        direction = "more" if diff > 0 else "less"
        insights.append(
            f"🏷 <b>Observed relationship:</b> customers placing larger first-order baskets are "
            f"<b>{direction} likely to have used a discount</b> at acquisition ({lb_pct:.1f}% vs "
            f"{sb_pct:.1f}%). This may reflect deal-driven bulk-buying behaviour during "
            f"promotional campaigns — not a causal claim."
        )

    return insights


def generate_insights(
    rpr_table: pd.DataFrame,
    t2o: pd.DataFrame,
    discount_summary: pd.DataFrame,
    basket_summary: pd.DataFrame,
    discount_type_summary: pd.DataFrame,
    cohort_pct: pd.DataFrame,
) -> list[str]:
    insights = []

    # Insight 1: Month-0 → Month-1 drop
    if 1 in cohort_pct.columns:
        avg_m1 = cohort_pct[1].dropna().mean()
        insights.append(
            f"📉 <b>Biggest churn cliff is Month 0→1:</b> on average only "
            f"<b>{avg_m1:.1f}%</b> of customers return in the month after their first purchase. "
            f"Winning this window has the highest retention ROI."
        )

    # Insight 2: 30 vs 90-day RPR gap
    rpr_30 = rpr_table.set_index("Window (days)").loc[30, "RPR (%)"]
    rpr_90 = rpr_table.set_index("Window (days)").loc[90, "RPR (%)"]
    gap    = rpr_90 - rpr_30
    insights.append(
        f"🔁 <b>Repeat buying is a slow-burn behaviour:</b> 30-day RPR is "
        f"<b>{rpr_30:.1f}%</b> vs 90-day RPR of <b>{rpr_90:.1f}%</b> — a "
        f"<b>{gap:.1f} pp gap</b> means most loyal customers only reveal themselves "
        f"2–3 months post-acquisition."
    )

    # Insight 3: Median time to second order
    median_t2o = t2o["days_to_second_order"].median()
    p75_t2o    = t2o["days_to_second_order"].quantile(0.75)
    insights.append(
        f"⏱ <b>Optimal win-back trigger window:</b> the median time to a second "
        f"order is <b>{median_t2o:.0f} days</b> (P75 = {p75_t2o:.0f} days). "
        f"A Day 25–30 trigger catches repeaters just before they naturally convert, "
        f"while also nudging the long tail."
    )

    # Insight 4: Discount vs no-discount RPR
    if not discount_summary.empty and len(discount_summary) >= 2:
        disc_row     = discount_summary[discount_summary["Segment"].str.contains("Discount", case=False) &
                                        ~discount_summary["Segment"].str.contains("No", case=False)]
        no_disc_row  = discount_summary[discount_summary["Segment"].str.contains("No Discount", case=False)]
        if not disc_row.empty and not no_disc_row.empty:
            disc_rpr    = disc_row.iloc[0]["30-day RPR (%)"]
            no_disc_rpr = no_disc_row.iloc[0]["30-day RPR (%)"]
            direction   = "higher" if disc_rpr > no_disc_rpr else "lower"
            interpretation = (
                "suggesting discounts are successfully building habit"
                if disc_rpr > no_disc_rpr
                else "suggesting discount-acquired customers skew deal-seekers, not brand loyalists"
            )
            insights.append(
                f"🏷 <b>Discount acquisition effect:</b> customers who used a discount on their "
                f"first order have a <b>{direction} 30-day RPR</b> ({disc_rpr:.1f}% vs "
                f"{no_disc_rpr:.1f}%) — {interpretation}."
            )

    return insights[:4]


# ── 8. Recommendation builder ─────────────────────────────────────────────────

def build_recommendations(
    rpr_table: pd.DataFrame,
    t2o: pd.DataFrame,
    discount_summary: pd.DataFrame,
    basket_summary: pd.DataFrame,
    discount_type_summary: pd.DataFrame,
) -> list[dict]:
    median_t2o = t2o["days_to_second_order"].median()
    rpr_30     = rpr_table.set_index("Window (days)").loc[30, "RPR (%)"]

    recs = []

    # Rec 1: Day-25 win-back trigger
    trigger_day = max(20, int(median_t2o * 0.85))
    recs.append({
        "target":   "All first-time buyers who haven't placed a second order",
        "campaign": "Personalised replenishment / 'we miss you' email + SMS triggered at day "
                    f"{trigger_day}–{trigger_day + 5}, using the category of their first purchase.",
        "timing":   f"Day {trigger_day}–{trigger_day + 5} post first purchase "
                    f"(85% of median time-to-second-order = {median_t2o:.0f} days)",
        "impact":   f"Lift the 30-day RPR baseline of {rpr_30:.1f}% by catching fence-sitters "
                    f"just before they would naturally convert — even a +2–3 pp improvement "
                    f"compounds significantly at scale.",
    })

    # Rec 2: Segment by basket size
    if not basket_summary.empty:
        large_rpr = basket_summary[basket_summary["Segment"] == "Large Basket"]["30-day RPR (%)"].values
        small_rpr = basket_summary[basket_summary["Segment"] == "Small Basket"]["30-day RPR (%)"].values
        if len(large_rpr) and len(small_rpr):
            higher_seg = "Large Basket" if large_rpr[0] >= small_rpr[0] else "Small Basket"
            lower_seg  = "Small Basket" if higher_seg == "Large Basket" else "Large Basket"
            recs.append({
                "target": f"{higher_seg} customers (first-order quantity > median)",
                "campaign": "Loyalty or bundle up-sell: invite into subscription / VIP tier "
                            "while they're still in the honeymoon period.",
                "timing": "Day 7–10 post first purchase (capitalise on high-intent signal early).",
                "impact": f"{higher_seg} customers already show a higher 30-day RPR "
                          f"({max(large_rpr[0], small_rpr[0]):.1f}%). Early reinforcement can "
                          f"convert them into habitual buyers before competitors can.",
            })



    return recs
