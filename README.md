# 📊 E-Commerce Retention Analytics Dashboard

A production-grade Streamlit dashboard for customer retention and repeat-purchase analysis, built from a Shopify-style order export.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place your data file

Copy your Shopify order export to the project root:

```
ecommerce_orders.csv   ← same folder as app.py
```

### 3. Run the dashboard

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## 📁 Project Structure

```
├── app.py              # Main Streamlit application & all tab rendering
├── analysis.py         # Pure data functions: loading, cleaning, metrics
├── utils.py            # UI helpers: colours, chart layout, cards, callouts
├── requirements.txt    # Pinned Python dependencies
└── README.md           # This file
```

---

## 📋 Dashboard Sections

| Tab | What it shows |
|-----|--------------|
| 🏠 Executive Summary | KPI cards, RPR bar chart, customer funnel |
| 🔵 Cohort Retention | Monthly cohort heatmap, average retention curve |
| 🔁 Repeat Purchase | 30/60/90-day RPR bar chart, incremental waterfall |
| ⏱ Time to 2nd Order | Histogram, boxplot, P25/P50/P75 stats |
| 🏷 Discount Analysis | Discount vs no-discount RPR & time-to-second-order |
| 🛒 Basket Analysis | Large vs small first-order basket comparison |
| 📋 Discount Types | Searchable/sortable table + top/bottom performer charts |
| 💡 Insights | Auto-generated, data-driven narrative bullets |
| 🎯 Recommendations | Prioritised campaigns with effort/impact matrix |

---

## 🎛 Sidebar Filters

All sections respond live to:

- **RPR Time Window** — 30 / 60 / 90 days
- **Discount Type** — filter to a specific promo code
- **Basket Segment** — Large Basket / Small Basket / All
- **Cohort Month Range** — slide to focus on specific cohort windows

---

## 📦 Required CSV Schema

The CSV must be a Shopify-style flat order export containing at minimum:

| Column | Description |
|--------|-------------|
| `created_at` | ISO-8601 order timestamp (mixed UTC offsets supported) |
| `customer.id` | Numeric customer identifier |
| `line_items[N].quantity` | Quantity for line-item N (N = 0, 1, 2, …) |
| `discount_applications[N].title` | Discount title for slot N (optional) |

Missing `customer.id` or `created_at` rows are automatically dropped (<1% of typical exports).

---

## 🧠 How Metrics Are Calculated

- **Repeat Purchase Rate (RPR):** % of customers whose 2nd+ order occurred within N days of their *first* order date.
- **Time to Second Order:** calendar days between `order_rank=1` and `order_rank=2` for customers with ≥2 orders.
- **Cohort Month:** the calendar month of a customer's first (`order_rank=1`) purchase.
- **Basket Size:** sum of all `line_items[*].quantity` values on the order (proxy for order value when revenue is absent).
- **Large Basket:** first-order basket size strictly above the median.

---

## ⚙️ Performance Notes

- All heavy computations are wrapped in `@st.cache_data` — the dataset is processed only once per session.
- Sidebar filters rebuild only the filtered subset; cached base tables are reused.
- Tested with datasets up to ~500 k orders on a standard laptop.

---

## 📌 Design Principles

- **Storytelling first:** every section ends with auto-generated, data-driven insights.
- **Zero hardcoding:** all numbers in insights and recommendations are computed from the live dataset.
- **Executive-friendly:** KPI cards + funnel on the landing tab; deep-dive tabs for analysts.
- **Portfolio-grade:** Inter font, consistent Indigo/Cyan/Emerald palette, responsive Plotly charts with hover tooltips.
