"""
Temu Ireland Dashboard - Synthetic Data Pipeline
================================================

Combines a real Temu product catalog (Kaggle) with a synthetically
generated Irish order history, producing a star schema for Power BI:

    output/dim_products.csv   - real catalog, cleaned (~94,749 products)
    output/dim_customers.csv  - synthetic Irish customers (9,000)
    output/fact_orders.csv    - synthetic orders, Jan 2025 - Jun 2026 (60,000)

Design notes
------------
- Products are sampled proportionally to their REAL Temu sales_volume
  (sqrt-damped so bestsellers don't dominate every order).
- Counties are weighted by actual Irish population (Dublin ~28%).
- Seasonality: weekend lift, Black Friday week spike, Christmas ramp,
  and ~1.8x year-over-year growth into 2026.
- Payment methods reflect the Irish market (Card, PayPal, Apple Pay,
  Google Pay, Klarna).
- Product titles are scrubbed of quote characters and newlines, which
  otherwise break CSV parsing in Power Query (12 rows in the raw
  catalog contained stray double quotes).

Source dataset (place in the same folder before running):
    temu_product_sales_dataset.csv
    https://www.kaggle.com/datasets/polartech/temu-dataset-us-online-mareket-place

Usage:
    pip install pandas numpy
    python generate_temu_ireland.py
"""

import os
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)   
USD_TO_EUR = 0.92

SOURCE_CSV = "temu_product_sales_dataset.csv"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)


raw = pd.read_csv(SOURCE_CSV, encoding="utf-8-sig")

dim = (raw
       .rename(columns={
           "leve_1_category_id": "category_l1_id",
           "leve_1_category_name": "category_l1",
           "leve_2_category_id": "category_l2_id",
           "leve_2_category_name": "category_l2",
           "title": "product_name",
           "goods_score": "rating",
           "comment_num": "review_count"})
       .drop(columns=["sales_info", "comments_num_raw", "visible",
                      "price_str", "category_id"])
       .drop_duplicates(subset="goods_id", keep="first")
       .copy())

dim["sales_volume"] = dim["sales_volume"].fillna(1).astype(int)
dim["rating"] = dim["rating"].fillna(dim["rating"].median()).round(1)
dim["review_count"] = dim["review_count"].fillna(0).astype(int)
dim["unit_price_eur"] = (dim["price"] * USD_TO_EUR).round(2)
dim = dim.drop(columns=["price"])

dim["product_name"] = (dim["product_name"]
                       .str.replace('"', "", regex=False)
                       .str.replace("'", "", regex=False)
                       .str.replace("\n", " ", regex=False)
                       .str.replace("\r", " ", regex=False)
                       .str.strip().str.slice(0, 120))

print(f"dim_products: {len(dim):,} rows")


counties = {
    "Dublin": (1458, "Dublin"), "Cork": (584, "Cork"),
    "Galway": (277, "Galway"), "Kildare": (247, "Naas"),
    "Meath": (220, "Navan"), "Limerick": (209, "Limerick"),
    "Tipperary": (168, "Clonmel"), "Donegal": (167, "Letterkenny"),
    "Wexford": (164, "Wexford"), "Kerry": (156, "Tralee"),
    "Wicklow": (155, "Bray"), "Louth": (139, "Drogheda"),
    "Mayo": (138, "Castlebar"), "Clare": (127, "Ennis"),
    "Waterford": (127, "Waterford"), "Kilkenny": (104, "Kilkenny"),
    "Westmeath": (96, "Athlone"), "Laois": (92, "Portlaoise"),
    "Offaly": (83, "Tullamore"), "Cavan": (81, "Cavan"),
    "Sligo": (70, "Sligo"), "Roscommon": (70, "Roscommon"),
    "Monaghan": (65, "Monaghan"), "Carlow": (62, "Carlow"),
    "Longford": (47, "Longford"), "Leitrim": (35, "Carrick-on-Shannon")}

county_names = list(counties)
county_w = np.array([v[0] for v in counties.values()], dtype=float)
county_p = county_w / county_w.sum()


N_CUST = 9000
cust_county = rng.choice(county_names, size=N_CUST, p=county_p)

customers = pd.DataFrame({
    "customer_id": [f"C{100000 + i}" for i in range(N_CUST)],
    "county": cust_county,
    "city": [counties[c][1] for c in cust_county],
    "segment": rng.choice(["New", "Regular", "Premium", "Inactive"],
                          size=N_CUST, p=[0.28, 0.42, 0.14, 0.16]),
    "age_band": rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"],
                           size=N_CUST, p=[0.24, 0.34, 0.22, 0.13, 0.07]),
    "signup_date": pd.to_datetime("2024-06-01") +
                   pd.to_timedelta(rng.integers(0, 600, N_CUST), unit="D")})

print(f"dim_customers: {len(customers):,} rows")


dates = pd.date_range("2025-01-01", "2026-06-30", freq="D")

month_season = {1: 0.80, 2: 0.85, 3: 0.95, 4: 1.00, 5: 1.05, 6: 1.05,
                7: 1.00, 8: 1.05, 9: 1.10, 10: 1.20, 11: 1.60, 12: 1.45}
dow_season = {0: 0.95, 1: 0.90, 2: 0.92, 3: 0.98, 4: 1.05, 5: 1.15, 6: 1.10}

daily_w = np.array([
    (1.0 if d.year == 2025 else 1.8)
    * month_season[d.month] * dow_season[d.dayofweek]
    * (2.2 if (d.month == 11 and 24 <= d.day <= 30) else 1.0)  # Black Friday
    for d in dates])
daily_p = daily_w / daily_w.sum()

N_ORDERS = 60000
order_dates = pd.to_datetime(dates[rng.choice(len(dates), size=N_ORDERS, p=daily_p)])

# Products: sampled by sqrt(real sales_volume)
prod_w = np.sqrt(dim["sales_volume"].to_numpy(dtype=float))
prod_idx = rng.choice(len(dim), size=N_ORDERS, p=prod_w / prod_w.sum())

# Customers: Regular/Premium order more often than New/Inactive
seg_mult = customers["segment"].map(
    {"New": 0.7, "Regular": 1.4, "Premium": 2.2, "Inactive": 0.15}).to_numpy()
cust_idx = rng.choice(N_CUST, size=N_ORDERS, p=seg_mult / seg_mult.sum())

orders = pd.DataFrame({
    "order_id": [f"IE{2000000 + i}" for i in range(N_ORDERS)],
    "order_date": order_dates,
    "customer_id": customers["customer_id"].to_numpy()[cust_idx],
    "goods_id": dim["goods_id"].to_numpy()[prod_idx],
    "quantity": rng.choice([1, 2, 3, 4], size=N_ORDERS,
                           p=[0.62, 0.24, 0.10, 0.04]),
    "unit_price_eur": dim["unit_price_eur"].to_numpy()[prod_idx],
    "payment_method": rng.choice(
        ["Card", "PayPal", "Apple Pay", "Google Pay", "Klarna"],
        size=N_ORDERS, p=[0.44, 0.24, 0.15, 0.09, 0.08]),
    "delivery_days": rng.integers(6, 15, size=N_ORDERS),
    "order_status": rng.choice(
        ["Delivered", "In Transit", "Returned", "Cancelled"],
        size=N_ORDERS, p=[0.87, 0.06, 0.05, 0.02])})

orders["revenue_eur"] = (orders["quantity"] * orders["unit_price_eur"]).round(2)
orders = orders.sort_values("order_date").reset_index(drop=True)
orders["order_date"] = orders["order_date"].dt.date

h1_25 = (pd.to_datetime(orders.order_date) < "2025-07-01").sum()
h1_26 = (pd.to_datetime(orders.order_date) >= "2026-01-01").sum()
print(f"fact_orders: {len(orders):,} rows | H1 2025: {h1_25:,} | H1 2026: {h1_26:,}")
print(f"Total revenue: EUR {orders['revenue_eur'].sum():,.0f}")

dim.to_csv(f"{OUT_DIR}/dim_products.csv", index=False)
customers.to_csv(f"{OUT_DIR}/dim_customers.csv", index=False)
orders.to_csv(f"{OUT_DIR}/fact_orders.csv", index=False)
print(f"Saved 3 files to ./{OUT_DIR}/")