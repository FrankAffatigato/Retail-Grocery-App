import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta

# -------- Generate Sample Retail Dataset --------
def generate_data():
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq='D')
    products = ['Milk', 'Bread', 'Eggs', 'Butter', 'Apples', 'Bananas', 'Chicken', 'Beef']
    categories = {'Milk': 'Dairy', 'Bread': 'Bakery', 'Eggs': 'Dairy', 'Butter': 'Dairy',
                  'Apples': 'Produce', 'Bananas': 'Produce', 'Chicken': 'Meat', 'Beef': 'Meat'}
    stores = ['Store A', 'Store B', 'Store C', 'Store D']
    regions = ['North', 'South', 'East', 'West']

    data = []
    for date in dates:
        for store in stores:
            for product in products:
                units_sold = np.random.poisson(20)
                price = round(np.random.uniform(1.0, 10.0), 2)
                cost = round(price * np.random.uniform(0.5, 0.8), 2)
                revenue = units_sold * price
                region = np.random.choice(regions)
                data.append({
                    'Date': date,
                    'Store': store,
                    'Region': region,
                    'Product': product,
                    'Category': categories[product],
                    'Units Sold': units_sold,
                    'Revenue': revenue,
                    'Cost': cost * units_sold
                })
    return pd.DataFrame(data)

# -------- Load Data --------
df = generate_data()

# -------- Streamlit UI --------
st.set_page_config(layout="wide")
st.title("🛒 Retail Sales Dashboard")

# Sidebar Filters
with st.sidebar:
    st.header("Filter Options")
    date_range = st.date_input("Select Date Range", [df["Date"].min(), df["Date"].max()])
    category_filter = st.selectbox("Select Category", options=["All"] + sorted(df['Category'].unique().tolist()))
    store_filter = st.multiselect("Select Stores", options=sorted(df['Store'].unique()), default=df['Store'].unique())

# Apply filters
filtered_df = df.copy()
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[(filtered_df['Date'] >= pd.to_datetime(start_date)) &
                              (filtered_df['Date'] <= pd.to_datetime(end_date))]
if category_filter != "All":
    filtered_df = filtered_df[filtered_df['Category'] == category_filter]
if store_filter:
    filtered_df = filtered_df[filtered_df['Store'].isin(store_filter)]

# KPIs
total_revenue = filtered_df['Revenue'].sum()
total_units = filtered_df['Units Sold'].sum()
total_profit = (filtered_df['Revenue'] - filtered_df['Cost']).sum()

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("💰 Total Revenue", f"${total_revenue:,.0f}")
kpi2.metric("📦 Units Sold", f"{total_units:,}")
kpi3.metric("📈 Gross Profit", f"${total_profit:,.0f}")

# Charts
st.markdown("### 📊 Revenue Over Time")
revenue_trend = filtered_df.groupby("Date")["Revenue"].sum().reset_index()
st.altair_chart(
    alt.Chart(revenue_trend).mark_line().encode(
        x="Date:T",
        y="Revenue:Q"
    ).properties(width=800, height=300),
    use_container_width=True
)

st.markdown("### 🧾 Revenue by Category")
category_chart = filtered_df.groupby("Category")["Revenue"].sum().reset_index()
st.bar_chart(category_chart.set_index("Category"))

st.markdown("### 📍 Revenue by Region")
region_chart = filtered_df.groupby("Region")["Revenue"].sum().reset_index()
st.bar_chart(region_chart.set_index("Region"))

# Raw Data Expander
with st.expander("🔍 View Raw Data"):
    st.dataframe(filtered_df.reset_index(drop=True))
