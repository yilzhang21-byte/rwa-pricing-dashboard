import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="RWA Pricing Deviation Dashboard", layout="wide")
st.title("🏦 RWA Asset Pricing Deviation Monitor")
st.markdown("### Real-Time Pricing Deviation Monitor for RWA Assets")
st.markdown("**OUSG (Treasury Token) vs MPL (Governance Token)** | Data Range: 2024-01-01 to Present")

# ---------- Data Loading ----------
@st.cache_data
def load_data():
    # OUSG - 改成相对路径
    ousg = pd.read_csv('ousg_merged.csv')
    ousg['date'] = pd.to_datetime(ousg['date'])
    ousg = ousg[ousg['date'] >= '2024-01-01']
    ousg['asset'] = 'OUSG'
    ousg['asset_en'] = 'OUSG (Treasury)'
    
    # MPL - 改成相对路径
    mpl = pd.read_csv('mpl-usd-max.csv')
    mpl = mpl[['snapped_at', 'price']].copy()
    mpl.columns = ['date', 'price_usd']
    mpl['date'] = pd.to_datetime(mpl['date'])
    mpl = mpl[mpl['date'] >= '2024-01-01']
    mpl['nav_usd'] = mpl['price_usd'].rolling(30, min_periods=1).mean()
    mpl['gap'] = (mpl['price_usd'] - mpl['nav_usd']) / mpl['nav_usd'] * 100
    mpl['asset'] = 'MPL'
    mpl['asset_en'] = 'MPL (Governance)'
    
    # Merge
    df = pd.concat([
        ousg[['date', 'price_usd', 'nav_usd', 'gap', 'asset', 'asset_en']],
        mpl[['date', 'price_usd', 'nav_usd', 'gap', 'asset', 'asset_en']]
    ])
    return df

df = load_data()

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Filters")
assets = st.sidebar.multiselect("Select Assets", df['asset'].unique(), default=['OUSG', 'MPL'])
lookback = st.sidebar.slider("Lookback Days", 7, 90, 30)

df_filtered = df[df['asset'].isin(assets)]
df_filtered = df_filtered[df_filtered['date'] >= df_filtered['date'].max() - pd.Timedelta(days=lookback)]

# ---------- Current Gap Cards ----------
st.subheader("📊 Current Pricing Deviation (Gap)")
cols = st.columns(len(assets))
for i, asset in enumerate(assets):
    latest = df[df['asset'] == asset].iloc[-1]
    asset_en = df[df['asset'] == asset]['asset_en'].iloc[-1]
    cols[i].metric(
        label=f"{asset_en}",
        value=f"{latest['gap']:.2f}%",
        delta=f"Price ${latest['price_usd']:.2f} | NAV ${latest['nav_usd']:.2f}"
    )

# ---------- Experimental Health Score (MPL only) ----------
st.subheader("🧪 MPL Experimental Health Score")
if "MPL" in assets:
    mpl_data = df[df['asset'] == 'MPL'].iloc[-1]
    mpl_gap = mpl_data['gap']
    score = max(0, min(100, 50 - abs(mpl_gap) * 5))
    st.metric(
        label="MPL Health Score (gap-based)",
        value=f"{score:.0f} / 100",
        delta="Higher is healthier",
        delta_color="off"
    )
    st.caption("📐 Formula: base 50 − |gap|×5. Closer to 100 → smaller pricing deviation.")
else:
    st.info("Select MPL to see its gap‑based health score.")
    
# ---------- Historical Gap Trend ----------
st.subheader("📈 Historical Gap Trend")
fig = px.line(df_filtered, x='date', y='gap', color='asset_en', 
              title="Daily Gap Change (%)")
fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Upper Alert")
fig.add_hline(y=-0.5, line_dash="dash", line_color="red", annotation_text="Lower Alert")
fig.update_layout(legend_title_text='Asset')
st.plotly_chart(fig, use_container_width=True)

# ---------- Price vs NAV Scatter ----------
st.subheader("🔍 Price vs NAV")
fig2 = px.scatter(df_filtered, x='nav_usd', y='price_usd', color='asset_en', 
                  hover_data=['date', 'gap'], title='Price vs NAV')
# 45-degree reference line
fig2.add_shape(
    type='line',
    x0=df_filtered['nav_usd'].min(),
    y0=df_filtered['nav_usd'].min(),
    x1=df_filtered['nav_usd'].max(),
    y1=df_filtered['nav_usd'].max(),
    line=dict(dash='dash', color='gray')
)
st.plotly_chart(fig2, use_container_width=True)

# ---------- Anomaly Alerts ----------
st.subheader("🚨 Recent Anomaly Alerts (|Gap| > 0.5%)")
df_filtered['alert'] = abs(df_filtered['gap']) > 0.5
alerts = df_filtered[df_filtered['alert']].sort_values('date', ascending=False)
if len(alerts) > 0:
    st.dataframe(alerts[['date', 'asset_en', 'gap', 'price_usd', 'nav_usd']].head(10))
else:
    st.success("✅ No abnormal deviations in the last 30 days")

# ---------- Asset Comparison Table ----------
st.subheader("📋 Asset Pricing Comparison")
summary = df.groupby('asset_en').agg(
    LatestGap=('gap', 'last'),
    MeanGap=('gap', 'mean'),
    StdGap=('gap', 'std'),
    LatestPrice=('price_usd', 'last')
).round(2)
summary.columns = ['Latest Gap (%)', 'Mean Gap (%)', 'Std Gap (%)', 'Latest Price ($USD)']
st.dataframe(summary)

st.caption("Note: OUSG NAV is back-calculated from official data; MPL uses 30-day MA as NAV proxy.")