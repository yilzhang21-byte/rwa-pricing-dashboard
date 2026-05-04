import pandas as pd
import numpy as np

# 1. 读取你的价格数据
df_price = pd.read_csv('C:/Users/91290/Downloads/ousg-usd-max.csv')

# 查看列名，确认哪一列是日期和价格
print("列名：", df_price.columns.tolist())cd C:\Users\91290\Downloads

df_price = df_price[['snapped_at', 'price', 'total_volume']].copy()
df_price.columns = ['date', 'price_usd', 'volume_usd']

# 转换日期格式
df_price['date'] = pd.to_datetime(df_price['date'])
df_price = df_price.sort_values('date').reset_index(drop=True)

# 生成完整日期范围
dates = df_price['date']

# 2. 当前 NAV
latest_nav = 115.0278

# 3. 年化收益率 3.48%
annual_yield = 0.0348
daily_return = (1 + annual_yield) ** (1/365) - 1

# 4. 反向推算历史 NAV
nav_values = [latest_nav]
for i in range(1, len(dates)):
    prev_nav = nav_values[-1] / (1 + daily_return)
    nav_values.append(prev_nav)

df_nav = pd.DataFrame({'date': dates, 'nav_usd': nav_values[::-1]})

# 5. 合并
df = pd.merge(df_price, df_nav, on='date', how='left')
df['gap'] = (df['price_usd'] - df['nav_usd']) / df['nav_usd'] * 100

# 6. 查看结果
print(df[['date', 'price_usd', 'nav_usd', 'gap', 'volume_usd']].head(10))
print(f"数据范围: {df['date'].min()} 到 {df['date'].max()}")
print(f"总行数: {len(df)}")

df.to_csv('C:/Users/91290/Downloads/ousg_merged.csv', index=False)

print(df[['date', 'price_usd', 'nav_usd', 'gap', 'volume_usd']].head(20))

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

# 读取数据
df = pd.read_csv('C:/Users/91290/Downloads/ousg_merged.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# 只保留 2024 年之后
df = df[df['date'] >= '2024-01-01'].copy()

# 填充缺失日期
df = df.set_index('date').asfreq('D')
df['price_usd'] = df['price_usd'].fillna(method='ffill')
df['nav_usd'] = df['nav_usd'].fillna(method='ffill')
df['gap'] = df['gap'].fillna(method='ffill')
df = df.reset_index()

# 构造特征（不用 volume）
df['gap_target'] = df['gap'].shift(-1)
df['gap_lag1'] = df['gap'].shift(1)
df['gap_lag3'] = df['gap'].shift(3)
df['gap_lag7'] = df['gap'].shift(7)
df['returns'] = df['price_usd'].pct_change()
df['volatility_7d'] = df['returns'].rolling(7).std() * 100

# 删除缺失值
df = df.dropna().reset_index(drop=True)

print(f"最终行数: {len(df)}")

# 特征列（只用这4个）
feature_cols = ['gap_lag1', 'gap_lag3', 'gap_lag7', 'volatility_7d']
X = df[feature_cols]
y = df['gap_target']

# 划分
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 训练
model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
mae = mean_absolute_error(y_test, y_pred)
print(f"测试集 MAE: {mae:.6f} 个百分点")
print(f"基准 MAE: {abs(y_test - y_test.mean()).mean():.6f} 个百分点")

print(f"gap 均值: {df['gap'].mean():.6f}")
print(f"gap 标准差: {df['gap'].std():.6f}")
print(f"gap 变化幅度: {df['gap'].max() - df['gap'].min():.6f}")
print(df[['date', 'gap']].set_index('date').plot(figsize=(12,4)))

print("特征与 gap_target 的相关系数：")
for col in ['gap_lag1', 'gap_lag3', 'gap_lag7', 'volatility_7d']:
    corr = df[col].corr(df['gap_target'])
    print(f"{col}: {corr:.4f}")

import pandas as pd
import numpy as np

# 读取 MPL 价格数据
df_price = pd.read_csv('C:/Users/91290/Downloads/mpl-usd-max.csv')  # 你的文件路径

# 查看列名
print("列名：", df_price.columns.tolist())

# 提取需要的列（假设是 'snapped_at', 'price', 'total_volume'）
df_price = df_price[['snapped_at', 'price']].copy()
df_price.columns = ['date', 'price_usd']
df_price['date'] = pd.to_datetime(df_price['date'])
df_price = df_price.sort_values('date').reset_index(drop=True)

# 只保留 2024 年之后
df_price = df_price[df_price['date'] >= '2024-01-01'].copy()

# MPL 没有官方 NAV，用当前价格作为 NAV 基准
# 取最近一天的价格作为参考值
latest_price = df_price['price_usd'].iloc[-1]
print(f"最新价格（作为 NAV 参考）: ${latest_price:.4f}")

# 构造 NAV（假设变化极小，用移动平均平滑）
df_price['nav_usd'] = df_price['price_usd'].rolling(30, min_periods=1).mean()
df_price['gap'] = (df_price['price_usd'] - df_price['nav_usd']) / df_price['nav_usd'] * 100

# 查看结果
print(df_price[['date', 'price_usd', 'nav_usd', 'gap']].head(10))
print(f"数据范围: {df_price['date'].min()} 到 {df_price['date'].max()}")
print(f"总行数: {len(df_price)}")

from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

# 复制数据
df = df_price.copy()

# 构造特征
df['gap_target'] = df['gap'].shift(-1)
df['gap_lag1'] = df['gap'].shift(1)
df['gap_lag3'] = df['gap'].shift(3)
df['gap_lag7'] = df['gap'].shift(7)
df['returns'] = df['price_usd'].pct_change()
df['volatility_7d'] = df['returns'].rolling(7).std() * 100

# 删除缺失值
df = df.dropna().reset_index(drop=True)

print(f"特征构造完成，最终行数: {len(df)}")

# 特征和目标
feature_cols = ['gap_lag1', 'gap_lag3', 'gap_lag7', 'volatility_7d']
X = df[feature_cols]
y = df['gap_target']

# 划分
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 训练
model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
mae = mean_absolute_error(y_test, y_pred)
baseline_mae = abs(y_test - y_test.mean()).mean()

print(f"\n=== MPL 模型结果 ===")
print(f"测试集 MAE: {mae:.6f} 个百分点")
print(f"基准 MAE: {baseline_mae:.6f} 个百分点")
print(f"改善: {(baseline_mae - mae) / baseline_mae * 100:.2f}%")

# 特征相关性
print(f"\n特征与 gap_target 相关系数：")
for col in feature_cols:
    corr = df[col].corr(df['gap_target'])
    print(f"{col}: {corr:.4f}")


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

# 读取 MPL 数据
df = pd.read_csv('C:/Users/91290/Downloads/mpl-usd-max.csv')
df = df[['snapped_at', 'price']].copy()
df.columns = ['date', 'price_usd']
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
df = df[df['date'] >= '2024-01-01'].copy()

# 构造 NAV（30日移动平均）
df['nav_usd'] = df['price_usd'].rolling(30, min_periods=1).mean()
df['gap'] = (df['price_usd'] - df['nav_usd']) / df['nav_usd'] * 100

# 构造特征
df['gap_target'] = df['gap'].shift(-1)
df['gap_lag1'] = df['gap'].shift(1)
df['gap_lag3'] = df['gap'].shift(3)
df['gap_lag7'] = df['gap'].shift(7)
df['returns'] = df['price_usd'].pct_change()
df['volatility_7d'] = df['returns'].rolling(7).std() * 100

# 删除缺失值
df = df.dropna().reset_index(drop=True)
print(f"总行数: {len(df)}")

# 特征和目标
feature_cols = ['gap_lag1', 'gap_lag3', 'gap_lag7', 'volatility_7d']
X = df[feature_cols]
y = df['gap_target']

# 划分
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 1. 线性回归
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)

# 2. 随机森林
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# 3. LightGBM（对比）
lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)

# 基准
baseline_mae = abs(y_test - y_test.mean()).mean()

# 打印结果
print("\n=== MPL 模型对比 ===")
print(f"基准 MAE（猜均值）: {baseline_mae:.4f}")
print(f"线性回归 MAE: {mae_lr:.4f}")
print(f"随机森林 MAE: {mae_rf:.4f} ")
print(f"LightGBM MAE: {mae_lgb:.4f} ")

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
    # OUSG
    ousg = pd.read_csv('C:/Users/91290/Downloads/ousg_merged.csv')
    ousg['date'] = pd.to_datetime(ousg['date'])
    ousg = ousg[ousg['date'] >= '2024-01-01']
    ousg['asset'] = 'OUSG'
    ousg['asset_en'] = 'OUSG (Treasury)'
    
    # MPL
    mpl = pd.read_csv('C:/Users/91290/Downloads/mpl-usd-max.csv')
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
