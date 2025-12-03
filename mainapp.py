# app.py
import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

st.set_page_config(page_title="Comprehensive E-Commerce Analysis", layout="wide")

# -----------------------------
# Load Dataset
# -----------------------------
st.title("📊 E-Commerce Comprehensive Analysis App")

with st.spinner("Downloading & Loading Dataset..."):
    path = kagglehub.dataset_download("geethasagarbonthu/marketing-and-e-commerce-analytics-dataset")
    
products = pd.read_csv(os.path.join(path, "products.csv"))
customers = pd.read_csv(os.path.join(path, "customers.csv"))
campaigns = pd.read_csv(os.path.join(path, "campaigns.csv"))
events = pd.read_csv(os.path.join(path, "events.csv"))
transactions = pd.read_csv(os.path.join(path, "transactions.csv"))

df = transactions.merge(customers, on="customer_id", how="left")
df = df.merge(products, on="product_id", how="left")

# Ensure datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['launch_date'] = pd.to_datetime(df['launch_date'], errors='coerce')

# Handle missing values
numeric_cols = ['quantity','discount_applied','age','base_price','gross_revenue']
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

categorical_cols = ['country','gender','loyalty_tier','acquisition_channel','category','brand','is_premium']
for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

# -----------------------------
# Step 1: Statistical Analysis
# -----------------------------
if st.button("Step 1: Show Statistical Analysis"):
    st.subheader("📈 Statistical Analysis")
    st.write("**Dataset Shape:**", df.shape)
    st.write("**Columns:**", df.columns.tolist())
    st.write("**Missing Values:**")
    st.dataframe(df.isnull().sum())
    
    st.write("### Numerical Descriptive Statistics")
    num_stats = df[numeric_cols].describe().T
    num_stats['skew'] = df[numeric_cols].skew()
    num_stats['kurtosis'] = df[numeric_cols].kurtosis()
    st.dataframe(num_stats)
    
    st.write("### Categorical Stats")
    for col in categorical_cols:
        st.write(f"**{col} Value Counts**")
        st.dataframe(df[col].value_counts())
    
    st.write("### Correlation Matrix")
    st.dataframe(df[numeric_cols].corr().round(2))
    
    st.write("### Revenue Summary")
    st.write(df['gross_revenue'].describe())
    
    st.write("### Top Customers & Products")
    top_customers = df.groupby('customer_id')['gross_revenue'].sum().sort_values(ascending=False).head(10)
    st.dataframe(top_customers)
    top_products = df.groupby('product_id')['gross_revenue'].sum().sort_values(ascending=False).head(10)
    st.dataframe(top_products)

# -----------------------------
# Step 2: Basic Visualizations (1-30)
# -----------------------------
if st.button("Step 2: Basic Visualizations"):
    st.subheader("📊 Basic Visualizations (1-30)")
    
    df['year_month'] = df['timestamp'].dt.to_period('M').astype(str)
    df['weekday'] = df['timestamp'].dt.day_name()
    df['launch_year'] = pd.to_datetime(df['launch_date'], errors='coerce').dt.year
    
    # Monthly Revenue
    fig1 = px.line(df.groupby('year_month')['gross_revenue'].sum().reset_index(),
                   x='year_month', y='gross_revenue', title='Monthly Revenue', markers=True)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Daily Revenue
    fig2 = px.line(df.groupby(df['timestamp'].dt.date)['gross_revenue'].sum().reset_index(),
                   x='timestamp', y='gross_revenue', title='Daily Revenue')
    st.plotly_chart(fig2, use_container_width=True)
    
    # Revenue by Weekday
    fig3 = px.bar(df.groupby('weekday')['gross_revenue'].sum().reindex(
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).reset_index(),
                   x='weekday', y='gross_revenue', title='Revenue by Weekday')
    st.plotly_chart(fig3, use_container_width=True)
    
    # Revenue by country
    fig4 = px.bar(df.groupby('country')['gross_revenue'].sum().reset_index(),
                  x='country', y='gross_revenue', title='Revenue by Country', color='country')
    st.plotly_chart(fig4, use_container_width=True)
    
    # Revenue by age
    fig5 = px.bar(df.groupby('age')['gross_revenue'].sum().reset_index(),
                  x='age', y='gross_revenue', title='Revenue by Age')
    st.plotly_chart(fig5, use_container_width=True)
    
    # Revenue by gender
    fig6 = px.bar(df.groupby('gender')['gross_revenue'].sum().reset_index(),
                  x='gender', y='gross_revenue', title='Revenue by Gender', color='gender')
    st.plotly_chart(fig6, use_container_width=True)
    
    # Revenue by loyalty tier
    fig7 = px.bar(df.groupby('loyalty_tier')['gross_revenue'].sum().reset_index(),
                  x='loyalty_tier', y='gross_revenue', title='Revenue by Loyalty Tier', color='loyalty_tier')
    st.plotly_chart(fig7, use_container_width=True)
    
    # Revenue by acquisition channel
    fig8 = px.bar(df.groupby('acquisition_channel')['gross_revenue'].sum().reset_index(),
                  x='acquisition_channel', y='gross_revenue', title='Revenue by Acquisition Channel', color='acquisition_channel')
    st.plotly_chart(fig8, use_container_width=True)
    
    # Revenue by product category
    fig9 = px.bar(df.groupby('category')['gross_revenue'].sum().reset_index(),
                  x='category', y='gross_revenue', title='Revenue by Category', color='category')
    st.plotly_chart(fig9, use_container_width=True)
    
    # Revenue by brand
    fig10 = px.bar(df.groupby('brand')['gross_revenue'].sum().reset_index(),
                   x='brand', y='gross_revenue', title='Revenue by Brand', color='brand')
    st.plotly_chart(fig10, use_container_width=True)

# -----------------------------
# Step 3: Advanced Visualizations (31-56)
# -----------------------------
if st.button("Step 3: Advanced Visualizations"):
    st.subheader("📊 Advanced Visualizations (31-56)")
    
    # Example: Revenue by Age Group
    bins = [18,25,35,45,55,65,100]
    labels = ['18-24','25-34','35-44','45-54','55-64','65+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    fig33 = px.bar(df.groupby('age_group')['gross_revenue'].sum().reset_index(),
                   x='age_group', y='gross_revenue', title='Revenue by Age Group')
    st.plotly_chart(fig33, use_container_width=True)
    
    # Heatmap: Revenue by Category & Month
    cat_month_rev = df.groupby(['category','year_month'])['gross_revenue'].sum().reset_index()
    fig45 = px.density_heatmap(cat_month_rev, x='year_month', y='category', z='gross_revenue',
                               color_continuous_scale='Viridis', title='Revenue Heatmap by Category & Month')
    st.plotly_chart(fig45, use_container_width=True)
    
    # Treemap: Revenue by Country & Category
    treemap_rev = df.groupby(['country','category'])['gross_revenue'].sum().reset_index()
    fig47 = px.treemap(treemap_rev, path=['country','category'], values='gross_revenue',
                       title='Revenue Distribution by Country & Category', color='gross_revenue', color_continuous_scale='RdBu')
    st.plotly_chart(fig47, use_container_width=True)
    
    # Scatter Matrix
    fig49 = px.scatter_matrix(df, dimensions=['quantity','gross_revenue','discount_applied','age'],
                              color='category', title='Scatter Matrix: Revenue, Quantity, Discount, Age')
    st.plotly_chart(fig49, use_container_width=True)
    
    # Time Series Decomposition (Optional)
    ts = df.groupby('timestamp')['gross_revenue'].sum().asfreq('D').fillna(0)
    result = seasonal_decompose(ts, model='additive', period=30)
    fig56 = go.Figure()
    fig56.add_trace(go.Scatter(y=result.trend, name='Trend'))
    fig56.add_trace(go.Scatter(y=result.seasonal, name='Seasonality'))
    fig56.add_trace(go.Scatter(y=result.resid, name='Residual'))
    fig56.update_layout(title='Revenue Time Series Decomposition')
    st.plotly_chart(fig56, use_container_width=True)

# -----------------------------
# Step 4: ML Models
# -----------------------------
if st.button("Step 4: ML Models"):
    st.subheader("🤖 ML Models to Predict Gross Revenue")
    
    features = ['quantity','discount_applied','age','base_price'] + categorical_cols
    X = df[features].copy()
    y = df['gross_revenue']
    
    # Encode categorical
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ----------------- Linear Regression -----------------
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_train_pred = lr.predict(X_train)
    y_val_pred = lr.predict(X_val)
    st.write("### Linear Regression Metrics")
    st.write("Train RMSE:", round(mean_squared_error(y_train, y_train_pred, squared=False),2))
    st.write("Validation RMSE:", round(mean_squared_error(y_val, y_val_pred, squared=False),2))
    st.write("R2 Score:", round(r2_score(y_val, y_val_pred),2))
    
    # ----------------- LightGBM -----------------
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    params = {
        'objective':'regression',
        'metric':'rmse',
        'boosting_type':'gbdt',
        'learning_rate':0.05,
        'num_leaves':31
    }
    model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_val], num_boost_round=500, verbose_eval=False)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    st.write("### LightGBM Metrics")
    st.write("Train RMSE:", round(mean_squared_error(y_train, y_train_pred, squared=False),2))
    st.write("Validation RMSE:", round(mean_squared_error(y_val, y_val_pred, squared=False),2))
    st.write("R2 Score:", round(r2_score(y_val, y_val_pred),2))
    
    # ----------------- ANN -----------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    ann = Sequential()
    ann.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
    ann.add(Dropout(0.2))
    ann.add(Dense(32, activation='relu'))
    ann.add(Dense(1))
    ann.compile(optimizer='adam', loss='mse')
    ann.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
    
    y_train_pred = ann.predict(X_train_scaled).flatten()
    y_val_pred = ann.predict(X_val_scaled).flatten()
    st.write("### ANN Metrics")
    st.write("Train RMSE:", round(mean_squared_error(y_train, y_train_pred, squared=False),2))
    st.write("Validation RMSE:", round(mean_squared_error(y_val, y_val_pred, squared=False),2))
    st.write("R2 Score:", round(r2_score(y_val, y_val_pred),2))
