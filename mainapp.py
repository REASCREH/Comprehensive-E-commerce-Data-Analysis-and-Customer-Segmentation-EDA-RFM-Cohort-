# app.py
import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------
# Step 0: Load Data
# -----------------------------
@st.cache_data
def load_data():
    st.info("Downloading dataset from Kagglehub...")
    dataset_path = kagglehub.dataset_download("geethasagarbonthu/marketing-and-e-commerce-analytics-dataset")
    
    products = pd.read_csv(os.path.join(dataset_path, "products.csv"))
    customers = pd.read_csv(os.path.join(dataset_path, "customers.csv"))
    campaigns = pd.read_csv(os.path.join(dataset_path, "campaigns.csv"))
    events = pd.read_csv(os.path.join(dataset_path, "events.csv"))
    transactions = pd.read_csv(os.path.join(dataset_path, "transactions.csv"))
    
    # Merge datasets
    df = transactions.merge(customers, on="customer_id", how="left")
    df = df.merge(products, on="product_id", how="left")
    
    # Convert datetime columns
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['launch_date'] = pd.to_datetime(df['launch_date'], errors='coerce')
    
    return df

df = load_data()
st.title("📊 Comprehensive E-Commerce Data Analysis & ML App")

# -----------------------------
# Sidebar Steps
# -----------------------------
st.sidebar.title("Steps")
step = st.sidebar.radio("Select Step:", ["1️⃣ Statistical Analysis", 
                                         "2️⃣ Visualizations", 
                                         "3️⃣ RFM & Cohort Analysis", 
                                         "4️⃣ ML Models"])

# -----------------------------
# Step 1: Statistical Analysis
# -----------------------------
if step.startswith("1"):
    st.header("Step 1: Statistical Analysis")
    
    st.subheader("Basic Info")
    st.write(f"Dataset shape: {df.shape}")
    st.write(df.head(5))
    st.write(df.describe(include='all').T)
    
    st.subheader("Missing Values")
    st.write(df.isnull().sum())
    
    st.subheader("Numeric Stats with Skew & Kurtosis")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_stats = df[numeric_cols].describe().T
    num_stats['skew'] = df[numeric_cols].skew()
    num_stats['kurtosis'] = df[numeric_cols].kurtosis()
    st.write(num_stats)
    
    st.subheader("Categorical Value Counts")
    categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    for col in categorical_cols:
        st.write(f"Column: {col}")
        st.write(df[col].value_counts())
        st.write(df[col].value_counts(normalize=True).round(2))

# -----------------------------
# Step 2: Visualizations
# -----------------------------
elif step.startswith("2"):
    st.header("Step 2: Visualizations")
    
    df['year_month'] = df['timestamp'].dt.to_period('M').astype(str)
    df['weekday'] = df['timestamp'].dt.day_name()
    df['launch_year'] = df['launch_date'].dt.year
    df['age_group'] = pd.cut(df['age'], bins=[18,25,35,45,55,65,100],
                             labels=['18-24','25-34','35-44','45-54','55-64','65+'])
    
    st.subheader("Revenue Trends")
    fig1 = px.line(df.groupby('year_month')['gross_revenue'].sum().reset_index(),
                   x='year_month', y='gross_revenue', title='Monthly Revenue', markers=True)
    st.plotly_chart(fig1)
    
    fig2 = px.bar(df.groupby('country')['gross_revenue'].sum().reset_index(),
                  x='country', y='gross_revenue', title='Revenue by Country', color='country')
    st.plotly_chart(fig2)
    
    fig3 = px.bar(df.groupby('category')['gross_revenue'].sum().reset_index(),
                  x='category', y='gross_revenue', title='Revenue by Category', color='category')
    st.plotly_chart(fig3)
    
    fig4 = px.bar(df.groupby('brand')['gross_revenue'].sum().reset_index(),
                  x='brand', y='gross_revenue', title='Revenue by Brand', color='brand')
    st.plotly_chart(fig4)
    
    fig5 = px.bar(df.groupby('gender')['gross_revenue'].sum().reset_index(),
                  x='gender', y='gross_revenue', title='Revenue by Gender', color='gender')
    st.plotly_chart(fig5)
    
    fig6 = px.bar(df.groupby('age_group')['gross_revenue'].sum().reset_index(),
                  x='age_group', y='gross_revenue', title='Revenue by Age Group')
    st.plotly_chart(fig6)
    
    # Add more visualizations similarly
    st.write("✅ Additional 50+ visualizations can be added here following the same method using plotly.")

# -----------------------------
# Step 3: RFM & Cohort Analysis
# -----------------------------
elif step.startswith("3"):
    st.header("Step 3: RFM & Cohort Analysis")
    today = df['timestamp'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('customer_id').agg({
        'timestamp': lambda x: (today - x.max()).days,
        'transaction_id': 'count',
        'gross_revenue': 'sum'
    }).rename(columns={'timestamp':'Recency','transaction_id':'Frequency','gross_revenue':'Monetary'})
    rfm['CLV'] = rfm['Monetary']
    
    st.subheader("RFM Summary (first 10 customers)")
    st.write(rfm.head(10))
    
    df['signup_month'] = df['signup_date'].dt.to_period('M').astype(str)
    cohort = df.groupby(['signup_month','year_month'])['gross_revenue'].sum().reset_index()
    
    st.subheader("Cohort Revenue Analysis")
    st.write(cohort.head(10))
    
# -----------------------------
# Step 4: ML Models
# -----------------------------
elif step.startswith("4"):
    st.header("Step 4: ML Models")
    
    st.subheader("Feature Engineering")
    numeric_cols = ['quantity','discount_applied','age','base_price']
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = ['country','gender','loyalty_tier','acquisition_channel','category','brand','is_premium']
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    df['days_since_signup'] = (df['timestamp'] - df['signup_date']).dt.days
    df['product_age_days'] = (df['timestamp'] - df['launch_date']).dt.days.fillna(0)
    
    # Features & target
    features = numeric_cols + ['days_since_signup','product_age_days'] + categorical_cols
    X = df[features]
    y = df['gross_revenue']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # -----------------------------
    # Linear Regression
    st.subheader("Linear Regression")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_train_pred = lr.predict(X_train)
    y_val_pred = lr.predict(X_val)
    
    st.write("Train MAE:", mean_absolute_error(y_train, y_train_pred))
    st.write("Validation MAE:", mean_absolute_error(y_val, y_val_pred))
    
    # -----------------------------
    # LightGBM
    st.subheader("LightGBM")
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 131,
        'max_depth': -1,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    model = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=[lgb_train, lgb_val], verbose_eval=False)
    
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    st.write("Train RMSE:", mean_squared_error(y_train, y_train_pred, squared=False))
    st.write("Validation RMSE:", mean_squared_error(y_val, y_val_pred, squared=False))
    
    # -----------------------------
    # ANN Model
    st.subheader("ANN Model")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    ann = Sequential()
    ann.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
    ann.add(Dense(32, activation='relu'))
    ann.add(Dense(1, activation='linear'))
    
    ann.compile(optimizer='adam', loss='mse')
    ann.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), epochs=50, batch_size=32, verbose=0)
    
    y_train_pred = ann.predict(X_train_scaled)
    y_val_pred = ann.predict(X_val_scaled)
    
    st.write("Train RMSE:", mean_squared_error(y_train, y_train_pred, squared=False))
    st.write("Validation RMSE:", mean_squared_error(y_val, y_val_pred, squared=False))
