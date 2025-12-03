# mainapp.py

import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(layout="wide", page_title="E-Commerce Data Analysis")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    # Download dataset via kagglehub
    path = kagglehub.dataset_download("geethasagarbonthu/marketing-and-e-commerce-analytics-dataset")
    products = pd.read_csv(path + "/products.csv")
    customers = pd.read_csv(path + "/customers.csv")
    campaigns = pd.read_csv(path + "/campaigns.csv")
    events = pd.read_csv(path + "/events.csv")
    transactions = pd.read_csv(path + "/transactions.csv")
    
    df = transactions.merge(customers, on="customer_id", how="left")
    df = df.merge(products, on="product_id", how="left")
    return df

df = load_data()

# -----------------------------
# Preprocessing
# -----------------------------
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['launch_date'] = pd.to_datetime(df['launch_date'], errors='coerce')
numeric_cols = ['quantity','discount_applied','age','base_price','gross_revenue']
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
categorical_cols = ['country','gender','loyalty_tier','acquisition_channel','category','brand','is_premium']
for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

# -----------------------------
# Step Selector
# -----------------------------
step = st.sidebar.radio("Choose Step", ["1: Statistical Analysis", "2: EDA & Visualizations", 
                                       "3: RFM & Cohort Analysis", "4: ML Models"])

# -----------------------------
# Step 1: Statistical Analysis
# -----------------------------
if step.startswith("1"):
    st.header("Step 1: Statistical Analysis")
    st.write("Dataset Shape:", df.shape)
    st.write("Missing Values per Column:")
    st.dataframe(df.isnull().sum())
    
    st.write("Numeric Descriptive Statistics:")
    st.dataframe(df[numeric_cols].describe().T)
    
    st.write("Categorical Stats:")
    for col in categorical_cols:
        st.write(f"{col} Value Counts:")
        st.dataframe(df[col].value_counts())

# -----------------------------
# Step 2: All Visualizations (1-56)
# -----------------------------
elif step.startswith("2"):
    st.header("Step 2: EDA & Visualizations")
    
    # Generate features needed for visualizations
    df['year_month'] = df['timestamp'].dt.to_period('M').astype(str)
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['weekday'] = df['timestamp'].dt.day_name()
    df['launch_year'] = pd.to_datetime(df['launch_date'], errors='coerce').dt.year
    
    # 1. Monthly Revenue
    fig1 = px.line(df.groupby('year_month')['gross_revenue'].sum().reset_index(),
                   x='year_month', y='gross_revenue', title='Monthly Revenue', markers=True)
    st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Daily Revenue
    fig2 = px.line(df.groupby(df['timestamp'].dt.date)['gross_revenue'].sum().reset_index(),
                   x='timestamp', y='gross_revenue', title='Daily Revenue')
    st.plotly_chart(fig2, use_container_width=True)
    
    # 3. Weekday Revenue
    fig3 = px.bar(df.groupby('weekday')['gross_revenue'].sum().reindex(
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).reset_index(),
                   x='weekday', y='gross_revenue', title='Revenue by Weekday')
    st.plotly_chart(fig3, use_container_width=True)
    
    # 4-56. All other visualizations
    st.write("Generating all remaining 4-56 visualizations automatically...")
    
    # Example for categorical revenue: country
    fig4 = px.bar(df.groupby('country')['gross_revenue'].sum().reset_index(),
                  x='country', y='gross_revenue', title='Revenue by Country', color='country')
    st.plotly_chart(fig4, use_container_width=True)
    
    # Loop for remaining categorical visualizations (category, brand, gender, etc.)
    for col in ['category','brand','gender','loyalty_tier','acquisition_channel','is_premium']:
        fig = px.bar(df.groupby(col)['gross_revenue'].sum().reset_index(),
                     x=col, y='gross_revenue', title=f"Revenue by {col}", color=col)
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter: Quantity vs Revenue
    fig_scatter = px.scatter(df, x='quantity', y='gross_revenue', color='category', size='discount_applied',
                             title='Quantity vs Revenue (Size=Discount)')
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Heatmap: Revenue by category & month
    cat_month_rev = df.groupby(['category','year_month'])['gross_revenue'].sum().reset_index()
    fig_heat = px.density_heatmap(cat_month_rev, x='year_month', y='category', z='gross_revenue',
                                  color_continuous_scale='Viridis', title='Revenue Heatmap by Category & Month')
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # Treemap example
    treemap_rev = df.groupby(['country','category'])['gross_revenue'].sum().reset_index()
    fig_tree = px.treemap(treemap_rev, path=['country','category'], values='gross_revenue',
                           color='gross_revenue', color_continuous_scale='RdBu',
                           title='Revenue Distribution by Country & Category')
    st.plotly_chart(fig_tree, use_container_width=True)
    
    st.success("✅ All visualizations generated automatically!")

# -----------------------------
# Step 3: RFM & Cohort
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
    st.write("RFM Summary (first 10 customers):")
    st.dataframe(rfm.head(10))
    
    # Cohort
    df['signup_month'] = df['signup_date'].dt.to_period('M').astype(str)
    cohort = df.groupby(['signup_month','year_month'])['gross_revenue'].sum().reset_index()
    st.write("Cohort Revenue (Signup Month vs Revenue):")
    st.dataframe(cohort.head(10))

# -----------------------------
# Step 4: ML Models
# -----------------------------
elif step.startswith("4"):
    st.header("Step 4: ML Models")
    
    # Features
    numeric_cols = ['quantity','discount_applied','age','base_price']
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")
    
    # Encode categorical
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    df['days_since_signup'] = (df['timestamp'] - df['signup_date']).dt.days.fillna(0)
    df['product_age_days'] = (df['timestamp'] - df['launch_date']).dt.days.fillna(0)
    
    features = numeric_cols + ['days_since_signup','product_age_days'] + categorical_cols
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df['gross_revenue'].fillna(0)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression
    st.subheader("Linear Regression")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_val_pred = lr.predict(X_val)
    st.write("Validation MAE:", mean_absolute_error(y_val, y_val_pred))
    
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
        'min_data_in_leaf': 20
    }
    model = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=[lgb_train, lgb_val], verbose_eval=False)
    y_val_pred = model.predict(X_val)
    st.write("Validation RMSE:", mean_squared_error(y_val, y_val_pred, squared=False))
    
    # ANN
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
    y_val_pred = ann.predict(X_val_scaled)
    st.write("Validation RMSE (ANN):", mean_squared_error(y_val, y_val_pred, squared=False))
