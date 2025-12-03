# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------
# Load datasets
# -----------------------------
st.title("Comprehensive E-Commerce Data Analysis & ML App")

path = "/kaggle/input/marketing-and-e-commerce-analytics-dataset/"

@st.cache_data
def load_data():
    products = pd.read_csv(path + "products.csv")
    customers = pd.read_csv(path + "customers.csv")
    campaigns = pd.read_csv(path + "campaigns.csv")
    events = pd.read_csv(path + "events.csv")
    transactions = pd.read_csv(path + "transactions.csv")
    df = transactions.merge(customers, on="customer_id", how="left")
    df = df.merge(products, on="product_id", how="left")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['launch_date'] = pd.to_datetime(df['launch_date'], errors='coerce')
    return df

df = load_data()

# -----------------------------
# Step 1: Statistical Analysis
# -----------------------------
if st.button("Step 1: Statistical Analysis"):
    st.subheader("✅ Statistical Analysis")

    st.write("**Dataset Shape:**", df.shape)
    st.write("**Columns:**", df.columns.tolist())
    st.write("**Missing Values:**")
    st.dataframe(df.isnull().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_stats = df[numeric_cols].describe().T
    num_stats['skew'] = df[numeric_cols].skew()
    num_stats['kurtosis'] = df[numeric_cols].kurtosis()
    st.write("**Numerical Descriptive Stats:**")
    st.dataframe(num_stats)

    categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    st.write("**Categorical Value Counts:**")
    for col in categorical_cols:
        st.write(f"**{col}**")
        st.dataframe(df[col].value_counts())

    st.write("**Revenue Summary:**")
    st.write(df['gross_revenue'].describe())
    st.write("Revenue by key columns:")
    for col in ['country','gender','loyalty_tier','acquisition_channel','category','brand','is_premium']:
        st.write(f"Revenue by {col}")
        st.dataframe(df.groupby(col)['gross_revenue'].agg(['sum','mean','median','std','count']).sort_values('sum', ascending=False))

# -----------------------------
# Step 2: Basic Visualizations (1-30)
# -----------------------------
if st.button("Step 2: Basic Visualizations (1–30)"):
    st.subheader("📊 Basic Visualizations")

    # Monthly Revenue
    fig1 = px.line(df.groupby(df['timestamp'].dt.to_period('M'))['gross_revenue'].sum().reset_index(),
                   x='timestamp', y='gross_revenue', title='Monthly Revenue', markers=True)
    st.plotly_chart(fig1)

    # Daily Revenue
    fig2 = px.line(df.groupby(df['timestamp'].dt.date)['gross_revenue'].sum().reset_index(),
                   x='timestamp', y='gross_revenue', title='Daily Revenue')
    st.plotly_chart(fig2)

    # Revenue by Weekday
    df['weekday'] = df['timestamp'].dt.day_name()
    fig3 = px.bar(df.groupby('weekday')['gross_revenue'].sum().reindex(
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).reset_index(),
                  x='weekday', y='gross_revenue', title='Revenue by Weekday')
    st.plotly_chart(fig3)

    # Revenue by Country
    fig4 = px.bar(df.groupby('country')['gross_revenue'].sum().reset_index(),
                  x='country', y='gross_revenue', color='country', title='Revenue by Country')
    st.plotly_chart(fig4)

    # Revenue by Age
    fig5 = px.bar(df.groupby('age')['gross_revenue'].sum().reset_index(),
                  x='age', y='gross_revenue', title='Revenue by Age')
    st.plotly_chart(fig5)

    # Revenue by Gender
    fig6 = px.bar(df.groupby('gender')['gross_revenue'].sum().reset_index(),
                  x='gender', y='gross_revenue', color='gender', title='Revenue by Gender')
    st.plotly_chart(fig6)

    # Revenue by Loyalty Tier
    fig7 = px.bar(df.groupby('loyalty_tier')['gross_revenue'].sum().reset_index(),
                  x='loyalty_tier', y='gross_revenue', color='loyalty_tier', title='Revenue by Loyalty Tier')
    st.plotly_chart(fig7)

    # Revenue by Acquisition Channel
    fig8 = px.bar(df.groupby('acquisition_channel')['gross_revenue'].sum().reset_index(),
                  x='acquisition_channel', y='gross_revenue', color='acquisition_channel', title='Revenue by Acquisition Channel')
    st.plotly_chart(fig8)

    # Revenue by Category
    fig9 = px.bar(df.groupby('category')['gross_revenue'].sum().reset_index(),
                  x='category', y='gross_revenue', color='category', title='Revenue by Category')
    st.plotly_chart(fig9)

    # Revenue by Brand
    fig10 = px.bar(df.groupby('brand')['gross_revenue'].sum().reset_index(),
                   x='brand', y='gross_revenue', color='brand', title='Revenue by Brand')
    st.plotly_chart(fig10)

# -----------------------------
# Step 3: Advanced Visualizations (31-56 + more)
# -----------------------------
if st.button("Step 3: Advanced Visualizations (31–56+)"):
    st.subheader("📊 Advanced Visualizations")

    df['year_month'] = df['timestamp'].dt.to_period('M').astype(str)
    df['launch_year'] = df['launch_date'].dt.year
    df['age_group'] = pd.cut(df['age'], bins=[18,25,35,45,55,65,100], labels=['18-24','25-34','35-44','45-54','55-64','65+'])

    # Revenue by Country & Gender
    fig31 = px.bar(df.groupby(['country','gender'])['gross_revenue'].sum().reset_index(),
                   x='country', y='gross_revenue', color='gender', barmode='group',
                   title='Revenue by Country & Gender')
    st.plotly_chart(fig31)

    # Revenue by Category & Gender
    fig32 = px.bar(df.groupby(['category','gender'])['gross_revenue'].sum().reset_index(),
                   x='category', y='gross_revenue', color='gender', barmode='group',
                   title='Revenue by Category & Gender')
    st.plotly_chart(fig32)

    # Revenue by Age Group
    fig33 = px.bar(df.groupby('age_group')['gross_revenue'].sum().reset_index(),
                   x='age_group', y='gross_revenue', title='Revenue by Age Group')
    st.plotly_chart(fig33)

    # Revenue by Weekday & Category
    fig34 = px.bar(df.groupby(['weekday','category'])['gross_revenue'].sum().reset_index(),
                   x='weekday', y='gross_revenue', color='category', barmode='group',
                   title='Revenue by Weekday & Category')
    st.plotly_chart(fig34)

    # Quantity Sold by Country & Category
    fig35 = px.bar(df.groupby(['country','category'])['quantity'].sum().reset_index(),
                   x='category', y='quantity', color='country', barmode='group',
                   title='Quantity Sold by Country & Category')
    st.plotly_chart(fig35)

    # Heatmap: Revenue by Category & Month
    cat_month_rev = df.groupby(['category','year_month'])['gross_revenue'].sum().reset_index()
    fig45 = px.density_heatmap(cat_month_rev, x='year_month', y='category', z='gross_revenue',
                               color_continuous_scale='Viridis', title='Revenue Heatmap by Category & Month')
    st.plotly_chart(fig45)

    # Correlation Heatmap
    numeric_cols = ['quantity','discount_applied','gross_revenue','age','base_price']
    corr_matrix = df[numeric_cols].corr().round(2)
    fig55 = ff.create_annotated_heatmap(z=corr_matrix.values, x=list(corr_matrix.columns),
                                        y=list(corr_matrix.index), colorscale='Viridis', showscale=True)
    st.plotly_chart(fig55)

# -----------------------------
# Step 4: ML Models
# -----------------------------
if st.button("Step 4: ML Models"):
    st.subheader("🤖 ML Models: Predict Gross Revenue")

    # Handle missing & encode
    numeric_cols = ['quantity','discount_applied','age','base_price']
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    categorical_cols = ['country','gender','loyalty_tier','acquisition_channel','category','brand','is_premium']
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Feature engineering
    df['product_age_days'] = (df['timestamp'] - df['launch_date']).dt.days.fillna(0)
    df['days_since_signup'] = (df['timestamp'] - df['signup_date']).dt.days.fillna(0)
    features = numeric_cols + ['product_age_days','days_since_signup'] + categorical_cols
    X = df[features]
    y = df['gross_revenue']

    # Train/Validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    st.write("### 1️⃣ Linear Regression")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_train_pred = lr.predict(X_train)
    y_val_pred = lr.predict(X_val)
    st.write("Train R²:", r2_score(y_train, y_train_pred))
    st.write("Validation R²:", r2_score(y_val, y_val_pred))

    st.write("### 2️⃣ LightGBM")
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    params = {'objective':'regression','metric':'rmse','boosting_type':'gbdt','learning_rate':0.05,'num_leaves':31}
    model = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=[lgb_train,lgb_val], verbose_eval=False)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    st.write("Train R²:", r2_score(y_train, y_train_pred))
    st.write("Validation R²:", r2_score(y_val, y_val_pred))

    st.write("### 3️⃣ ANN Model")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    ann = Sequential()
    ann.add(Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]))
    ann.add(Dense(32, activation='relu'))
    ann.add(Dense(1))
    ann.compile(optimizer='adam', loss='mse')
    ann.fit(X_train_scaled, y_train, epochs=20, batch_size=64, verbose=0)
    y_train_pred = ann.predict(X_train_scaled)
    y_val_pred = ann.predict(X_val_scaled)
    st.write("Train R²:", r2_score(y_train, y_train_pred))
    st.write("Validation R²:", r2_score(y_val, y_val_pred))
