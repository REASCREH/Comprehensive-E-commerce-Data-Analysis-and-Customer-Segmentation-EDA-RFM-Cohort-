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

# Clean the data
def clean_data(df):
    """Clean the dataframe by handling missing values and infinite values"""
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values in numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col in df_clean.columns:
            # Replace infinite values with NaN first
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
            # Fill NaN with median
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Handle missing values in categorical columns
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('Unknown')
    
    # Handle datetime columns
    if 'launch_date' in df_clean.columns:
        df_clean['launch_date'] = df_clean['launch_date'].fillna(df_clean['timestamp'].min())
    
    # Remove any rows where gross_revenue is extremely negative (likely data errors)
    if 'gross_revenue' in df_clean.columns:
        df_clean = df_clean[df_clean['gross_revenue'] > -1000]  # Remove extreme negative values
    
    return df_clean

# Load and clean data
df = load_data()
df_clean = clean_data(df)

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
    st.write(f"Dataset shape: {df_clean.shape}")
    st.write(df_clean.head(5))
    st.write(df_clean.describe(include='all').T)
    
    st.subheader("Missing Values")
    missing_values = df_clean.isnull().sum()
    st.write(missing_values[missing_values > 0])
    if missing_values.sum() == 0:
        st.success("✅ No missing values found!")
    
    st.subheader("Infinite Values Check")
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    inf_counts = {}
    for col in numeric_cols:
        if col in df_clean.columns:
            inf_count = np.isinf(df_clean[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
    if inf_counts:
        st.warning(f"Infinite values found: {inf_counts}")
    else:
        st.success("✅ No infinite values found!")
    
    st.subheader("Numeric Stats with Skew & Kurtosis")
    num_stats = df_clean[numeric_cols].describe().T
    num_stats['skew'] = df_clean[numeric_cols].skew()
    num_stats['kurtosis'] = df_clean[numeric_cols].kurtosis()
    st.write(num_stats)
    
    st.subheader("Categorical Value Counts")
    categorical_cols = df_clean.select_dtypes(include=['object','category']).columns.tolist()
    for col in categorical_cols[:5]:  # Show first 5 to avoid too much output
        st.write(f"Column: {col}")
        st.write(df_clean[col].value_counts().head(10))
        st.write(df_clean[col].value_counts(normalize=True).round(2).head(10))

# -----------------------------
# Step 2: Visualizations
# -----------------------------
elif step.startswith("2"):
    st.header("Step 2: Visualizations")
    
    # Create derived columns for visualizations
    df_vis = df_clean.copy()
    df_vis['year_month'] = df_vis['timestamp'].dt.to_period('M').astype(str)
    df_vis['weekday'] = df_vis['timestamp'].dt.day_name()
    df_vis['month'] = df_vis['timestamp'].dt.month
    df_vis['year'] = df_vis['timestamp'].dt.year
    df_vis['hour'] = df_vis['timestamp'].dt.hour
    df_vis['launch_year'] = df_vis['launch_date'].dt.year
    df_vis['age_group'] = pd.cut(df_vis['age'], bins=[18,25,35,45,55,65,100],
                                 labels=['18-24','25-34','35-44','45-54','55-64','65+'])
    
    # Count of visualizations
    visualization_count = 0
    
    # Visualizations from the notebook
    st.subheader("📈 Temporal and Revenue Analysis")
    
    # 1. Monthly Revenue Line Chart
    with st.spinner('Creating Monthly Revenue Chart...'):
        monthly_rev = df_vis.groupby('year_month')['gross_revenue'].sum().reset_index()
        fig1 = px.line(monthly_rev, x='year_month', y='gross_revenue', 
                       title='Monthly Revenue Trend', markers=True)
        fig1.update_layout(xaxis_title='Month', yaxis_title='Revenue ($)')
        st.plotly_chart(fig1, use_container_width=True)
        visualization_count += 1
    
    # 2. Daily Revenue Line Chart
    with st.spinner('Creating Daily Revenue Chart...'):
        daily_rev = df_vis.groupby(df_vis['timestamp'].dt.date)['gross_revenue'].sum().reset_index()
        daily_rev.columns = ['date', 'gross_revenue']
        fig2 = px.line(daily_rev, x='date', y='gross_revenue', 
                       title='Daily Revenue Pattern')
        fig2.update_layout(xaxis_title='Date', yaxis_title='Revenue ($)')
        st.plotly_chart(fig2, use_container_width=True)
        visualization_count += 1
    
    # 3. Revenue by Weekday
    with st.spinner('Creating Weekday Revenue Chart...'):
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_rev = df_vis.groupby('weekday')['gross_revenue'].sum().reindex(weekday_order).reset_index()
        fig3 = px.bar(weekday_rev, x='weekday', y='gross_revenue', 
                      title='Revenue by Day of Week', color='weekday')
        fig3.update_layout(xaxis_title='Day of Week', yaxis_title='Revenue ($)')
        st.plotly_chart(fig3, use_container_width=True)
        visualization_count += 1
    
    # 4. Revenue by Month
    with st.spinner('Creating Monthly Distribution Chart...'):
        month_rev = df_vis.groupby('month')['gross_revenue'].sum().reset_index()
        fig4 = px.bar(month_rev, x='month', y='gross_revenue', 
                      title='Revenue by Month', color='month')
        fig4.update_layout(xaxis_title='Month', yaxis_title='Revenue ($)')
        st.plotly_chart(fig4, use_container_width=True)
        visualization_count += 1
    
    st.subheader("🌍 Geographic Analysis")
    
    # 5. Revenue by Country
    with st.spinner('Creating Country Revenue Chart...'):
        country_rev = df_vis.groupby('country')['gross_revenue'].sum().sort_values(ascending=False).reset_index()
        fig5 = px.bar(country_rev, x='country', y='gross_revenue', 
                      title='Total Revenue by Country', color='country')
        fig5.update_layout(xaxis_title='Country', yaxis_title='Revenue ($)')
        st.plotly_chart(fig5, use_container_width=True)
        visualization_count += 1
    
    st.subheader("👥 Customer Analysis")
    
    # 6. Revenue by Age Group
    with st.spinner('Creating Age Group Revenue Chart...'):
        age_rev = df_vis.groupby('age_group')['gross_revenue'].sum().reset_index()
        fig6 = px.bar(age_rev, x='age_group', y='gross_revenue', 
                      title='Revenue by Age Group', color='age_group')
        fig6.update_layout(xaxis_title='Age Group', yaxis_title='Revenue ($)')
        st.plotly_chart(fig6, use_container_width=True)
        visualization_count += 1
    
    # 7. Age Distribution
    with st.spinner('Creating Age Distribution Chart...'):
        fig7 = px.histogram(df_vis, x='age', nbins=30, 
                            title='Customer Age Distribution',
                            labels={'age': 'Age', 'count': 'Number of Customers'})
        fig7.update_layout(bargap=0.1)
        st.plotly_chart(fig7, use_container_width=True)
        visualization_count += 1
    
    # 8. Revenue by Gender
    with st.spinner('Creating Gender Revenue Chart...'):
        gender_rev = df_vis.groupby('gender')['gross_revenue'].sum().reset_index()
        fig8 = px.bar(gender_rev, x='gender', y='gross_revenue', 
                      title='Revenue by Gender', color='gender')
        fig8.update_layout(xaxis_title='Gender', yaxis_title='Revenue ($)')
        st.plotly_chart(fig8, use_container_width=True)
        visualization_count += 1
    
    # 9. Revenue by Loyalty Tier
    with st.spinner('Creating Loyalty Tier Revenue Chart...'):
        loyalty_rev = df_vis.groupby('loyalty_tier')['gross_revenue'].sum().reset_index()
        fig9 = px.bar(loyalty_rev, x='loyalty_tier', y='gross_revenue', 
                       title='Revenue by Loyalty Tier', color='loyalty_tier')
        fig9.update_layout(xaxis_title='Loyalty Tier', yaxis_title='Revenue ($)')
        st.plotly_chart(fig9, use_container_width=True)
        visualization_count += 1
    
    st.subheader("📦 Product Analysis")
    
    # 10. Revenue by Category
    with st.spinner('Creating Category Revenue Chart...'):
        category_rev = df_vis.groupby('category')['gross_revenue'].sum().sort_values(ascending=False).reset_index()
        fig10 = px.bar(category_rev, x='category', y='gross_revenue', 
                       title='Revenue by Product Category', color='category')
        fig10.update_layout(xaxis_title='Category', yaxis_title='Revenue ($)')
        st.plotly_chart(fig10, use_container_width=True)
        visualization_count += 1
    
    # 11. Quantity Sold by Category
    with st.spinner('Creating Quantity by Category Chart...'):
        category_qty = df_vis.groupby('category')['quantity'].sum().sort_values(ascending=False).reset_index()
        fig11 = px.bar(category_qty, x='category', y='quantity', 
                       title='Quantity Sold by Category', color='category')
        fig11.update_layout(xaxis_title='Category', yaxis_title='Quantity Sold')
        st.plotly_chart(fig11, use_container_width=True)
        visualization_count += 1
    
    st.subheader("💰 Discount Analysis")
    
    # 12. Discount vs Gross Revenue Scatter
    with st.spinner('Creating Discount Analysis Chart...'):
        # Sample data for scatter plot to improve performance
        sample_df = df_vis.sample(min(2000, len(df_vis)), random_state=42)
        fig12 = px.scatter(sample_df, x='discount_applied', y='gross_revenue',
                           title='Discount Applied vs Gross Revenue (Sample)',
                           labels={'discount_applied': 'Discount Applied (%)', 'gross_revenue': 'Revenue ($)'},
                           opacity=0.6)
        st.plotly_chart(fig12, use_container_width=True)
        visualization_count += 1
    
    st.subheader("📊 Advanced Analytics")
    
    # 13. Correlation Heatmap
    with st.spinner('Creating Correlation Heatmap...'):
        numeric_cols_corr = ['quantity', 'discount_applied', 'gross_revenue', 'age', 'base_price']
        # Ensure all columns exist and have finite values
        available_cols = [col for col in numeric_cols_corr if col in df_vis.columns]
        corr_df = df_vis[available_cols].copy()
        
        # Remove any remaining infinite values
        corr_df = corr_df.replace([np.inf, -np.inf], np.nan)
        corr_df = corr_df.dropna()
        
        if len(corr_df) > 0:
            corr_matrix = corr_df.corr()
            fig13 = ff.create_annotated_heatmap(
                z=corr_matrix.values,
                x=list(corr_matrix.columns),
                y=list(corr_matrix.index),
                annotation_text=corr_matrix.round(2).values,
                colorscale='RdBu',
                showscale=True
            )
            fig13.update_layout(title='Correlation Matrix Heatmap')
            st.plotly_chart(fig13, use_container_width=True)
            visualization_count += 1
        else:
            st.warning("Not enough data for correlation heatmap after cleaning")
    
    # Show completion message
    st.success(f"✅ Successfully displayed {visualization_count} key visualizations!")

# -----------------------------
# Step 3: RFM & Cohort Analysis
# -----------------------------
elif step.startswith("3"):
    st.header("Step 3: RFM & Cohort Analysis")
    
    # Create necessary time-based columns
    df_rfm = df_clean.copy()
    df_rfm['year_month'] = df_rfm['timestamp'].dt.to_period('M').astype(str)
    df_rfm['signup_month'] = df_rfm['signup_date'].dt.to_period('M').astype(str)
    
    today = df_rfm['timestamp'].max() + pd.Timedelta(days=1)
    
    # RFM Analysis
    with st.spinner('Calculating RFM Scores...'):
        rfm = df_rfm.groupby('customer_id').agg({
            'timestamp': lambda x: (today - x.max()).days,
            'transaction_id': 'count',
            'gross_revenue': 'sum'
        }).rename(columns={'timestamp':'Recency','transaction_id':'Frequency','gross_revenue':'Monetary'})
        
        # Remove any infinite or NaN values
        rfm = rfm.replace([np.inf, -np.inf], np.nan)
        rfm = rfm.dropna()
        
        if len(rfm) > 0:
            # Calculate RFM scores
            rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=['4','3','2','1'], duplicates='drop')
            rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=['1','2','3','4'], duplicates='drop')
            rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=['1','2','3','4'], duplicates='drop')
            
            rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
            
            # Segment customers based on RFM
            def rfm_segment(row):
                if row['RFM_Score'] == '444':
                    return 'Champions'
                elif row['R_Score'] in ['4','3'] and row['F_Score'] in ['4','3']:
                    return 'Loyal Customers'
                elif row['R_Score'] in ['4','3'] and row['M_Score'] in ['4','3']:
                    return 'Potential Loyalists'
                elif row['R_Score'] in ['2','1'] and row['F_Score'] in ['4','3']:
                    return 'At Risk'
                elif row['R_Score'] in ['2','1'] and row['F_Score'] in ['2','1']:
                    return 'Lost Customers'
                else:
                    return 'Others'
            
            rfm['Segment'] = rfm.apply(rfm_segment, axis=1)
            rfm['CLV'] = rfm['Monetary']
    
    st.subheader("RFM Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("RFM Summary (first 10 customers)")
        st.dataframe(rfm.head(10))
    
    with col2:
        if len(rfm) > 0:
            st.write("Customer Segments Distribution")
            segment_counts = rfm['Segment'].value_counts()
            fig_seg = px.pie(values=segment_counts.values, names=segment_counts.index,
                             title='Customer Segments Distribution')
            st.plotly_chart(fig_seg, use_container_width=True)
        else:
            st.warning("No data available for RFM analysis after cleaning")
    
    # Cohort Analysis
    st.subheader("Cohort Analysis")
    
    with st.spinner('Performing Cohort Analysis...'):
        # Calculate cohort index (months since signup)
        df_rfm['cohort_index'] = ((pd.to_datetime(df_rfm['year_month'] + '-01') - 
                                  pd.to_datetime(df_rfm['signup_month'] + '-01')).dt.days // 30)
        
        # Cohort retention analysis
        cohort_counts = df_rfm.groupby(['signup_month', 'cohort_index'])['customer_id'].nunique().reset_index()
        cohort_pivot = cohort_counts.pivot(index='signup_month', columns='cohort_index', values='customer_id')
        
        if not cohort_pivot.empty:
            cohort_size = cohort_pivot.iloc[:, 0]
            cohort_retention = cohort_pivot.divide(cohort_size, axis=0) * 100
            
            # Display retention matrix
            st.write("Customer Retention Rate (%) by Cohort")
            fig_coh = px.imshow(cohort_retention.round(1), 
                                labels=dict(x="Months Since Signup", y="Signup Month", color="Retention %"),
                                title='Cohort Retention Analysis',
                                aspect='auto',
                                color_continuous_scale='Viridis')
            st.plotly_chart(fig_coh, use_container_width=True)
        else:
            st.warning("Not enough data for cohort analysis")
    
    st.success("✅ RFM and Cohort Analysis completed!")

# -----------------------------
# Step 4: ML Models
# -----------------------------
elif step.startswith("4"):
    st.header("Step 4: ML Models")
    
    st.subheader("Feature Engineering")
    
    with st.spinner('Preparing data for ML models...'):
        # Use cleaned data
        df_ml = df_clean.copy()
        
        # Handle missing values
        numeric_cols = ['quantity', 'discount_applied', 'age', 'base_price']
        for col in numeric_cols:
            if col in df_ml.columns:
                df_ml[col] = df_ml[col].replace([np.inf, -np.inf], np.nan)
                df_ml[col] = df_ml[col].fillna(df_ml[col].median())
        
        categorical_cols = ['country', 'gender', 'loyalty_tier', 'acquisition_channel', 'category', 'brand']
        for col in categorical_cols:
            if col in df_ml.columns:
                df_ml[col] = df_ml[col].fillna("Unknown")
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in df_ml.columns:
                le = LabelEncoder()
                # Handle any NaN in categorical columns
                df_ml[col] = df_ml[col].astype(str)
                df_ml[col] = le.fit_transform(df_ml[col])
        
        # Create time-based features
        df_ml['days_since_signup'] = (df_ml['timestamp'] - df_ml['signup_date']).dt.days
        df_ml['product_age_days'] = (df_ml['timestamp'] - df_ml['launch_date']).dt.days
        df_ml['product_age_days'] = df_ml['product_age_days'].fillna(0)
        df_ml['is_premium'] = df_ml['is_premium'].fillna(0)
        
        # Ensure all values are finite
        df_ml = df_ml.replace([np.inf, -np.inf], np.nan)
        df_ml = df_ml.dropna()
        
        if len(df_ml) == 0:
            st.error("No data available for ML models after cleaning!")
            st.stop()
        
        # Features & target
        features = numeric_cols + ['days_since_signup', 'product_age_days', 'is_premium'] + categorical_cols
        features = [f for f in features if f in df_ml.columns]
        
        X = df_ml[features]
        y = df_ml['gross_revenue']
        
        # Check for any remaining infinite values
        if X.isnull().any().any() or np.isinf(X.values).any():
            st.error("Data still contains NaN or infinite values!")
            st.stop()
        
        if len(X) < 100:
            st.warning(f"Low sample size for ML models: {len(X)} samples")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        
        st.success(f"✅ Data prepared: {len(X)} samples, {len(features)} features")
    
    # Model selection
    model_option = st.selectbox("Choose Model:", ["Linear Regression", "LightGBM", "ANN", "Compare All Models"])
    
    if model_option == "Linear Regression":
        st.subheader("Linear Regression")
        
        with st.spinner('Training Linear Regression model...'):
            try:
                lr = LinearRegression()
                lr.fit(X_train, y_train)
                
                y_train_pred = lr.predict(X_train)
                y_val_pred = lr.predict(X_val)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Train MAE", f"${mean_absolute_error(y_train, y_train_pred):.2f}")
                with col2:
                    st.metric("Validation MAE", f"${mean_absolute_error(y_val, y_val_pred):.2f}")
                with col3:
                    st.metric("R² Score", f"{r2_score(y_val, y_val_pred):.4f}")
                
                # Actual vs Predicted plot
                fig_lr = px.scatter(x=y_val[:1000], y=y_val_pred[:1000], 
                                   labels={'x': 'Actual Revenue', 'y': 'Predicted Revenue'},
                                   title='Linear Regression: Actual vs Predicted (First 1000 samples)')
                fig_lr.add_shape(type='line', x0=y_val.min(), y0=y_val.min(),
                                x1=y_val.max(), y1=y_val.max(),
                                line=dict(color='red', dash='dash'))
                st.plotly_chart(fig_lr, use_container_width=True)
                
                st.success("✅ Linear Regression model trained successfully!")
            except Exception as e:
                st.error(f"Error training Linear Regression: {str(e)}")
        
    elif model_option == "LightGBM":
        st.subheader("LightGBM")
        
        with st.spinner('Training LightGBM model...'):
            try:
                # LightGBM can handle NaN values, but let's ensure clean data
                X_train_clean = X_train.copy()
                X_val_clean = X_val.copy()
                
                # Replace any remaining NaN with median
                for col in X_train_clean.columns:
                    X_train_clean[col] = X_train_clean[col].fillna(X_train_clean[col].median())
                    X_val_clean[col] = X_val_clean[col].fillna(X_train_clean[col].median())
                
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42,
                    verbose=-1
                )
                
                lgb_model.fit(X_train_clean, y_train)
                
                y_train_pred = lgb_model.predict(X_train_clean)
                y_val_pred = lgb_model.predict(X_val_clean)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Train RMSE", f"${mean_squared_error(y_train, y_train_pred, squared=False):.2f}")
                with col2:
                    st.metric("Validation RMSE", f"${mean_squared_error(y_val, y_val_pred, squared=False):.2f}")
                with col3:
                    st.metric("R² Score", f"{r2_score(y_val, y_val_pred):.4f}")
                
                # Feature importance
                importance = pd.DataFrame({
                    'feature': features,
                    'importance': lgb_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig_imp = px.bar(importance.head(10), x='importance', y='feature',
                                title='Top 10 Feature Importance (LightGBM)',
                                orientation='h')
                st.plotly_chart(fig_imp, use_container_width=True)
                
                st.success("✅ LightGBM model trained successfully!")
            except Exception as e:
                st.error(f"Error training LightGBM: {str(e)}")
        
    elif model_option == "ANN":
        st.subheader("ANN Model")
        
        with st.spinner('Training ANN model...'):
            try:
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Simple ANN architecture
                ann = Sequential([
                    Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
                    Dense(32, activation='relu'),
                    Dense(16, activation='relu'),
                    Dense(1, activation='linear')
                ])
                
                ann.compile(optimizer='adam', loss='mse', metrics=['mae'])
                
                # Train with validation
                history = ann.fit(
                    X_train_scaled, y_train,
                    validation_data=(X_val_scaled, y_val),
                    epochs=30,
                    batch_size=32,
                    verbose=0
                )
                
                y_train_pred = ann.predict(X_train_scaled)
                y_val_pred = ann.predict(X_val_scaled)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Train RMSE", f"${mean_squared_error(y_train, y_train_pred, squared=False):.2f}")
                with col2:
                    st.metric("Validation RMSE", f"${mean_squared_error(y_val, y_val_pred, squared=False):.2f}")
                with col3:
                    st.metric("R² Score", f"{r2_score(y_val, y_val_pred):.4f}")
                
                # Training history plot
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Train Loss'))
                fig_hist.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation Loss'))
                fig_hist.update_layout(title='ANN Training History', 
                                      xaxis_title='Epoch', 
                                      yaxis_title='Loss (MSE)')
                st.plotly_chart(fig_hist, use_container_width=True)
                
                st.success("✅ ANN model trained successfully!")
            except Exception as e:
                st.error(f"Error training ANN: {str(e)}")
    
    elif model_option == "Compare All Models":
        st.subheader("Model Comparison")
        
        with st.spinner('Training and comparing all models...'):
            try:
                models_performance = {}
                
                # 1. Linear Regression
                lr = LinearRegression()
                lr.fit(X_train, y_train)
                lr_pred = lr.predict(X_val)
                models_performance['Linear Regression'] = {
                    'RMSE': mean_squared_error(y_val, lr_pred, squared=False),
                    'MAE': mean_absolute_error(y_val, lr_pred),
                    'R2': r2_score(y_val, lr_pred)
                }
                
                # 2. LightGBM (simplified)
                lgb_simple = lgb.LGBMRegressor(n_estimators=50, learning_rate=0.05, random_state=42, verbose=-1)
                lgb_simple.fit(X_train, y_train)
                lgb_pred = lgb_simple.predict(X_val)
                models_performance['LightGBM'] = {
                    'RMSE': mean_squared_error(y_val, lgb_pred, squared=False),
                    'MAE': mean_absolute_error(y_val, lgb_pred),
                    'R2': r2_score(y_val, lgb_pred)
                }
                
                # Display comparison
                comparison_df = pd.DataFrame(models_performance).T
                st.dataframe(comparison_df.style.highlight_min(axis=0, color='lightgreen').highlight_max(axis=0, subset=['R2'], color='lightblue'))
                
                # Visual comparison
                fig_comp = go.Figure(data=[
                    go.Bar(name='RMSE', x=list(models_performance.keys()), 
                          y=[models_performance[m]['RMSE'] for m in models_performance]),
                    go.Bar(name='MAE', x=list(models_performance.keys()), 
                          y=[models_performance[m]['MAE'] for m in models_performance])
                ])
                fig_comp.update_layout(title='Model Comparison: RMSE & MAE (lower is better)',
                                      barmode='group')
                st.plotly_chart(fig_comp, use_container_width=True)
                
                st.success("✅ All models trained and compared successfully!")
            except Exception as e:
                st.error(f"Error comparing models: {str(e)}")

# -----------------------------
# Footer
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.info("""
**App Features:**
- 📊 Statistical Analysis
- 📈 Interactive Visualizations
- 👥 RFM & Cohort Analysis
- 🤖 ML Model Training
- 🔍 Real-time Insights
""")

# Add data quality check in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Data Quality")
if 'df_clean' in locals():
    st.sidebar.write(f"**Samples:** {len(df_clean):,}")
    st.sidebar.write(f"**Features:** {len(df_clean.columns)}")
    missing_pct = (df_clean.isnull().sum().sum() / (len(df_clean) * len(df_clean.columns))) * 100
    st.sidebar.write(f"**Missing Values:** {missing_pct:.2f}%")
