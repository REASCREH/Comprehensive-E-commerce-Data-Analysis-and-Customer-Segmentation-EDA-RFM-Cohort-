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
    
    # Create derived columns for visualizations
    df['year_month'] = df['timestamp'].dt.to_period('M').astype(str)
    df['weekday'] = df['timestamp'].dt.day_name()
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['hour'] = df['timestamp'].dt.hour
    df['launch_year'] = df['launch_date'].dt.year
    df['age_group'] = pd.cut(df['age'], bins=[18,25,35,45,55,65,100],
                             labels=['18-24','25-34','35-44','45-54','55-64','65+'])
    
    # Count of visualizations
    visualization_count = 0
    
    # Visualizations from the notebook
    st.subheader("📈 Temporal and Revenue Analysis")
    
    # 1. Monthly Revenue Line Chart
    with st.spinner('Creating Monthly Revenue Chart...'):
        monthly_rev = df.groupby('year_month')['gross_revenue'].sum().reset_index()
        fig1 = px.line(monthly_rev, x='year_month', y='gross_revenue', 
                       title='Monthly Revenue Trend', markers=True)
        fig1.update_layout(xaxis_title='Month', yaxis_title='Revenue ($)')
        st.plotly_chart(fig1, use_container_width=True)
        visualization_count += 1
    
    # 2. Daily Revenue Line Chart
    with st.spinner('Creating Daily Revenue Chart...'):
        daily_rev = df.groupby(df['timestamp'].dt.date)['gross_revenue'].sum().reset_index()
        daily_rev.columns = ['date', 'gross_revenue']
        fig2 = px.line(daily_rev, x='date', y='gross_revenue', 
                       title='Daily Revenue Pattern')
        fig2.update_layout(xaxis_title='Date', yaxis_title='Revenue ($)')
        st.plotly_chart(fig2, use_container_width=True)
        visualization_count += 1
    
    # 3. Revenue by Weekday
    with st.spinner('Creating Weekday Revenue Chart...'):
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_rev = df.groupby('weekday')['gross_revenue'].sum().reindex(weekday_order).reset_index()
        fig3 = px.bar(weekday_rev, x='weekday', y='gross_revenue', 
                      title='Revenue by Day of Week', color='weekday')
        fig3.update_layout(xaxis_title='Day of Week', yaxis_title='Revenue ($)')
        st.plotly_chart(fig3, use_container_width=True)
        visualization_count += 1
    
    # 4. Revenue by Month
    with st.spinner('Creating Monthly Distribution Chart...'):
        month_rev = df.groupby('month')['gross_revenue'].sum().reset_index()
        fig4 = px.bar(month_rev, x='month', y='gross_revenue', 
                      title='Revenue by Month', color='month')
        fig4.update_layout(xaxis_title='Month', yaxis_title='Revenue ($)')
        st.plotly_chart(fig4, use_container_width=True)
        visualization_count += 1
    
    st.subheader("🌍 Geographic Analysis")
    
    # 5. Revenue by Country
    with st.spinner('Creating Country Revenue Chart...'):
        country_rev = df.groupby('country')['gross_revenue'].sum().sort_values(ascending=False).reset_index()
        fig5 = px.bar(country_rev, x='country', y='gross_revenue', 
                      title='Total Revenue by Country', color='country')
        fig5.update_layout(xaxis_title='Country', yaxis_title='Revenue ($)')
        st.plotly_chart(fig5, use_container_width=True)
        visualization_count += 1
    
    # 6. Transaction Count by Country
    with st.spinner('Creating Transaction Count by Country Chart...'):
        country_trans = df.groupby('country')['transaction_id'].count().sort_values(ascending=False).reset_index()
        fig6 = px.bar(country_trans, x='country', y='transaction_id', 
                      title='Number of Transactions by Country', color='country')
        fig6.update_layout(xaxis_title='Country', yaxis_title='Number of Transactions')
        st.plotly_chart(fig6, use_container_width=True)
        visualization_count += 1
    
    st.subheader("👥 Customer Analysis")
    
    # 7. Revenue by Age Group
    with st.spinner('Creating Age Group Revenue Chart...'):
        age_rev = df.groupby('age_group')['gross_revenue'].sum().reset_index()
        fig7 = px.bar(age_rev, x='age_group', y='gross_revenue', 
                      title='Revenue by Age Group', color='age_group')
        fig7.update_layout(xaxis_title='Age Group', yaxis_title='Revenue ($)')
        st.plotly_chart(fig7, use_container_width=True)
        visualization_count += 1
    
    # 8. Age Distribution
    with st.spinner('Creating Age Distribution Chart...'):
        fig8 = px.histogram(df, x='age', nbins=30, 
                            title='Customer Age Distribution',
                            labels={'age': 'Age', 'count': 'Number of Customers'})
        fig8.update_layout(bargap=0.1)
        st.plotly_chart(fig8, use_container_width=True)
        visualization_count += 1
    
    # 9. Revenue by Gender
    with st.spinner('Creating Gender Revenue Chart...'):
        gender_rev = df.groupby('gender')['gross_revenue'].sum().reset_index()
        fig9 = px.bar(gender_rev, x='gender', y='gross_revenue', 
                      title='Revenue by Gender', color='gender')
        fig9.update_layout(xaxis_title='Gender', yaxis_title='Revenue ($)')
        st.plotly_chart(fig9, use_container_width=True)
        visualization_count += 1
    
    # 10. Revenue by Loyalty Tier
    with st.spinner('Creating Loyalty Tier Revenue Chart...'):
        loyalty_rev = df.groupby('loyalty_tier')['gross_revenue'].sum().reset_index()
        fig10 = px.bar(loyalty_rev, x='loyalty_tier', y='gross_revenue', 
                       title='Revenue by Loyalty Tier', color='loyalty_tier')
        fig10.update_layout(xaxis_title='Loyalty Tier', yaxis_title='Revenue ($)')
        st.plotly_chart(fig10, use_container_width=True)
        visualization_count += 1
    
    # 11. Revenue by Acquisition Channel
    with st.spinner('Creating Acquisition Channel Revenue Chart...'):
        channel_rev = df.groupby('acquisition_channel')['gross_revenue'].sum().reset_index()
        fig11 = px.bar(channel_rev, x='acquisition_channel', y='gross_revenue', 
                       title='Revenue by Acquisition Channel', color='acquisition_channel')
        fig11.update_layout(xaxis_title='Acquisition Channel', yaxis_title='Revenue ($)')
        st.plotly_chart(fig11, use_container_width=True)
        visualization_count += 1
    
    st.subheader("📦 Product Analysis")
    
    # 12. Revenue by Category
    with st.spinner('Creating Category Revenue Chart...'):
        category_rev = df.groupby('category')['gross_revenue'].sum().sort_values(ascending=False).reset_index()
        fig12 = px.bar(category_rev, x='category', y='gross_revenue', 
                       title='Revenue by Product Category', color='category')
        fig12.update_layout(xaxis_title='Category', yaxis_title='Revenue ($)')
        st.plotly_chart(fig12, use_container_width=True)
        visualization_count += 1
    
    # 13. Quantity Sold by Category
    with st.spinner('Creating Quantity by Category Chart...'):
        category_qty = df.groupby('category')['quantity'].sum().sort_values(ascending=False).reset_index()
        fig13 = px.bar(category_qty, x='category', y='quantity', 
                       title='Quantity Sold by Category', color='category')
        fig13.update_layout(xaxis_title='Category', yaxis_title='Quantity Sold')
        st.plotly_chart(fig13, use_container_width=True)
        visualization_count += 1
    
    # 14. Top 10 Brands by Revenue
    with st.spinner('Creating Top Brands Chart...'):
        top_brands = df.groupby('brand')['gross_revenue'].sum().nlargest(10).reset_index()
        fig14 = px.bar(top_brands, x='brand', y='gross_revenue', 
                       title='Top 10 Brands by Revenue', color='brand')
        fig14.update_layout(xaxis_title='Brand', yaxis_title='Revenue ($)', xaxis_tickangle=45)
        st.plotly_chart(fig14, use_container_width=True)
        visualization_count += 1
    
    # 15. Premium vs Non-Premium Revenue
    with st.spinner('Creating Premium vs Non-Premium Chart...'):
        premium_rev = df.groupby('is_premium')['gross_revenue'].sum().reset_index()
        premium_rev['is_premium'] = premium_rev['is_premium'].map({0: 'Non-Premium', 1: 'Premium'})
        fig15 = px.bar(premium_rev, x='is_premium', y='gross_revenue', 
                       title='Revenue: Premium vs Non-Premium Products', color='is_premium')
        fig15.update_layout(xaxis_title='Product Type', yaxis_title='Revenue ($)')
        st.plotly_chart(fig15, use_container_width=True)
        visualization_count += 1
    
    st.subheader("💰 Discount Analysis")
    
    # 16. Discount vs Gross Revenue Scatter
    with st.spinner('Creating Discount Analysis Chart...'):
        fig16 = px.scatter(df.sample(1000, random_state=42), x='discount_applied', y='gross_revenue',
                           title='Discount Applied vs Gross Revenue (Sample)',
                           labels={'discount_applied': 'Discount Applied (%)', 'gross_revenue': 'Revenue ($)'},
                           opacity=0.6)
        st.plotly_chart(fig16, use_container_width=True)
        visualization_count += 1
    
    # 17. Average Discount by Category
    with st.spinner('Creating Average Discount Chart...'):
        avg_discount = df.groupby('category')['discount_applied'].mean().reset_index()
        fig17 = px.bar(avg_discount, x='category', y='discount_applied',
                       title='Average Discount by Category', color='category')
        fig17.update_layout(xaxis_title='Category', yaxis_title='Average Discount (%)')
        st.plotly_chart(fig17, use_container_width=True)
        visualization_count += 1
    
    st.subheader("🔄 Refund Analysis")
    
    # 18. Refunds by Category
    with st.spinner('Creating Refunds by Category Chart...'):
        refunds_by_category = df[df['refund_flag'] == 1].groupby('category')['refund_flag'].count().reset_index()
        fig18 = px.bar(refunds_by_category, x='category', y='refund_flag',
                       title='Number of Refunds by Category', color='category')
        fig18.update_layout(xaxis_title='Category', yaxis_title='Number of Refunds')
        st.plotly_chart(fig18, use_container_width=True)
        visualization_count += 1
    
    # 19. Refunds by Country
    with st.spinner('Creating Refunds by Country Chart...'):
        refunds_by_country = df[df['refund_flag'] == 1].groupby('country')['refund_flag'].count().reset_index()
        fig19 = px.bar(refunds_by_country, x='country', y='refund_flag',
                       title='Number of Refunds by Country', color='country')
        fig19.update_layout(xaxis_title='Country', yaxis_title='Number of Refunds')
        st.plotly_chart(fig19, use_container_width=True)
        visualization_count += 1
    
    # 20. Overall Refund Rate Gauge
    with st.spinner('Creating Refund Rate Gauge...'):
        total_transactions = len(df)
        total_refunds = df['refund_flag'].sum()
        refund_rate = (total_refunds / total_transactions) * 100
        
        fig20 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=refund_rate,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Refund Rate (%)"},
            delta={'reference': 5, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 10]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 3], 'color': "lightgreen"},
                    {'range': [3, 6], 'color': "yellow"},
                    {'range': [6, 10], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 5
                }
            }
        ))
        fig20.update_layout(height=400)
        st.plotly_chart(fig20, use_container_width=True)
        visualization_count += 1
    
    st.subheader("📊 Advanced Analytics")
    
    # 21. Correlation Heatmap
    with st.spinner('Creating Correlation Heatmap...'):
        numeric_cols_corr = ['quantity', 'discount_applied', 'gross_revenue', 'age', 'base_price', 'is_premium']
        corr_matrix = df[numeric_cols_corr].corr()
        
        fig21 = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            annotation_text=corr_matrix.round(2).values,
            colorscale='RdBu',
            showscale=True
        )
        fig21.update_layout(title='Correlation Matrix Heatmap')
        st.plotly_chart(fig21, use_container_width=True)
        visualization_count += 1
    
    # 22. Campaign Performance
    with st.spinner('Creating Campaign Performance Chart...'):
        if 'campaign_id' in df.columns:
            campaign_perf = df.groupby('campaign_id')['gross_revenue'].agg(['sum', 'count', 'mean']).reset_index()
            campaign_perf = campaign_perf[campaign_perf['campaign_id'] != 0]  # Exclude no campaign
            
            fig22 = px.scatter(campaign_perf, x='count', y='sum', size='mean',
                               title='Campaign Performance: Revenue vs Number of Transactions',
                               labels={'count': 'Number of Transactions', 'sum': 'Total Revenue ($)', 'mean': 'Avg Revenue'},
                               hover_data=['campaign_id'])
            st.plotly_chart(fig22, use_container_width=True)
            visualization_count += 1
    
    # 23. Customer Lifetime Value Distribution
    with st.spinner('Creating Customer Lifetime Value Distribution...'):
        customer_clv = df.groupby('customer_id')['gross_revenue'].sum()
        fig23 = px.histogram(customer_clv, nbins=50, 
                             title='Customer Lifetime Value Distribution',
                             labels={'value': 'Customer Lifetime Value ($)', 'count': 'Number of Customers'})
        st.plotly_chart(fig23, use_container_width=True)
        visualization_count += 1
    
    # Show completion message at the very end
    st.success(f"✅ Successfully displayed {visualization_count} key visualizations from the comprehensive analysis!")
    
    # Add a summary table of visualizations created
    with st.expander("📋 Visualization Summary"):
        visualization_types = {
            "Temporal and Revenue Analysis": 4,
            "Geographic Analysis": 2,
            "Customer Analysis": 5,
            "Product Analysis": 4,
            "Discount Analysis": 2,
            "Refund Analysis": 3,
            "Advanced Analytics": 3
        }
        
        summary_df = pd.DataFrame({
            'Category': list(visualization_types.keys()),
            'Number of Visualizations': list(visualization_types.values())
        })
        
        # Add total row
        total_row = pd.DataFrame({
            'Category': ['TOTAL'],
            'Number of Visualizations': [sum(visualization_types.values())]
        })
        summary_df = pd.concat([summary_df, total_row], ignore_index=True)
        
        st.dataframe(summary_df.style.highlight_max(subset=['Number of Visualizations'], color='lightgreen'))

# -----------------------------
# Step 3: RFM & Cohort Analysis
# -----------------------------
elif step.startswith("3"):
    st.header("Step 3: RFM & Cohort Analysis")
    
    # Create necessary time-based columns
    df['year_month'] = df['timestamp'].dt.to_period('M').astype(str)
    df['signup_month'] = df['signup_date'].dt.to_period('M').astype(str)
    
    today = df['timestamp'].max() + pd.Timedelta(days=1)
    
    # RFM Analysis
    with st.spinner('Calculating RFM Scores...'):
        rfm = df.groupby('customer_id').agg({
            'timestamp': lambda x: (today - x.max()).days,
            'transaction_id': 'count',
            'gross_revenue': 'sum'
        }).rename(columns={'timestamp':'Recency','transaction_id':'Frequency','gross_revenue':'Monetary'})
        
        # Calculate RFM scores
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=['4','3','2','1'])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=['1','2','3','4'])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=['1','2','3','4'])
        
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
        st.write("Customer Segments Distribution")
        segment_counts = rfm['Segment'].value_counts()
        fig_seg = px.pie(values=segment_counts.values, names=segment_counts.index,
                         title='Customer Segments Distribution')
        st.plotly_chart(fig_seg, use_container_width=True)
    
    # Top 10 Customers by CLV
    st.subheader("Top 10 Customers by Lifetime Value")
    top_clv = rfm.sort_values('CLV', ascending=False).head(10)
    fig_top = px.bar(top_clv.reset_index(), x='customer_id', y='CLV',
                     title='Top 10 Customers by Customer Lifetime Value',
                     labels={'customer_id': 'Customer ID', 'CLV': 'Lifetime Value ($)'})
    st.plotly_chart(fig_top, use_container_width=True)
    
    # Cohort Analysis
    st.subheader("Cohort Analysis")
    
    with st.spinner('Performing Cohort Analysis...'):
        # Calculate cohort index (months since signup)
        df['cohort_index'] = ((pd.to_datetime(df['year_month'] + '-01') - 
                              pd.to_datetime(df['signup_month'] + '-01')).dt.days // 30)
        
        # Cohort retention analysis
        cohort_counts = df.groupby(['signup_month', 'cohort_index'])['customer_id'].nunique().reset_index()
        cohort_pivot = cohort_counts.pivot(index='signup_month', columns='cohort_index', values='customer_id')
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
        
        # Cohort revenue analysis
        cohort_rev = df.groupby(['signup_month', 'cohort_index'])['gross_revenue'].sum().reset_index()
        st.write("Cohort Revenue Analysis (first 20 rows)")
        st.dataframe(cohort_rev.head(20))
    
    st.success("✅ RFM and Cohort Analysis completed successfully!")

# -----------------------------
# Step 4: ML Models
# -----------------------------
elif step.startswith("4"):
    st.header("Step 4: ML Models")
    
    st.subheader("Feature Engineering")
    
    with st.spinner('Preparing data for ML models...'):
        # Handle missing values
        numeric_cols = ['quantity', 'discount_applied', 'age', 'base_price']
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        categorical_cols = ['country', 'gender', 'loyalty_tier', 'acquisition_channel', 'category', 'brand']
        for col in categorical_cols:
            df[col] = df[col].fillna("Unknown")
        
        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        
        # Create time-based features
        df['days_since_signup'] = (df['timestamp'] - df['signup_date']).dt.days
        df['product_age_days'] = (df['timestamp'] - df['launch_date']).dt.days.fillna(0)
        df['is_premium'] = df['is_premium'].fillna(0)
        
        # Features & target
        features = numeric_cols + ['days_since_signup', 'product_age_days', 'is_premium'] + categorical_cols
        X = df[features]
        y = df['gross_revenue']
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model selection
    model_option = st.selectbox("Choose Model:", ["Linear Regression", "LightGBM", "ANN", "Compare All Models"])
    
    if model_option == "Linear Regression":
        st.subheader("Linear Regression")
        
        with st.spinner('Training Linear Regression model...'):
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
        fig_lr = px.scatter(x=y_val, y=y_val_pred, 
                           labels={'x': 'Actual Revenue', 'y': 'Predicted Revenue'},
                           title='Linear Regression: Actual vs Predicted')
        fig_lr.add_shape(type='line', x0=y_val.min(), y0=y_val.min(),
                        x1=y_val.max(), y1=y_val.max(),
                        line=dict(color='red', dash='dash'))
        st.plotly_chart(fig_lr, use_container_width=True)
        
        st.success("✅ Linear Regression model trained successfully!")
        
    elif model_option == "LightGBM":
        st.subheader("LightGBM")
        
        with st.spinner('Training LightGBM model...'):
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': -1,
                'min_data_in_leaf': 20,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
            
            model = lgb.train(params, lgb_train, num_boost_round=200, 
                             valid_sets=[lgb_train, lgb_val], verbose_eval=False)
            
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
        
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
            'importance': model.feature_importance()
        }).sort_values('importance', ascending=False)
        
        fig_imp = px.bar(importance.head(10), x='importance', y='feature',
                        title='Top 10 Feature Importance (LightGBM)',
                        orientation='h')
        st.plotly_chart(fig_imp, use_container_width=True)
        
        # Actual vs Predicted plot
        fig_lgb = px.scatter(x=y_val, y=y_val_pred, 
                            labels={'x': 'Actual Revenue', 'y': 'Predicted Revenue'},
                            title='LightGBM: Actual vs Predicted')
        fig_lgb.add_shape(type='line', x0=y_val.min(), y0=y_val.min(),
                         x1=y_val.max(), y1=y_val.max(),
                         line=dict(color='red', dash='dash'))
        st.plotly_chart(fig_lgb, use_container_width=True)
        
        st.success("✅ LightGBM model trained successfully!")
        
    elif model_option == "ANN":
        st.subheader("ANN Model")
        
        with st.spinner('Training ANN model...'):
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            ann = Sequential()
            ann.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
            ann.add(Dense(64, activation='relu'))
            ann.add(Dense(32, activation='relu'))
            ann.add(Dense(1, activation='linear'))
            
            ann.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Train with progress bar
            epochs = 50
            batch_size = 32
            
            history = ann.fit(X_train_scaled, y_train, 
                             validation_data=(X_val_scaled, y_val), 
                             epochs=epochs, batch_size=batch_size, verbose=0)
        
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
        
        # Actual vs Predicted plot
        fig_ann = px.scatter(x=y_val, y=y_val_pred.flatten(), 
                            labels={'x': 'Actual Revenue', 'y': 'Predicted Revenue'},
                            title='ANN: Actual vs Predicted')
        fig_ann.add_shape(type='line', x0=y_val.min(), y0=y_val.min(),
                         x1=y_val.max(), y1=y_val.max(),
                         line=dict(color='red', dash='dash'))
        st.plotly_chart(fig_ann, use_container_width=True)
        
        st.success("✅ ANN model trained successfully!")
    
    elif model_option == "Compare All Models":
        st.subheader("Model Comparison")
        
        with st.spinner('Training and comparing all models...'):
            models = {}
            
            # Linear Regression
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            lr_pred = lr.predict(X_val)
            models['Linear Regression'] = {
                'RMSE': mean_squared_error(y_val, lr_pred, squared=False),
                'MAE': mean_absolute_error(y_val, lr_pred),
                'R2': r2_score(y_val, lr_pred)
            }
            
            # LightGBM
            lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
            lgb_model.fit(X_train, y_train)
            lgb_pred = lgb_model.predict(X_val)
            models['LightGBM'] = {
                'RMSE': mean_squared_error(y_val, lgb_pred, squared=False),
                'MAE': mean_absolute_error(y_val, lgb_pred),
                'R2': r2_score(y_val, lgb_pred)
            }
            
            # ANN
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            ann_simple = Sequential()
            ann_simple.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
            ann_simple.add(Dense(32, activation='relu'))
            ann_simple.add(Dense(1, activation='linear'))
            ann_simple.compile(optimizer='adam', loss='mse')
            ann_simple.fit(X_train_scaled, y_train, epochs=20, batch_size=32, verbose=0)
            ann_pred = ann_simple.predict(X_val_scaled)
            models['ANN'] = {
                'RMSE': mean_squared_error(y_val, ann_pred, squared=False),
                'MAE': mean_absolute_error(y_val, ann_pred),
                'R2': r2_score(y_val, ann_pred)
            }
            
            # Display comparison
            comparison_df = pd.DataFrame(models).T
            st.dataframe(comparison_df.style.highlight_min(axis=0, color='lightgreen'))
            
            # Visual comparison
            fig_comp = go.Figure(data=[
                go.Bar(name='RMSE', x=list(models.keys()), 
                      y=[models[m]['RMSE'] for m in models]),
                go.Bar(name='MAE', x=list(models.keys()), 
                      y=[models[m]['MAE'] for m in models])
            ])
            fig_comp.update_layout(title='Model Comparison: RMSE & MAE (lower is better)',
                                  barmode='group')
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # R2 Score comparison
            fig_r2 = px.bar(x=list(models.keys()), 
                           y=[models[m]['R2'] for m in models],
                           labels={'x': 'Model', 'y': 'R² Score'},
                           title='Model Comparison: R² Score (higher is better)',
                           color=list(models.keys()))
            fig_r2.update_layout(showlegend=False)
            st.plotly_chart(fig_r2, use_container_width=True)
        
        st.success("✅ All models trained and compared successfully!")

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

st.sidebar.markdown("---")
st.sidebar.success("""
**Total Visualizations:**
- 23+ Interactive Charts
- 4 ML Models
- RFM & Cohort Analysis
- Real-time Updates
""")
