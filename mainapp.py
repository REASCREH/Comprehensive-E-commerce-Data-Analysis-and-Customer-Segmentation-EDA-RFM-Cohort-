import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import glob
import json

# Create output directories
os.makedirs('output/plots', exist_ok=True)
os.makedirs('output/data', exist_ok=True)

# Install required packages
def install_packages():
    """Install required packages"""
    import subprocess
    import importlib
    
    packages = ['kagglehub', 'kaleido', 'plotly', 'seaborn', 'matplotlib']
    
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")

# Function to save plotly figures with error handling
def save_plotly_fig(fig, filename):
    """Save plotly figure in multiple formats with error handling"""
    
    # Always save as HTML
    try:
        fig.write_html(f"output/plots/{filename}.html")
        print(f"✓ Saved HTML: output/plots/{filename}.html")
    except Exception as e:
        print(f"⚠ Error saving HTML: {e}")
    
    # Try to save as PNG if kaleido is available
    try:
        import kaleido
        fig.write_image(f"output/plots/{filename}.png")
        print(f"✓ Saved PNG: output/plots/{filename}.png")
    except Exception as e:
        print(f"⚠ Could not save PNG (kaleido issue): {e}")
    
    # Try to save as PDF if kaleido is available
    try:
        import kaleido
        fig.write_image(f"output/plots/{filename}.pdf")
        print(f"✓ Saved PDF: output/plots/{filename}.pdf")
    except Exception as e:
        print(f"⚠ Could not save PDF (kaleido issue): {e}")
    
    # Save matplotlib version as backup
    try:
        plt.figure(figsize=(12, 8))
        
        # For line charts
        if fig.data[0].type == 'scatter' and fig.data[0].mode == 'lines':
            for trace in fig.data:
                plt.plot(trace.x, trace.y, label=trace.name if hasattr(trace, 'name') else None)
            plt.title(fig.layout.title.text if hasattr(fig.layout, 'title') else filename)
            plt.legend()
        
        # For bar charts
        elif fig.data[0].type == 'bar':
            for trace in fig.data:
                plt.bar(trace.x, trace.y, label=trace.name if hasattr(trace, 'name') else None)
            plt.title(fig.layout.title.text if hasattr(fig.layout, 'title') else filename)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"output/plots/{filename}_matplotlib.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"output/plots/{filename}_matplotlib.pdf", bbox_inches='tight')
        plt.close()
        print(f"✓ Saved matplotlib backup: output/plots/{filename}_matplotlib.[png/pdf]")
        
    except Exception as e:
        print(f"⚠ Could not save matplotlib backup: {e}")

def download_kaggle_dataset():
    """Download dataset from Kaggle using kagglehub"""
    try:
        import kagglehub
        print("Downloading dataset from Kaggle...")
        path = kagglehub.dataset_download("geethasagarbonthu/marketing-and-e-commerce-analytics-dataset")
        print(f"✓ Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"⚠ Error downloading from Kaggle: {e}")
        print("Trying alternative approach...")
        return None

def load_local_dataset():
    """Load dataset from local files if they exist"""
    # Check for local CSV files
    local_files = glob.glob("*.csv")
    
    if len(local_files) >= 3:  # At least transactions, customers, products
        print(f"Found local CSV files: {local_files}")
        
        data_frames = {}
        for file in local_files:
            try:
                df = pd.read_csv(file)
                data_frames[file] = df
                print(f"  Loaded {file}: {df.shape}")
            except Exception as e:
                print(f"  Error loading {file}: {e}")
        
        return data_frames
    
    return None

def create_sample_data():
    """Create sample data for demonstration"""
    print("Creating sample data for demonstration...")
    
    np.random.seed(42)
    n_samples = 5000
    
    dates = pd.date_range('2021-01-01', '2023-12-31', freq='H')
    timestamps = np.random.choice(dates, n_samples)
    
    sample_data = pd.DataFrame({
        'transaction_id': range(10001, 10001 + n_samples),
        'timestamp': timestamps,
        'customer_id': np.random.randint(1000, 5000, n_samples),
        'product_id': np.random.randint(1, 200, n_samples),
        'quantity': np.random.randint(1, 5, n_samples),
        'discount_applied': np.random.uniform(0, 0.3, n_samples).round(2),
        'gross_revenue': np.random.exponential(100, n_samples).round(2),
        'campaign_id': np.random.choice([0] + list(range(1, 51)), n_samples),
        'refund_flag': np.random.choice([0, 1], n_samples, p=[0.97, 0.03]),
        'signup_date': np.random.choice(pd.date_range('2021-01-01', '2023-12-31'), n_samples),
        'country': np.random.choice(['US', 'IN', 'UK', 'CA', 'DE', 'AU', 'BR'], n_samples),
        'age': np.random.randint(18, 70, n_samples),
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.48, 0.04]),
        'loyalty_tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], n_samples, p=[0.54, 0.27, 0.15, 0.04]),
        'acquisition_channel': np.random.choice(['Organic', 'Paid Search', 'Email', 'Social', 'Referral'], n_samples),
        'category': np.random.choice(['Electronics', 'Fashion', 'Home', 'Grocery', 'Sports', 'Beauty'], n_samples),
        'brand': ['Brand_' + str(np.random.randint(1, 100)) for _ in range(n_samples)],
        'base_price': np.random.uniform(10, 500, n_samples).round(2),
        'launch_date': np.random.choice(pd.date_range('2020-01-01', '2023-01-01'), n_samples),
        'is_premium': np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    })
    
    return sample_data

def main():
    print("="*70)
    print("E-COMMERCE CUSTOMER ANALYTICS ANALYSIS")
    print("="*70)
    
    # Install required packages
    print("\n📦 Checking and installing required packages...")
    install_packages()
    
    # -----------------------------
    # Load datasets
    # -----------------------------
    print("\n1️⃣ LOADING DATASETS...")
    
    # Try multiple approaches to get data
    df = None
    
    # Approach 1: Try Kaggle download
    print("Attempt 1: Downloading from Kaggle...")
    kaggle_path = download_kaggle_dataset()
    
    if kaggle_path:
        try:
            # Load files from Kaggle path
            csv_files = glob.glob(os.path.join(kaggle_path, "*.csv"))
            print(f"Found {len(csv_files)} CSV files in Kaggle download")
            
            data_frames = {}
            for csv_file in csv_files:
                try:
                    df_temp = pd.read_csv(csv_file)
                    file_name = os.path.basename(csv_file)
                    data_frames[file_name] = df_temp
                    print(f"  ✓ {file_name}: {df_temp.shape}")
                except Exception as e:
                    print(f"  ✗ Error loading {csv_file}: {e}")
            
            # Merge files
            if 'transactions.csv' in data_frames:
                df = data_frames['transactions.csv']
                if 'customers.csv' in data_frames:
                    df = df.merge(data_frames['customers.csv'], on="customer_id", how="left")
                if 'products.csv' in data_frames:
                    df = df.merge(data_frames['products.csv'], on="product_id", how="left")
                    
        except Exception as e:
            print(f"Error processing Kaggle files: {e}")
    
    # Approach 2: Try local files
    if df is None or df.empty:
        print("\nAttempt 2: Looking for local CSV files...")
        local_data = load_local_dataset()
        
        if local_data:
            # Merge local files
            for name, df_temp in local_data.items():
                if 'transactions' in name.lower():
                    df = df_temp
                elif 'customers' in name.lower() and df is not None:
                    df = df.merge(df_temp, on="customer_id", how="left")
                elif 'products' in name.lower() and df is not None:
                    df = df.merge(df_temp, on="product_id", how="left")
    
    # Approach 3: Create sample data
    if df is None or df.empty:
        print("\nAttempt 3: Creating sample data for demonstration...")
        df = create_sample_data()
        print("✓ Created sample dataset for analysis")
    
    # Check if we have data
    if df is None or df.empty:
        print("❌ ERROR: Could not load or create any data!")
        return
    
    print(f"\n✅ Successfully loaded data: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
    
    # -----------------------------
    # Data Preprocessing
    # -----------------------------
    print("\n2️⃣ PREPROCESSING DATA...")
    
    # Convert date columns
    date_columns = []
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower() or 'timestamp' in col.lower():
            date_columns.append(col)
    
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"  ✓ Converted {col} to datetime")
        except:
            print(f"  ✗ Could not convert {col} to datetime")
    
    # -----------------------------
    # Save processed data
    # -----------------------------
    print("\n3️⃣ SAVING PROCESSED DATA...")
    df.to_csv('output/data/merged_data.csv', index=False)
    print("✓ Saved merged data to output/data/merged_data.csv")
    
    # -----------------------------
    # Basic info
    # -----------------------------
    print("\n4️⃣ GENERATING BASIC DATASET INFO...")
    
    with open('output/data/basic_info.txt', 'w') as f:
        f.write("DATASET INFORMATION\n")
        f.write("="*60 + "\n")
        f.write(f"Shape: {df.shape}\n")
        f.write(f"Rows: {df.shape[0]:,}\n")
        f.write(f"Columns: {df.shape[1]}\n\n")
        
        f.write("COLUMN NAMES:\n")
        for i, col in enumerate(df.columns.tolist(), 1):
            f.write(f"  {i:2d}. {col}\n")
        
        f.write("\nDATA TYPES:\n")
        for col, dtype in df.dtypes.items():
            f.write(f"  - {col}: {dtype}\n")
        
        f.write("\nMISSING VALUES:\n")
        missing = df.isnull().sum()
        missing_count = 0
        for col, count in missing.items():
            if count > 0:
                missing_count += 1
                f.write(f"  - {col}: {count:,} missing values ({count/len(df)*100:.2f}%)\n")
        
        if missing_count == 0:
            f.write("  No missing values found!\n")
    
    # -----------------------------
    # Advanced Numerical Descriptive Stats
    # -----------------------------
    print("\n5️⃣ CALCULATING ADVANCED STATISTICS...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        num_stats = df[numeric_cols].describe().T
        num_stats['skew'] = df[numeric_cols].skew()
        num_stats['kurtosis'] = df[numeric_cols].kurtosis()
        num_stats['25%'] = df[numeric_cols].quantile(0.25)
        num_stats['50%'] = df[numeric_cols].quantile(0.50)
        num_stats['75%'] = df[numeric_cols].quantile(0.75)
        
        # Save numerical stats
        num_stats.to_csv('output/data/numerical_statistics.csv')
        print("✓ Saved numerical statistics")
    else:
        print("⚠ No numeric columns found")
    
    # -----------------------------
    # Categorical Stats
    # -----------------------------
    print("\n6️⃣ ANALYZING CATEGORICAL VARIABLES...")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    with open('output/data/categorical_analysis.txt', 'w') as f:
        for col in categorical_cols[:10]:  # Limit to first 10 categorical columns
            f.write(f"\n{'='*50}\n")
            f.write(f"COLUMN: {col}\n")
            f.write('='*50 + "\n")
            
            value_counts = df[col].value_counts().head(20)  # Top 20 values only
            proportions = (df[col].value_counts(normalize=True).head(20) * 100).round(2)
            
            f.write("\nTop 20 Value Counts:\n")
            for value, count in value_counts.items():
                f.write(f"  {value}: {count:,} ({proportions[value]:.1f}%)\n")
    
    # -----------------------------
    # Correlation Analysis
    # -----------------------------
    print("\n7️⃣ PERFORMING CORRELATION ANALYSIS...")
    
    if len(numeric_cols) > 1:
        correlation_matrix = df[numeric_cols].corr().round(2)
        correlation_matrix.to_csv('output/data/correlation_matrix.csv')
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix Heatmap', fontsize=16)
        plt.tight_layout()
        plt.savefig('output/plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig('output/plots/correlation_heatmap.pdf', bbox_inches='tight')
        plt.close()
        print("✓ Created correlation heatmap")
    else:
        print("⚠ Not enough numeric columns for correlation analysis")
    
    # -----------------------------
    # Revenue Analysis (if revenue column exists)
    # -----------------------------
    print("\n8️⃣ ANALYZING REVENUE PATTERNS...")
    
    revenue_columns = [col for col in df.columns if 'revenue' in col.lower() or 'price' in col.lower()]
    
    if revenue_columns:
        revenue_col = revenue_columns[0]  # Use first revenue-like column
        
        revenue_summary = {
            'Total Revenue': df[revenue_col].sum(),
            'Mean Revenue per Transaction': df[revenue_col].mean(),
            'Median Revenue': df[revenue_col].median(),
            'Revenue Std Dev': df[revenue_col].std(),
            'Max Revenue': df[revenue_col].max(),
            'Min Revenue': df[revenue_col].min()
        }
        
        revenue_percentiles = df[revenue_col].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        
        with open('output/data/revenue_analysis.txt', 'w') as f:
            f.write("REVENUE ANALYSIS\n")
            f.write("="*50 + "\n")
            for key, value in revenue_summary.items():
                f.write(f"{key}: ${value:,.2f}\n")
            
            f.write("\nRevenue Percentiles:\n")
            for quantile, value in revenue_percentiles.items():
                f.write(f"  {quantile*100:.0f}th percentile: ${value:,.2f}\n")
        
        print("✓ Revenue analysis completed")
    else:
        print("⚠ No revenue column found")
    
    # -----------------------------
    # Temporal Analysis
    # -----------------------------
    print("\n9️⃣ CREATING TEMPORAL VISUALIZATIONS...")
    
    # Check for timestamp column
    time_cols = [col for col in df.columns if 'timestamp' in col.lower() or 'date' in col.lower()]
    
    if time_cols:
        time_col = time_cols[0]
        
        # Create time-based columns
        df['analysis_date'] = pd.to_datetime(df[time_col])
        df['month'] = df['analysis_date'].dt.month
        df['year'] = df['analysis_date'].dt.year
        df['weekday'] = df['analysis_date'].dt.day_name()
        df['year_month'] = df['analysis_date'].dt.to_period('M').astype(str)
        
        # Monthly Plot (using first numeric column as value)
        if len(numeric_cols) > 0:
            value_col = numeric_cols[0]
            
            # Monthly aggregation
            monthly_data = df.groupby('year_month')[value_col].sum().reset_index()
            
            fig1 = px.line(monthly_data, x='year_month', y=value_col,
                         title=f'Monthly {value_col.replace("_", " ").title()}',
                         labels={'year_month': 'Month', value_col: value_col.replace("_", " ").title()},
                         markers=True)
            fig1.update_layout(template='plotly_white')
            save_plotly_fig(fig1, 'monthly_trend')
            
            # Daily aggregation
            daily_data = df.groupby(df['analysis_date'].dt.date)[value_col].sum().reset_index()
            daily_data['date'] = pd.to_datetime(daily_data['analysis_date'])
            
            fig2 = px.line(daily_data, x='date', y=value_col,
                         title=f'Daily {value_col.replace("_", " ").title()}',
                         labels={'date': 'Date', value_col: value_col.replace("_", " ").title()})
            fig2.update_layout(template='plotly_white')
            save_plotly_fig(fig2, 'daily_trend')
            
            print("✓ Created temporal visualizations")
    else:
        print("⚠ No timestamp column found for temporal analysis")
    
    # -----------------------------
    # Distribution Analysis
    # -----------------------------
    print("\n🔟 CREATING DISTRIBUTION VISUALIZATIONS...")
    
    # Analyze categorical distributions
    for cat_col in categorical_cols[:5]:  # First 5 categorical columns
        try:
            value_counts = df[cat_col].value_counts().head(10)  # Top 10 values
            
            fig = px.bar(x=value_counts.index, y=value_counts.values,
                        title=f'Top 10 {cat_col.replace("_", " ").title()} Distribution',
                        labels={'x': cat_col.replace("_", " ").title(), 'y': 'Count'})
            fig.update_layout(template='plotly_white')
            save_plotly_fig(fig, f'distribution_{cat_col}')
            
        except Exception as e:
            print(f"⚠ Could not create distribution for {cat_col}: {e}")
    
    # -----------------------------
    # RFM Analysis (if customer_id exists)
    # -----------------------------
    print("\n1️⃣1️⃣ PERFORMING RFM ANALYSIS...")
    
    if 'customer_id' in df.columns and time_cols and revenue_columns:
        today = df[time_cols[0]].max() + pd.Timedelta(days=1)
        
        # Determine frequency column
        if 'transaction_id' in df.columns:
            freq_col = 'transaction_id'
            freq_agg = 'count'
        else:
            # Use first column as frequency counter
            freq_col = df.columns[0]
            freq_agg = 'count'
        
        rfm = df.groupby('customer_id').agg({
            time_cols[0]: lambda x: (today - x.max()).days,   # Recency
            freq_col: freq_agg,                               # Frequency
            revenue_columns[0]: 'sum'                         # Monetary
        })
        
        # Rename columns
        rfm = rfm.rename(columns={
            time_cols[0]: 'Recency',
            freq_col: 'Frequency',
            revenue_columns[0]: 'Monetary'
        })
        
        rfm['CLV'] = rfm['Monetary']
        rfm.to_csv('output/data/rfm_analysis.csv')
        print("✓ RFM analysis completed and saved")
    else:
        print("⚠ Missing required columns for RFM analysis")
    
    # -----------------------------
    # Create Summary Report
    # -----------------------------
    print("\n1️⃣2️⃣ GENERATING SUMMARY REPORT...")
    
    with open('output/data/summary_report.txt', 'w') as f:
        f.write("E-COMMERCE ANALYTICS SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("DATASET OVERVIEW\n")
        f.write("-"*40 + "\n")
        f.write(f"Total Records: {len(df):,}\n")
        f.write(f"Total Columns: {len(df.columns)}\n")
        f.write(f"Date Range: {df[time_cols[0]].min().date() if time_cols else 'N/A'} to {df[time_cols[0]].max().date() if time_cols else 'N/A'}\n")
        f.write(f"Missing Values: {df.isnull().sum().sum():,}\n\n")
        
        f.write("KEY INSIGHTS\n")
        f.write("-"*40 + "\n")
        
        # Add insights based on available data
        if revenue_columns:
            f.write(f"• Total Revenue: ${df[revenue_columns[0]].sum():,.2f}\n")
            f.write(f"• Average Transaction Value: ${df[revenue_columns[0]].mean():.2f}\n")
        
        if 'customer_id' in df.columns:
            f.write(f"• Total Unique Customers: {df['customer_id'].nunique():,}\n")
        
        if 'category' in df.columns:
            top_category = df['category'].value_counts().index[0] if not df['category'].isnull().all() else 'N/A'
            f.write(f"• Most Popular Category: {top_category}\n")
        
        if 'country' in df.columns:
            top_country = df['country'].value_counts().index[0] if not df['country'].isnull().all() else 'N/A'
            f.write(f"• Top Country: {top_country}\n")
        
        f.write("\nRECOMMENDATIONS\n")
        f.write("-"*40 + "\n")
        f.write("1. Focus on high-value customer segments identified in RFM analysis\n")
        f.write("2. Optimize marketing campaigns based on acquisition channel performance\n")
        f.write("3. Monitor and improve customer retention rates\n")
        f.write("4. Analyze seasonal patterns to optimize inventory and promotions\n")
        f.write("5. Address any data quality issues identified in the analysis\n")
    
    print("✓ Summary report generated")
    
    # -----------------------------
    # Create combined PDF report
    # -----------------------------
    print("\n1️⃣3️⃣ CREATING COMBINED PDF REPORT...")
    
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages('output/plots/analysis_report.pdf') as pdf:
            # Title page
            plt.figure(figsize=(11, 8.5))
            plt.text(0.5, 0.5, 'E-commerce Analytics Report\n\nAnalysis Date: ' + datetime.now().strftime('%Y-%m-%d'), 
                    horizontalalignment='center', verticalalignment='center', 
                    fontsize=20, fontweight='bold')
            plt.axis('off')
            pdf.savefig()
            plt.close()
            
            # Dataset overview
            plt.figure(figsize=(11, 8.5))
            plt.text(0.1, 0.9, 'Dataset Overview', fontsize=16, fontweight='bold')
            plt.text(0.1, 0.8, f'Total Records: {len(df):,}', fontsize=12)
            plt.text(0.1, 0.75, f'Total Columns: {len(df.columns)}', fontsize=12)
            plt.text(0.1, 0.7, f'Missing Values: {df.isnull().sum().sum():,}', fontsize=12)
            plt.axis('off')
            pdf.savefig()
            plt.close()
            
            # Add correlation heatmap if available
            if len(numeric_cols) > 1:
                plt.figure(figsize=(11, 8.5))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
                plt.title('Correlation Matrix', fontsize=14)
                plt.tight_layout()
                pdf.savefig()
                plt.close()
        
        print("✓ PDF report created: output/plots/analysis_report.pdf")
        
    except Exception as e:
        print(f"⚠ Could not create PDF report: {e}")
    
    # -----------------------------
    # Final Output Summary
    # -----------------------------
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    
    print(f"\n📁 OUTPUT FILES SAVED:")
    print(f"   Data Files:     output/data/")
    print(f"   Visualizations: output/plots/")
    print(f"   PDF Report:     output/plots/analysis_report.pdf")
    
    print(f"\n📊 KEY FILES GENERATED:")
    
    # List important files
    import glob
    data_files = glob.glob("output/data/*")
    plot_files = glob.glob("output/plots/*.html") + glob.glob("output/plots/*.pdf")
    
    print(f"\n   Data Files ({len(data_files)}):")
    for file in sorted(data_files)[:5]:  # Show first 5
        print(f"     • {os.path.basename(file)}")
    
    print(f"\n   Visualization Files ({len(plot_files)}):")
    for file in sorted(plot_files)[:5]:  # Show first 5
        print(f"     • {os.path.basename(file)}")
    
    if len(data_files) > 5:
        print(f"     ... and {len(data_files) - 5} more files")
    
    print(f"\n📈 NEXT STEPS:")
    print("   1. Review output/data/summary_report.txt for key insights")
    print("   2. Open HTML files in browser for interactive visualizations")
    print("   3. Share PDF report with stakeholders")
    print("   4. Use RFM analysis for customer segmentation campaigns")

if __name__ == "__main__":
    main()
