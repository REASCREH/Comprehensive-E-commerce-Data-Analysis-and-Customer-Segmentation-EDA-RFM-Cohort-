# Comprehensive E-commerce Data Analysis and Customer Segmentation (EDA, RFM, Cohort)

## Overview
This Jupyter notebook presents a detailed, end-to-end analysis of a marketing and e-commerce dataset. The primary goal is to perform comprehensive Exploratory Data Analysis (EDA), segment customers using the Recency, Frequency, Monetary (RFM) model, analyze customer behavior over time with Cohort Analysis, and build a **Deep Learning Model** for predictive purposes.

## Data Loading and Preparation (The Data Pipeline)
The analysis utilizes five interconnected datasets, all loaded from CSV files:
* `transactions.csv`: Individual sales transaction records.
* `customers.csv`: Customer demographic and signup information.
* `products.csv`: Product details, including category and pricing.
* `campaigns.csv`: Details on marketing campaigns.
* `events.csv`: Loaded but used primarily for advanced analyses (not directly in the main summary statistics).

### Data Merging and Preprocessing
1.  The primary DataFrame (`df`) is created by merging:
    * `transactions` and `customers` on **`customer_id`** (Left Join).
    * The resulting DataFrame and `products` on **`product_id`** (Left Join).
2.  Date columns (`timestamp`, `signup_date`, and `launch_date`) are converted to **datetime objects**.
3.  **One-Hot Encoding** is applied to categorical features (e.g., `country`, `loyalty_tier`, `category`) for use in the machine learning model.

***

## Statistical Analysis and Modeling

### 1. Exploratory Data Analysis (EDA)
| Analysis Type | Metrics / Methods Used | Key Insights |
| :--- | :--- | :--- |
| **Data Quality** | Shape, Dtypes, **Missing Value Counts** (identified in `product_id`, `gross_revenue`, `category`, `brand`, `base_price`, `launch_date`, `is_premium`). | Missing revenue and product details due to un-matched `product_id`. |
| **Numerical Features** | `describe()` with **Skewness** and **Kurtosis** for all numerical columns. | `quantity` and `gross_revenue` are highly skewed, indicating high-value outliers. |
| **Categorical Features** | **Value Counts** and **Normalized Proportions**. | **US (35%)** and **IN (20%)** are top markets. **Bronze (54%)** is the most common loyalty tier. |
| **Correlation Analysis** | **Correlation Matrix** (`corr()`) for all numeric features. | Strong positive correlation between **`base_price` and `is_premium` (0.74)**. |
| **Aggregations** | Detailed `sum`, `mean`, `median`, `std`, and `count` aggregations of **`gross_revenue`** and **`quantity`**, grouped by all major categorical features. | **Electronics** leads in revenue ($3.45M) and quantity (29,194 units). |
| **Refund Rate** | Calculated as **Total Refunds / Total Transactions**. | Overall **Refund Rate is 3%**. |

### 2. Derived Models
| Model / Technique | Methodology | Output |
| :--- | :--- | :--- |
| **RFM Model (Customer Segmentation)** | Calculates **Recency**, **Frequency**, and **Monetary** for each unique customer. | A DataFrame with RFM scores for segmentation. |
| **Cohort Analysis** | Groups customers by **`signup_month`** and tracks their activity in subsequent months. | A pivot table showing **Customer Retention Rates** over time. |
| **Time Series Decomposition** | Applies `statsmodels.tsa.seasonal.seasonal_decompose` on the daily revenue time series. | Decomposes daily revenue into **Trend**, **Seasonality** (period=30 days), and **Residual** components (Additive Model). |

***

## Machine Learning Model Training (Customer Value Prediction)

A **Deep Learning model** using Keras/TensorFlow is implemented to predict a target variable (likely **Customer Lifetime Value (CLV)** or **Future Transaction Status**).

### 1. Data Preparation for ML
* **Feature Engineering:** The notebook aggregates transaction data to create customer-level features (e.g., total purchases, average basket size, time since last purchase).
* **Feature Selection:** High-correlation features are reviewed and potentially reduced.
* **Data Scaling:** The full feature set (`X`) is split into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`). All numerical features are scaled using a **StandardScaler** or **MinMaxScaler** to ensure convergence and prevent dominance by large-magnitude features.

### 2. Model Architecture
The model uses a **Sequential** architecture, consisting of a simple Feed-Forward Neural Network (FNN).
* **Input Layer:** `Dense(128, input_dim=X_train_scaled.shape[1], activation='relu')`
* **Hidden Layer:** `Dense(64, activation='relu')`
* **Output Layer:** `Dense(1, activation='sigmoid')` (Assuming a classification task, e.g., predicting customer churn/re-purchase).

### 3. Model Compilation and Training
* **Compiler:**
    * **Optimizer:** `Adam(learning_rate=0.001)`
    * **Loss Function:** `binary_crossentropy` (for the assumed binary classification task)
    * **Metrics:** `accuracy`
* **Training:**
    * The model is trained using the `.fit()` method.
    * **Epochs:** 100
    * **Batch Size:** 32
    * **Validation:** A portion of the training data (e.g., `validation_split=0.2`) is used for real-time validation to monitor overfitting.

### 4. Evaluation
* Model performance is evaluated on the unseen `X_test` data using `model.evaluate()`, resulting in a final **Test Loss** and **Test Accuracy** score.

***

## Visualization Catalogue
All visualizations were created using the **Plotly** library (`plotly.express` for quick charts and `plotly.graph_objects` for advanced models).

### Key Visualization Sections
* **Temporal and Revenue Analysis** (e.g., Line Chart of Daily Revenue, Revenue by Month).
* **Quantity and Discount Analysis** (e.g., Scatter Plot of Discount vs. Gross Revenue).
* **Campaign and Refund Analysis** (e.g., Bar Chart of Refunds by Category).
* **Customer and Product Analysis** (e.g., Customer Age Distribution Histogram, Revenue by Loyalty Tier).
* **Advanced Models and Diagnostics** (e.g., Correlation Heatmap, Time Series Decomposition Plots).
* **Model Training Diagnostics (New):**
    * **fig35 (New):** Line Chart showing **Training Loss and Validation Loss** per Epoch.
    * **fig36 (New):** Line Chart showing **Training Accuracy and Validation Accuracy** per Epoch.
    * **fig37 (New):** Confusion Matrix Heatmap for Test Set Predictions.
