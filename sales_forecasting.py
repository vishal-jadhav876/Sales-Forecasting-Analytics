# ==============================================================================
# PROJECT: Sales Forecasting Analytics 
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ------------------------------------------------------------------------------
# STEP 1: LOAD & INSPECT DATASET
# ------------------------------------------------------------------------------
print("=== Step 1: Loading Dataset ===")
df = pd.read_csv('Walmart.csv')
print(f"Dataset Shape: {df.shape[0]} Rows, {df.shape[1]} Columns")
print(df.head())

# ------------------------------------------------------------------------------
# STEP 2: DATA PREPROCESSING & TIME-SERIES AGGREGATION
# ------------------------------------------------------------------------------
print("\n=== Step 2: Data Preprocessing & Aggregation ===")

# Convert Date to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Aggregate Weekly_Sales across all stores per date
daily_sales = df.groupby('Date').agg({
    'Weekly_Sales': 'sum',
    'Holiday_Flag': 'max',
    'Temperature': 'mean',
    'Fuel_Price': 'mean',
    'CPI': 'mean',
    'Unemployment': 'mean'
}).sort_index()

# Set weekly frequency (Weekly-Friday)
daily_sales = daily_sales.asfreq('W-FRI')

print("Aggregated Weekly Records:", daily_sales.shape[0])

# ------------------------------------------------------------------------------
# STEP 3: TIME-SERIES DECOMPOSITION (EDA)
# ------------------------------------------------------------------------------
print("\n=== Step 3: Generating Decomposition Plot ===")

decomposition = seasonal_decompose(daily_sales['Weekly_Sales'], model='additive', period=52)
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# STEP 4: TRAIN-TEST SPLIT (LAST 12 WEEKS FOR TESTING)
# ------------------------------------------------------------------------------
print("\n=== Step 4: Train-Test Split ===")

train = daily_sales.iloc[:-12].copy()
test = daily_sales.iloc[-12:].copy()

print(f"Training Set : {len(train)} weeks ({train.index.min().strftime('%Y-%m-%d')} to {train.index.max().strftime('%Y-%m-%d')})")
print(f"Testing Set  : {len(test)} weeks ({test.index.min().strftime('%Y-%m-%d')} to {test.index.max().strftime('%Y-%m-%d')})")

# ------------------------------------------------------------------------------
# STEP 5: MODEL TRAINING & FORECASTING (SARIMAX)
# ------------------------------------------------------------------------------
print("\n=== Step 5: Training SARIMAX Model ===")

# Features used as exogenous inputs
exog_cols = ['Holiday_Flag', 'Fuel_Price', 'Unemployment']

model = SARIMAX(
    train['Weekly_Sales'],
    exog=train[exog_cols],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 0, 52)
)
results = model.fit(disp=False)

# Forecast on test dataset
forecast = results.predict(
    start=len(train),
    end=len(train) + len(test) - 1,
    exog=test[exog_cols]
)

test['Forecast'] = forecast.values

# ------------------------------------------------------------------------------
# STEP 6: EVALUATION METRICS
# ------------------------------------------------------------------------------
print("\n=== Step 6: Evaluation Metrics ===")

rmse = np.sqrt(mean_squared_error(test['Weekly_Sales'], test['Forecast']))
mae = mean_absolute_error(test['Weekly_Sales'], test['Forecast'])
mape = np.mean(np.abs((test['Weekly_Sales'] - test['Forecast']) / test['Weekly_Sales'])) * 100

print(f"Root Mean Squared Error (RMSE) : ${rmse:,.2f}")
print(f"Mean Absolute Error (MAE)      : ${mae:,.2f}")
print(f"Mean Absolute Percentage Error : {mape:.2f}%")

# ------------------------------------------------------------------------------
# STEP 7: VISUALIZATION (FORECAST VS ACTUAL)
# ------------------------------------------------------------------------------
print("\n=== Step 7: Generating Forecast Plot ===")

plt.figure(figsize=(12, 6))
plt.plot(train.index[-30:], train['Weekly_Sales'][-30:], label='Historical Sales (Train)', color='blue')
plt.plot(test.index, test['Weekly_Sales'], label='Actual Sales (Test)', color='green', marker='o')
plt.plot(test.index, test['Forecast'], label='Forecasted Sales', color='red', linestyle='--', marker='x')

plt.title('Walmart Sales Forecast vs Actual Sales (SARIMAX Model)')
plt.xlabel('Date')
plt.ylabel('Total Weekly Sales ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()