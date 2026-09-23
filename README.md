# 📈 Retail Sales Forecasting Analytics (Walmart Dataset)

An end-to-end Time-Series Analytics and Machine Learning project designed to predict future sales trends for retail stores using historical sales data and exogenous market indicators.

---

## 👤 Author & Profile
* **Author:** Vishal Jadhav
* **GitHub Profile:** [@vishal-jadhav876](https://github.com/vishal-jadhav876)
* **Repository:** [Sales-Forecasting-Analytics](https://github.com/vishal-jadhav876/Sales-Forecasting-Analytics)

---

## 📌 Project Overview
Accurate sales forecasting enables retail enterprises to optimize inventory management, streamline supply chain logistics, and plan promotional events effectively.

This project analyzes weekly sales records from 45 Walmart stores to perform seasonal decomposition, evaluate macroeconomic factors (such as holiday events, fuel prices, CPI, and unemployment rates), and build a **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors)** time-series forecasting model.

---

## 📂 Dataset Information
* **Source:** Walmart Store Sales Dataset (`Walmart.csv`)
* **Total Records:** 6,435 rows across 45 stores
* **Timeframe:** February 2010 to October 2012 (143 Weekly Aggregates)
* **Key Features:** `Store`, `Date`, `Weekly_Sales`, `Holiday_Flag`, `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Time-Series Modeling:** `statsmodels` (SARIMAX, seasonal_decompose)
* **Evaluation Metrics:** `scikit-learn` (RMSE, MAE, MAPE)

---

## 📊 Key Insights & Results

### Model Performance Metrics
| Metric | Result | Description |
| :--- | :--- | :--- |
| **Model** | **SARIMAX** | Order: `(1,1,1)`, Seasonal Order: `(1,1,0,52)` |
| **MAPE** | **1.77%** | Outstanding forecast accuracy (~98.23% accurate) |
| **MAE** | **$818,242.95** | Mean Absolute Error across total weekly aggregate sales |
| **RMSE** | **$943,400.00** | Root Mean Squared Error |

---

## ⚙️ How to Run the Project Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/vishal-jadhav876/Sales-Forecasting-Analytics.git](https://github.com/vishal-jadhav876/Sales-Forecasting-Analytics.git)
cd Sales-Forecasting-Analytics
