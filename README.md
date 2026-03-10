# Sales & Demand Forecasting for Businesses  
**Machine Learning Task 1 – Future Interns (2026)**

## Project Overview
This project builds a **Machine Learning based sales forecasting system** using historical store sales data.  
The goal is to predict future sales trends and present insights that can help businesses make informed decisions.

Sales forecasting is a critical component of business planning. Accurate forecasts help organizations:

- Plan inventory efficiently
- Manage cash flow
- Optimize staffing
- Reduce overstock and stockouts
- Support strategic decision making

This project demonstrates how **Machine Learning can be applied to real business data** to predict future demand.

---

# Objective
The objective of this project is to:

- Analyze historical sales data
- Engineer time-based features
- Train a Machine Learning model to forecast sales
- Evaluate model performance using error metrics
- Visualize forecasts for business stakeholders

---

# Dataset
The dataset used in this project comes from Kaggle:

**Store Sales – Time Series Forecasting**

https://www.kaggle.com/competitions/store-sales-time-series-forecasting

### Key Features in Dataset

| Column | Description |
|------|-------------|
| date | Date of the sales record |
| store_nbr | Store identifier |
| family | Product category |
| sales | Number of items sold |
| onpromotion | Number of items on promotion |

---

# Technologies Used

### Programming Language
- Python 3.10.11

### Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Development Environment
- Jupyter Notebook
- VS Code

---

# Project Workflow

## 1 Data Collection
The dataset was downloaded from Kaggle and loaded into a Pandas DataFrame.

```
df = pd.read_csv("train.csv")
```

---

## 2 Data Cleaning

Steps performed:

- Converted `date` column to datetime
- Checked for missing values
- Filled missing values where required

```
df['date'] = pd.to_datetime(df['date'])
df.fillna(0, inplace=True)
```

---

## 3 Feature Engineering

Machine learning models cannot directly interpret date values.  
Therefore, new **time-based features** were created.

Features generated:

- Year
- Month
- Day
- Day of week

```
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
```

These features allow the model to learn **seasonality and time trends**.

---

## 4 Exploratory Data Analysis

Sales trends were visualized to understand patterns over time.

Example insights:

- Sales fluctuate over time
- Seasonal patterns may exist
- Promotions influence demand

Example visualization:

```
sales_trend = df.groupby('date')['sales'].sum()

plt.figure(figsize=(12,5))
plt.plot(sales_trend)
plt.title("Total Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()
```

---

# Machine Learning Model

## Model Used
Random Forest Regressor

Random Forest was selected because:

- It handles nonlinear relationships well
- It performs well on tabular datasets
- It is robust against overfitting

### Features Used

```
features = ['store_nbr','onpromotion','year','month','day','dayofweek']
```

### Training

```
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
```

---

# Model Evaluation

Two evaluation metrics were used:

### MAE (Mean Absolute Error)

Measures the average difference between predicted and actual values.

```
MAE: 490.24
```

This means predictions are off by about **490 sales units on average**.

---

### RMSE (Root Mean Squared Error)

Penalizes larger errors more heavily.

```
RMSE: 1100.47
```

RMSE indicates that some predictions have larger deviations.

---

# Forecast Visualization

Actual sales and predicted sales were compared using line plots.

```
plt.plot(y_test.values[:200], label="Actual Sales")
plt.plot(predictions[:200], label="Predicted Sales")
plt.legend()
plt.show()
```

These visualizations help **non-technical stakeholders understand forecast trends**.

---

# Business Insights

Sales forecasting provides important value for businesses.

Using this forecasting system, businesses can:

### Inventory Planning
Predict future demand to avoid stock shortages.

### Staffing Optimization
Prepare staffing levels during high-demand periods.

### Promotion Planning
Identify periods where promotions increase demand.

### Financial Forecasting
Estimate revenue trends and manage cash flow.

Example use case:

If the model predicts **higher sales next month**, the business can increase inventory in advance to prevent stockouts.

---

# Project Structure

```
sales-demand-forecasting
│
├── sales_forecasting.ipynb
├── train.csv
├── requirements.txt
├── .gitignore
└── README.md
```

---

# How to Run the Project

### 1 Clone the repository

```
git clone https://github.com/yourusername/sales-demand-forecasting.git
cd sales-demand-forecasting
```

### 2 Create virtual environment

```
python -m venv future
```

### 3 Activate environment

Windows

```
future\Scripts\activate
```

Mac / Linux

```
source future/bin/activate
```

### 4 Install dependencies

```
pip install -r requirements.txt
```

### 5 Run the notebook

```
jupyter notebook
```

Open:

```
sales_forecasting.ipynb
```

---

# Future Improvements

Possible improvements for this project include:

- Implementing advanced time-series models (ARIMA, Prophet)
- Using gradient boosting models such as XGBoost
- Adding lag features for better temporal learning
- Creating a dashboard using Power BI or Tableau
- Performing hyperparameter tuning

---

# Conclusion

This project demonstrates how machine learning can be used to forecast sales and support business decision-making.

By combining **data analysis, feature engineering, machine learning, and visualization**, businesses can gain valuable insights into future demand and improve operational planning.

---

# Author

Machine Learning Task – Future Interns (2026)

Sales & Demand Forecasting Project
