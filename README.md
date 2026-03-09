# Telecom Customer Churn Prediction

## Project Overview

Customer churn is a major challenge in the telecom industry. Retaining existing customers is significantly cheaper than acquiring new ones, making churn prediction an important business problem.

In this project, I built a machine learning pipeline to predict **high-value customer churn** using telecom usage data. The goal is to identify customers who are likely to leave the service provider so that the company can take proactive retention actions.

The project focuses on identifying behavioral signals that appear before a customer stops using the service.

---

## Business Problem

Telecom companies experience an annual churn rate of around **15–25%**. However, a small portion of customers contributes to the majority of revenue.

This project focuses specifically on **high-value customers**, defined as those whose recharge behavior places them in the **top 30% of users based on average recharge amount**.

Predicting churn early allows telecom companies to:

- Launch targeted retention campaigns
- Offer incentives or special plans
- Prevent revenue loss

---

## Dataset Description

The dataset contains telecom usage data across **four months**.

| Month | Phase |
|------|------|
| June (6) | Normal usage |
| July (7) | Normal usage |
| August (8) | Customer behavior begins changing |
| September (9) | Churn phase |

The objective is to **predict churn in September using data from the previous months**.

A customer is labeled as churned if during month 9 they show:

- No incoming calls
- No outgoing calls
- No mobile data usage

---

## Project Workflow

### 1. Exploratory Data Analysis

The dataset contains roughly **100,000 customers and over 200 features** describing telecom usage behavior.

Key observations:

- A large number of features contain missing values related to unused services (such as 2G or 3G data).
- Customer behavior changes gradually before churn rather than abruptly.

---

### 2. Data Preparation

Key preprocessing steps included:

- Handling missing values
- Removing data leakage by dropping all month-9 features
- Filtering only high-value customers
- Creating a churn target variable

This reduced the dataset to approximately **30,000 high-value customers**.

---

### 3. Feature Engineering

Several behavioral features were engineered to capture changes in usage patterns:

Examples:

- Recharge amount decline
- Incoming call decline
- Outgoing call decline

These features help detect **behavioral signals indicating churn risk**.

---

### 4. Model Development

Two machine learning models were developed.

#### Logistic Regression with PCA

PCA was used to reduce dimensionality due to the large number of correlated telecom features.

Performance:

- ROC-AUC: ~0.89
- Recall for churn class: ~0.68

Threshold tuning was applied to balance recall and precision.

---

#### Random Forest Model

A Random Forest classifier was trained without PCA to understand the **drivers of churn**.

Performance:

- ROC-AUC: ~0.93
- Precision: ~0.76
- Recall: ~0.43

Random Forest was used primarily for **feature importance analysis**.

---

## Key Churn Indicators

The most important predictors identified by the model include:

- Local incoming call minutes (Month 8)
- Local outgoing call minutes (Month 8)
- Total call activity (Month 8)
- ARPU in Month 8
- Recharge amount in Month 8
- Decline in call activity
- Decline in recharge amount

These features confirm that **customer engagement decreases before churn occurs**.

---

## Business Insights

Customers who are about to churn typically show:

- Reduced call activity
- Lower recharge amounts
- Declining usage patterns
- Lower ARPU in the final active month

These patterns act as **early warning signals**.

---

## Business Recommendations

Telecom companies can reduce churn by:

- Monitoring usage decline signals
- Running targeted retention campaigns
- Offering personalized recharge incentives
- Building real-time churn monitoring systems

---

## Project Structure

telecom-churn-prediction
│
├── data
│ ├── raw
│ └── processed
│
├── notebooks
│ ├── 01_EDA.ipynb
│ ├── 02_Feature_Engineering.ipynb
│ ├── 03_Modeling_PCA.ipynb
│ └── 04_Modeling_Interpretation.ipynb
│
├── src
│ ├── preprocessing.py
│ ├── feature_engineering.py
│ └── train_model.py
│
├── models
│ └── random_forest_model.pkl
│
├── requirements.txt
└── README.md


---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Future Improvements

Possible extensions of this project include:

- Using advanced models such as XGBoost or LightGBM
- Applying SHAP for model explainability
- Deploying the model using Streamlit
- Building a real-time churn prediction system

---

