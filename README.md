# Credit Mix Prediction Using Machine Learning

## 📌 Project Overview

This project focuses on predicting a customer’s Credit Mix category using financial and behavioral data through machine learning techniques. Accurate classification of credit mix is crucial for assessing credit risk, supporting informed lending decisions, and promoting long-term financial well-being.

The project implements an end-to-end machine learning pipeline, covering data preprocessing, model development, hyperparameter tuning, evaluation, and interpretation. Emphasis is placed on handling real-world data challenges such as class imbalance, noisy features, and model explainability.

## 🎯 Objectives

  - Clean and preprocess raw customer financial and behavioral data
  
  - Identify patterns influencing a customer’s credit mix
  
  - Train and evaluate multiple classification models
  
  - Compare models using appropriate metrics for imbalanced multiclass data
  
  - Provide actionable insights to support credit risk decision-making

## 🗂️ Dataset Description

- Target Variable
  
  - Credit_Mix – Categorical credit mix classification
  
- Feature Types
  
  - Financial attributes (income, balances, credit usage, etc.)
  
  - Behavioral indicators
  
  - Categorical and numerical variables
  
- Data Challenges
  
  - Numeric columns stored as strings
  
  - Missing values in key financial features
  
  - Imbalanced class distribution across credit categories

## 🔧 Project Workflow

1️⃣ Data Preprocessing

  - Converted numeric columns safely using pandas
  
  - Handled missing values using group-based median imputation for income-related variables
  
  - Cleaned and standardized categorical features (trimmed spaces, fixed inconsistencies)
  
  - Ensured feature consistency across the dataset

📍 Purpose: Improve data quality and ensure reliable model training

## 2️⃣ Exploratory Data Analysis (EDA)

  - Analyzed class distribution of Credit_Mix
  
  - Explored relationships between financial features and credit categories
  
  - Identified patterns contributing to class imbalance

📍 Purpose: Gain insight into data structure and potential modeling challenges

## 3️⃣ Feature Engineering

Encoded categorical variables

  - Scaled numerical features where required
  
  - Prepared data for both linear and ensemble-based models

📍 Purpose: Enable models to learn meaningful patterns effectively

## 4️⃣ Modeling Approach

The following classification models were trained and evaluated:

  - Logistic Regression
  Used as a baseline due to simplicity and interpretability.

  - Random Forest
  Captured non-linear relationships and feature interactions.

  - Gradient Boosting
  Provided ensemble-based performance with controlled bias–variance tradeoff.

  - XGBoost
  High-performance gradient boosting model optimized for tabular data.

Hyperparameter tuning was performed using RandomizedSearchCV, with macro and weighted F1-score as the primary optimization metrics to properly handle class imbalance.

## 5️⃣ Model Evaluation

Models were evaluated on a hold-out test set using:

  - F1-score (macro & weighted)
  
  - Multiclass ROC-AUC
  
  - Precision and recall analysis

📍 Purpose: Ensure fair evaluation across all credit categories, not just majority classes

## 📊 Results Summary

  - Logistic Regression provided a strong and interpretable baseline but struggled with complex patterns.
  
  - Random Forest improved performance by capturing non-linear feature interactions.
  
  - Gradient Boosting delivered competitive results with careful tuning.
  
  - XGBoost consistently achieved the best performance across weighted F1-score and multiclass ROC-AUC.

✅ XGBoost was selected as the final model, balancing accuracy, robustness, and scalability.

## 🔑 Key Takeaways
  
  - Data preprocessing and feature consistency significantly impact model performance.
  
  - Accuracy alone is insufficient for imbalanced multiclass problems; F1-score and ROC-AUC provide better insight.
  
  - Ensemble methods—especially boosting techniques—are highly effective for credit risk prediction.
  
  - Model explainability (feature importance, SHAP) is critical for translating ML outputs into business insights.

## 💡 Recommendations

  - Use the trained model as a decision-support tool, not a fully automated system.
  
  - Combine model predictions with business rules for high-risk credit decisions.
  
  - Periodically retrain the model to reflect changes in customer behavior.
  
  - Extend the solution by deploying it as a REST API or integrating outputs into a Power BI dashboard for stakeholders.

