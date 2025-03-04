# Customer Churn Prediction in R

## Project Overview**
This project analyzes customer churn in a telecommunications company using **R**. The goal is to identify key factors that contribute to churn and build predictive models using **logistic regression** and **decision trees**. This will help businesses develop strategies to improve customer retention.

## **Tools & Technologies**
- **RStudio** for scripting and model building
- **Libraries:** `tidyverse`, `caret`, `ggplot2`, `rpart`
- **Data Cleaning:** Handling missing values, duplicates, and formatting
- **EDA:** Visualization of churn trends and customer behavior
- **Modeling:** Logistic Regression & Decision Tree

## Dataset
- Telco Customer Churn Dataset
- **Source:** Publicly available on Kaggle & IBM
- **Rows:** 7,043
- **Columns:** 21
- **Target Variable:** `Churn` (Yes/No)
- **Key Features:** Customer demographics, contract type, monthly charges, tenure, and payment methods.

## **Results & Insights**
- **Key churn predictors:** **Tenure, MonthlyCharges, and Contract Type**.
- **Customers with month-to-month contracts are more likely to churn**.
- **Lower tenure customers have a higher churn rate**.
- **Further improvements**: Use Random Forest or Gradient Boosting for better predictions.

## **Future Improvements**
- Implement Random Forest and XGBoost models.
- Build an interactive Shiny dashboard for churn insights.
- Optimize feature engineering to improve model accuracy.
