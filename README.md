# Customer Churn Prediction in R

## Project Overview
### (The file to this project is right above)
This project analyzes customer churn in a telecommunications company using **R**. The goal is to identify key factors that contribute to churn and build predictive models using **logistic regression** and **decision trees**. This will help businesses develop strategies to improve customer retention.

## **Tools & Technologies**
- **RStudio** for scripting and model building
- **Libraries:** `tidyverse`, `caret`, `ggplot2`, `rpart`
- **Data Cleaning:** Handling missing values, duplicates, and formatting
- **EDA:** Visualization of churn trends and customer behavior
- **Modeling:** Logistic Regression & Decision Tree

## Dataset
- **Telco Customer Churn Dataset**
- **Source:** Publicly available on Kaggle & IBM
- **Rows:** 7,043
- **Columns:** 21
- **Target Variable:** `Churn` (Yes/No)
- **Key Features:** Customer demographics, contract type, monthly charges, tenure, and payment methods.

## **Results & Insights**
- **Key churn predictors:** ** Monthly Charges, and Contract Type**.
- **Customers with month-to-month contracts are more likely to churn**.
- **Lower tenure customers have a higher churn rate**.
- **Further improvements**: Use Random Forest or Gradient Boosting for better predictions.

## **How I Used AI to Improve this Project**
### Step 7 was created using AI, and what did it do?
  - 1️⃣ Splitting the Data
Before training, the dataset is split into two parts:

    - 70% for training – The model learns patterns from this data.
    - 30% for testing – The model is evaluated on this unseen data.
     - Why? This helps test how well the model performs on new data.

- 2️⃣ Logistic Regression Model (Predicting Churn Probability)
- A logistic regression model is trained to predict churn ("Yes" or "No") based on:

  -  tenure (how long a customer has been with the company)
  -  MonthlyCharges (how much they pay per month)
  -  Contract (Month-to-month, One year, or Two years)
    -  **What does this do?:**
    -  Finds relationships between churn and the selected variables.
    -  Outputs coefficients that tell how strongly each feature affects churn.

- 3️⃣ Making Predictions on New Data
- Once the model is trained, it predicts churn probability for the test dataset.
  - If the probability is > 50%, classify as "Yes" (Churn).
  - If it's < 50%, classify as "No" (Stay).

- 4️⃣ Evaluating Model Performance (Confusion Matrix)
- This part checks how well the model performed using a confusion matrix:
    - Results You Might See:
- Predicted "No"	Predicted "Yes"
- Actual "No" (Stayed)	✅ Correct	❌ Wrong
- Actual "Yes" (Churned)	❌ Wrong	✅ Correct
    - High accuracy? → The model is good at predicting churn.
    - Low accuracy? → The model needs improvement (try adding/removing variables)
 
- 🛠️ Decision Tree Model (Alternative Approach)
    - A Decision Tree works differently:

- Instead of calculating probabilities, it splits data into "Yes" and "No" groups.
    - It creates rules like:
    - “If tenure is low & contract is month-to-month → More likely to churn”.
    - This section only builds a logistic regression model, but you could compare it to a decision tree for different insights.

### **Final Takeaways:**
- ✅ We trained a logistic regression model using tenure, contract type, and charges.
- ✅ The model predicts churn probability for new customers.
- ✅ A confusion matrix checks how well the model performed.
- ✅ We could compare this to a Decision Tree for better insights.



