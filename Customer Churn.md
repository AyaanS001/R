# Customer Churn Prediction in R

# **Step 1: Introduction**
This project analyzes customer churn in a telecom dataset using **logistic regression** and **decision trees**. The objective is to identify key factors influencing churn and build predictive models.

## **Step 2: Load Required Libraries**
```r
# Install necessary packages (run once if not installed)
install.packages("tidyverse")
install.packages("caret")
install.packages("rpart")

# Load libraries
library(tidyverse)
library(caret)
library(rpart)
```

## **Step 3: Import Dataset**
```r
# Set working directory (modify accordingly)
setwd("path/to/your/project")

# Load dataset
churn_data <- read.csv("WA_Fn-UseC_-Telco-Customer-Churn.csv", stringsAsFactors = TRUE)

# Preview data
head(churn_data)
```

## **Step 4: Data Assaying**
```r
# Check structure of the dataset
str(churn_data)

# Summary statistics
summary(churn_data)

# Check for missing values
colSums(is.na(churn_data))
```
```r
' data.frame':  7043 obs. of  21 variables:
 $ customerID      : Factor w/ 7043 levels "0002-ORFBO","0003-MKNFE",..: 5376 3963 2565 5536 6512 ...
 $ gender          : Factor w/ 2 levels "Female","Male": 1 2 2 2 1 ...
 $ SeniorCitizen   : int  0 0 0 0 0 ...
 $ Partner         : Factor w/ 2 levels "No","Yes": 2 1 1 1 1 ...
 $ Dependents      : Factor w/ 2 levels "No","Yes": 1 1 1 1 1 ...
 $ tenure          : int  1 34 2 45 2 ...
 $ PhoneService    : Factor w/ 2 levels "No","Yes": 1 2 2 1 2 ...
 $ MultipleLines   : Factor w/ 3 levels "No","No phone service",..: 2 1 1 2 1 ...
 $ InternetService : Factor w/ 3 levels "DSL","Fiber optic",..: 1 1 1 1 2 ...
 $ OnlineSecurity  : Factor w/ 3 levels "No","No internet service",..: 1 3 3 3 1 ...
 $ OnlineBackup    : Factor w/ 3 levels "No","No internet service",..: 3 1 3 1 1 ...
 $ DeviceProtection: Factor w/ 3 levels "No","No internet service",..: 1 3 1 3 1 ...
 $ TechSupport     : Factor w/ 3 levels "No","No internet service",..: 1 1 1 3 1 ...
 $ StreamingTV     : Factor w/ 3 levels "No","No internet service",..: 1 1 1 1 1 ...
 $ StreamingMovies : Factor w/ 3 levels "No","No internet service",..: 1 1 1 1 1 ...
 $ Contract        : Factor w/ 3 levels "Month-to-month",..: 1 2 1 2 1 ...
 $ PaperlessBilling: Factor w/ 2 levels "No","Yes": 2 1 2 1 2 ...
 $ PaymentMethod   : Factor w/ 4 levels "Bank transfer (automatic)",..: 3 4 4 1 3 ...
 $ MonthlyCharges  : num  29.9 57 53.9 42.3 70.7 ...
 $ TotalCharges    : num  29.9 1889.5 108.2 1840.8 151.7 ...
 $ Churn           : Factor w/ 2 levels "No","Yes": 1 1 2 1 2 ...

   customerID      gender     SeniorCitizen    Partner    Dependents     tenure      PhoneService
 0002-ORFBO:   1   Female:3488   Min.   :0.0000   No :3641   No :4933   Min.   : 0.00   No : 682    
 0003-MKNFE:   1   Male  :3555   1st Qu.:0.0000   Yes:3402   Yes:2110   1st Qu.: 9.00   Yes:6361    
 ...
 MultipleLines     InternetService             OnlineSecurity              OnlineBackup 
 No              :3390   DSL        :2421   No                 :3498   No                 :3088  
 No phone service: 682   Fiber optic:3096   No internet service:1526   No internet service:1526  
 Yes             :2971   No         :1526   Yes                :2019   Yes                :2429  
 ...
 Contract    PaperlessBilling                   PaymentMethod 
 Month-to-month:3875   No :2872         Bank transfer (automatic):1544  
 One year      :1473   Yes:4171         Credit card (automatic)  :1522  
 Two year      :1695                    Electronic check         :2365  
 ...
 MonthlyCharges    TotalCharges    Churn     
 Min.   : 18.25   Min.   :  18.8   No :5174  
 1st Qu.: 35.50   1st Qu.: 402.2   Yes:1869  
 Median : 70.35   Median :1397.5             
 Mean   : 64.76   Mean   :2281.9             
 3rd Qu.: 89.85   3rd Qu.:3786.6             
 Max.   :118.75   Max.   :8684.8
```

## **Step 5: Data Cleaning**
```r
# Convert TotalCharges to numeric
churn_data$TotalCharges <- as.numeric(as.character(churn_data$TotalCharges))

# Fill missing values in TotalCharges with median
churn_data$TotalCharges[is.na(churn_data$TotalCharges)] <- median(churn_data$TotalCharges, na.rm = TRUE)

# Remove duplicate customer IDs
churn_data <- churn_data[!duplicated(churn_data$customerID), ]
```
```r
customerID           gender    SeniorCitizen          Partner       Dependents 
               0                0                0                0                0 
          tenure     PhoneService    MultipleLines  InternetService   OnlineSecurity 
               0                0                0                0                0 
    OnlineBackup DeviceProtection      TechSupport      StreamingTV  StreamingMovies 
               0                0                0                0                0 
        Contract PaperlessBilling    PaymentMethod   MonthlyCharges     TotalCharges 
               0                0                0                0                0 
           Churn 
               0
```

## **Step 6: Exploratory Data Analysis (EDA)**
```r
# Load ggplot2 for visualization
library(ggplot2)

# Churn distribution
ggplot(churn_data, aes(x = Churn, fill = Churn)) +
  geom_bar() +
  ggtitle("Customer Churn Distribution")

# Tenure vs. Churn
ggplot(churn_data, aes(x = tenure, fill = Churn)) +
  geom_histogram(binwidth = 5, position = "dodge") +
  ggtitle("Tenure Distribution by Churn Status")
```
<img width="620" alt="Screenshot 2025-03-03 at 6 55 50 PM" src="https://github.com/user-attachments/assets/2661c039-3678-4e8f-ba3e-f5ab12080dcb" />


## **Step 7: Model Building (Logistic Regression & Decision Tree)**
```r
# Convert Churn to factor
churn_data$Churn <- as.factor(churn_data$Churn)

# Split data into training (70%) and testing (30%)
set.seed(123)
trainIndex <- createDataPartition(churn_data$Churn, p = 0.7, list = FALSE)
trainData <- churn_data[trainIndex, ]
testData <- churn_data[-trainIndex, ]

# Train logistic regression model
logit_model <- glm(Churn ~ tenure + MonthlyCharges + Contract, data = trainData, family = "binomial")
summary(logit_model)

# Predict on test data
logit_pred <- predict(logit_model, testData, type = "response")
logit_pred_class <- ifelse(logit_pred > 0.5, "Yes", "No")

# Evaluate model performance
confusionMatrix(as.factor(logit_pred_class), testData$Churn)
```
```r
l:
glm(formula = Churn ~ tenure + MonthlyCharges + Contract, family = "binomial", 
    data = trainData)

Coefficients:
                  Estimate Std. Error z value Pr(>|z|)    
(Intercept)      -1.633451   0.107623 -15.178   <2e-16 ***
tenure           -0.034579   0.002441 -14.169   <2e-16 ***
MonthlyCharges    0.028870   0.001612  17.906   <2e-16 ***
ContractOne year -1.158798   0.123867  -9.355   <2e-16 ***
ContractTwo year -2.095441   0.196762 -10.650   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 5707.1  on 4930  degrees of freedom
Residual deviance: 4280.1  on 4926  degrees of freedom
AIC: 4290.1

Number of Fisher Scoring iterations: 6

> 
> # Predict on test data
> logit_pred <- predict(logit_model, testData, type = "response")
> logit_pred_class <- ifelse(logit_pred > 0.5, "Yes", "No")
> 
> # Confusion matrix to evaluate model performance
> confusionMatrix(as.factor(logit_pred_class), testData$Churn)
Confusion Matrix and Statistics

          Reference
Prediction   No  Yes
       No  1394  286
       Yes  158  274
                                         
               Accuracy : 0.7898         
                 95% CI : (0.7718, 0.807)
    No Information Rate : 0.7348         
    P-Value [Acc > NIR] : 2.764e-09      
                                         
                  Kappa : 0.418          
                                         
 Mcnemar's Test P-Value : 1.669e-09      
                                         
            Sensitivity : 0.8982         
            Specificity : 0.4893         
         Pos Pred Value : 0.8298         
         Neg Pred Value : 0.6343         
             Prevalence : 0.7348         
         Detection Rate : 0.6600         
   Detection Prevalence : 0.7955         
      Balanced Accuracy : 0.6937         
                                         
       'Positive' Class : No
```

## **Step 8: Conclusion**
- Key features influencing churn: **tenure, monthly charges, and contract type**.
- Logistic regression provides insights into how these factors contribute to customer retention.
- Future steps: Improve model performance using **random forests** or **ensemble learning**.

This concludes our analysis of customer churn prediction using R! 🚀
