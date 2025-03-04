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

## **Step 5: Data Cleaning**
```r
# Convert TotalCharges to numeric
churn_data$TotalCharges <- as.numeric(as.character(churn_data$TotalCharges))

# Fill missing values in TotalCharges with median
churn_data$TotalCharges[is.na(churn_data$TotalCharges)] <- median(churn_data$TotalCharges, na.rm = TRUE)

# Remove duplicate customer IDs
churn_data <- churn_data[!duplicated(churn_data$customerID), ]
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

## **Step 8: Conclusion**
- Key features influencing churn: **tenure, monthly charges, and contract type**.
- Logistic regression provides insights into how these factors contribute to customer retention.
- Future steps: Improve model performance using **random forests** or **ensemble learning**.

This concludes our analysis of customer churn prediction using R! 🚀
