# Telco Customer Churn Machine Learning Pipeline

This project was completed as part of the AI/ML Engineering Internship at DevelopersHub Corporation.

## Project Overview
This project focuses on building a reusable, production-ready machine learning pipeline using Scikit-Learn to predict customer churn for a telecom company. The goal is to automatically analyze customer behavior data, clean it, and determine whether a user is likely to cancel their service, allowing the business to take proactive retention actions.

## Dataset
The dataset used is the Telco Customer Churn dataset, which contains demographic, account, and service information for thousands of real-world customers.
* **Target variable:** `Churn` (Whether the customer left within the last month: Yes/No)
* **Key features used:** * Numerical attributes (Tenure, Monthly Charges, Total Charges)
  * Categorical attributes (Contract type, Payment method, Internet service type, Streaming options, etc.)

## Data Preprocessing
* **Feature Pruning:** Dropped irrelevant unique columns like `customerID` that do not contribute to predictive patterns.
* **Data Cleansing:** Converted string-based `TotalCharges` spaces into numerical formats and handled missing values securely by applying a zero-fill strategy.
* **Target Mapping:** Transformed the categorical target column (`Churn`) from text words ("Yes" / "No") into clean binary integers ($1$ / $0$) using an implicit lambda function.
* **Data Splitting:** Divided the processed dataset into an 80% training pool and a 20% validation pool to strictly isolate testing data.

## Pipeline Architecture & Preprocessing
Built an automated preprocessing conveyor belt using Scikit-Learn’s `ColumnTransformer` and `Pipeline` APIs to eliminate data leakage:
* **Numerical Transformer:** Scaled continuous values using `StandardScaler` to bring variance down to a uniform playing field.
* **Categorical Transformer:** Applied `OneHotEncoder` to translate multi-option text columns into independent binary vectors.
* **Production Guardrail:** Configured `handle_unknown='ignore'` to guarantee that if unrecognized categories appear in real-time server inference, the pipeline stays stable and ignores them instead of throwing system runtime errors.

## Models Used & Hyperparameter Tuning
An optimization tournament was carried out using `GridSearchCV` with a 3-fold cross-validation scheme to compare distinct algorithmic structures and fine-tune their hyperparameters:
* **Logistic Regression Configuration:** Checked inverse regularization strengths ($C = 0.1, 1, 10$) combined with a maximum iteration threshold of 1000.
* **Random Forest Configuration:** Evaluated ensembles consisting of $50$ and $100$ estimators (decision trees) combined with tree depth boundaries restricted to $5$ and $10$ layers to avoid overfitting.

## Evaluation & Results
The automated grid search crowned the **Logistic Regression** model configured with a regularization penalty of $C = 10$ as the optimal model setup.
* **Overall Accuracy:** Achieved a reliable score of **82%** on unseen validation test data.
* **Class 0 (Retained Customers):** Achieved a precision score of 0.86 and a recall score of 0.90.
* **Class 1 (Churned Customers):** Achieved a precision score of 0.67 and a recall score of 0.60.
* **Business Impact:** The pipeline successfully flagged 60% of the entire churning segment before they officially cut ties, giving the marketing team a data-driven window to distribute proactive discount incentives.

## Conclusion & Production Value
This project successfully showcases an end-to-end automated pipeline capable of parsing raw relational data and outputting high-fidelity risk evaluations. 
* **Model Serialization:** Exported the unified pipeline using `joblib` into a singular, portable binary file (`churn_pipeline_model.pkl`).
* **Deployment Readiness:** This saved artifact holds both the structural transformer rules and the final weights of the winning Logistic Regression model. It can be instantly loaded into a live web server or API endpoint to generate real-time inferences without running manual preprocessing workflows again.
