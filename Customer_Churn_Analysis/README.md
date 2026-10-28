Telco Customer Churn Prediction
Project Overview
Customer churn is a major challenge for subscription-based businesses because losing customers can affect recurring revenue and long-term growth.
This project uses machine learning to identify telecom customers who are likely to leave the service. It follows an end-to-end classification workflow, from data preparation and feature selection to model comparison, threshold tuning and final evaluation on an unseen test set.
The project also explores how different classification thresholds can support different customer-retention strategies, depending on whether a business prioritizes identifying more potential churners or reducing unnecessary customer outreach.
Objectives
•	Analyze customer characteristics and churn patterns.
•	Prepare data for machine learning.
•	Compare multiple feature-selection techniques.
•	Train and evaluate classification models.
•	Tune the decision threshold using validation data.
•	Compare higher-recall and more balanced prediction strategies.
•	Translate model results into actionable business considerations.
Dataset
Dataset: IBM Telco Customer Churn
The dataset contains 7,043 customer records and information about customer demographics, services, contracts, payment methods, tenure, and billing.
The target variable is Churn:
•	Yes — the customer churned.
•	No — the customer did not churn.
The original dataset contains 7,043 records, with approximately 26.5% of customers labelled as churners and 73.5% as non-churners.
Dataset source: IBM Telco Customer Churn dataset, commonly distributed through IBM sample data repositories and public machine-learning repositories. See the source and usage terms of the specific dataset file included or referenced in this project.
Tools and Technologies
•	Python
•	Jupyter Notebook
•	Pandas and NumPy — data manipulation and numerical operations
•	Scikit-learn — preprocessing, feature selection, model training and evaluation
•	Matplotlib / Seaborn — exploratory data analysis and visualization
Project Workflow
1. Data Preparation
•	Inspected the dataset structure and variable types.
•	Checked for duplicate records and missing data.
•	Separated the target variable from the input features.
•	Excluded customer identifiers from model training.
•	Split the data into training, validation and test sets using stratified sampling.
The test set was reserved for the final evaluation after model and threshold selection.
2. Preprocessing
•	Imputed missing numerical values using the median.
•	Imputed missing categorical values using the most frequent category.
•	Standardized numerical features for Logistic Regression.
•	Applied one-hot encoding to categorical features.
The preprocessing transformations were fitted on the training data and reused for validation and test data.
3. Feature Selection
Several feature-selection approaches were compared to understand whether a smaller feature set could maintain or improve predictive performance.
Method	Purpose
All Features	Establish a baseline using all 45 encoded features
Mutual Information	Identify features with strong statistical relationships with churn
L1 Regularization	Select features using non-zero coefficients
RFECV	Recursively eliminate features using cross-validation
Random Forest Feature Importance	Rank features using a Random Forest model
Permutation Importance	Measure performance changes when feature values are shuffled
Gradient Boosting Feature Importance	Rank features using a Gradient Boosting model
Gradient Boosting Permutation Importance	Evaluate feature contributions through permutation
Feature selection and model comparisons were conducted using training and validation data. The test set was kept separate until final evaluation.
4. Model Comparison
Three classification algorithms were evaluated:
•	Logistic Regression
•	Random Forest
•	Gradient Boosting
The models were compared using accuracy, precision, recall and F1-score for the churn class (Yes).
Best validation result: Logistic Regression with RFECV feature selection achieved an F1-score of 0.6145 at the default threshold of 0.50, using 36 selected features.
This model was retained for threshold tuning because it performed best on validation F1-score among the evaluated model and feature-selection combinations.
5. Decision-Threshold Tuning
Instead of relying only on the default threshold of 0.50, two alternative thresholds were selected using validation results to represent different business priorities.
•	Threshold 0.45: Prioritizes recall and aims to identify more potential churners.
•	Threshold 0.55: Provides a more balanced trade-off between precision and recall.
The selected model and features remained the same; only the decision threshold changed.




Final Test-Set Results
The selected Logistic Regression + RFECV model was evaluated on the held-out test set of 1,057 customers, using 36 selected features.
Metric	Threshold 0.45	Threshold 0.55
Accuracy	72.19%	75.59%
Precision — Churn = Yes	48.57%	52.72%
Recall — Churn = Yes	85.00%	76.07%
F1-score — Churn = Yes	0.6182	0.6228
Number of selected features	36	36
Confusion Matrix: Threshold 0.45
	Predicted No	Predicted Yes
Actual No	525	252
Actual Yes	42	238
Confusion Matrix: Threshold 0.55
	Predicted No	Predicted Yes
Actual No	586	191
Actual Yes	67	213
Business Interpretation
Strategy 1: Higher Recall — Threshold 0.45
This strategy identified 238 of the 280 actual churners, achieving 85.00% recall.
The model flagged 490 customers for potential retention action. Of these, 252 were false positives: customers who did not churn in the test data.
This strategy may be suitable when missing a potential churner is costly and the business has enough capacity for broader customer outreach.


Strategy 2: More Selective Outreach — Threshold 0.55
This strategy identified 213 of the 280 actual churners, achieving 76.07% recall.
The model flagged 404 customers, including 191 false positives. It achieved higher precision and a slightly higher F1-score than the 0.45 strategy.
This approach may be preferable when retention campaigns have limited budgets, expensive interventions or restricted contact-centre capacity.
Recommended Approach
Threshold 0.55 is the primary recommendation based on its slightly higher test F1-score and higher precision. Threshold 0.45 remains a useful alternative when the business places greater importance on identifying as many potential churners as possible.
The comparison illustrates that a classification threshold should reflect business priorities rather than be selected on accuracy alone.
The final production threshold would require further analysis of customer lifetime value, retention campaign costs, intervention effectiveness and the relative costs of false positives and false negatives.
Key Findings
•	Logistic Regression with RFECV achieved the strongest validation F1-score among the tested model and feature-selection combinations.
•	Feature selection reduced the model input from 45 encoded features to 36.
•	Lowering the decision threshold increased recall but also increased the number of false positives.
•	Threshold 0.45 detected more actual churners, while threshold 0.55 produced a more selective list of customers.
•	Evaluating both strategies connected model performance with practical customer-retention decisions.
Limitations and Future Improvements
•	The results are based on a historical telecom dataset and may not generalize to every business or customer population.
•	The analysis evaluates churn prediction rather than the actual success of retention campaigns.
•	Threshold selection was based on predictive metrics, not a quantified financial cost-benefit analysis.
•	Further work could include probability calibration, cost-sensitive threshold optimization, hyperparameter tuning and monitoring model performance over time.
•	A future project could test whether targeted retention interventions improve customer outcomes through controlled experiments.
Conclusion
This project demonstrates an end-to-end machine-learning workflow for customer churn prediction, including data preparation, feature selection, model comparison, threshold tuning and final test-set evaluation.
Beyond model accuracy, it highlights how classification thresholds can be adjusted to support different business objectives. This provides a practical example of connecting machine-learning outputs with customer-retention planning and resource allocation.
Repository Contents
•	README.md — project overview, methodology, evaluation and findings
•	Jupyter Notebook — data preparation, feature selection, model training, threshold tuning and evaluation
•	Dataset file


