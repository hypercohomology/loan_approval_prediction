# Loan Approval Prediction

This project explores predictive modeling for loan approval status using a combination of machine learning algorithms and strategies for handling class imbalance. 

The target, "loan status" is represented in the dataset as a binary outcome with values 0 (loan denied) and 1 (loan approved). Hence, this a supervised learning problem; specifically, binary classification.

[Kaggle Competition (playground series)](https://www.kaggle.com/competitions/playground-series-s4e10/overview)

## Project Overview

* **Objective:** Develop a reliable model to predict loan approval based on applicant features.
* **Approach:** Multiple machine learning models were trained and evaluated, including XGBoost, Random Forest, and Logistic Regression.
* **Challenge:** Managing class imbalance within the dataset to enhance model generalizability.

### Evaluation Metric

Submissions are evaluated using area under the receiver operating characteristic curve (ROC AUC) using the predicted probabilities and the ground truth targets.

  
### Dataset

The dataset includes features covering applicant demographics, financial history, and categorical information on application status. The imbalance in the target class, with a high number of approved vs. denied loans, necessitated data sampling methods to ensure fair model training.

The dataset for this competition (both train and test) was generated from a deep learning model trained on the [Loan Approval Prediction](https://www.kaggle.com/datasets/chilledwanker/loan-approval-prediction) dataset. Feature distributions are close to, but not exactly the same, as the original. 

### Submission File
For each id row in the test set, you must predict target loan_status. The file should contain a header and have the following format:
```
id,loan_status
58645,0.5
58646,0.5
58647,0.5
etc.
```

### Files
* **train.csv**- the training dataset; ```loan_status``` is the binary target.
* **test.csv**- the test dataset; your objective is to predict the probability of the target ```loan_status``` for each row
* **sample_submission.csv**- a sample submission file in the correct format.
* **my_submission_loans.ipynb**- the main notebook containing data preprocessing steps, model selection/training, and final predictions.
* **SMOTE_v_ADASYN.ipynb**- comparing two separate oversampling techniques (SMOTE and ADASYN) to fix class imbalance in the target. Unfortunately, the difference between these approaches was not statistically significant (determined by chi-squared test)!
* **submission.csv**- final submission file in the correct format.

## Project Steps

### Data Preprocessing:
* One-hot encoding for categorical variables.
* Numerical variables were standardized to improve model convergence.
### Model Selection and Training:
* XGBoost, Random Forest, and Logistic Regression were selected for model comparison, chosen based on their applicability to classification tasks and ability to handle feature interactions.
* Each model was evaluated using Stratified K-Fold cross-validation to ensure consistency across classes.
### Evaluation Metrics:
* ROC AUC was the primary evaluation metric, selected for its sensitivity to class imbalance.
* Performance was tracked across training and cross-validation sets, highlighting the strengths of each model.
### Sampling Techniques:
Given the imbalance in the target variable, SMOTE and ADASYN were applied to the training data in separate runs to assess their impact on model performance and predicted distribution.

## Results

* XGBoost outperformed other models, achieving the highest ROC AUC scores across training and cross-validation, indicating its robustness in this classification task.
* The models showed consistent performance with both SMOTE and ADASYN; however, XGBoost’s higher AUC scores on the cross-validation sets pointed to its effectiveness with the applied sampling methods.

## Conclusion

This analysis highlights the importance of testing multiple models and sampling techniques when dealing with imbalanced data. Although XGBoost emerged as the most effective model, further exploration with other sampling strategies or hyperparameter tuning may help refine performance further.

## Getting Started

### Prerequisites
* Python 3.x
* Libraries: scikit-learn, imbalanced-learn, xgboost, pandas, numpy, matplotlib

### Installation
1. Clone the repository:

bash
```
git clone https://github.com/hypercohomology/loan_approval_prediction.git
cd loan_approval_prediction
```
2. Install dependencies:
bash
```
pip install -r requirements.txt
```
### Running the Project
To replicate the results, run the main notebook in the notebooks/ folder, which details data preprocessing, model training, and evaluation steps. Adjust model parameters within the notebook to explore variations.
