# *Data Preprocessing*, *Logistic Regression* and *AdaBoost* from scratch
In this assignment, we had to implement logistic regression and adaboost from scratch and train them on 3 different datasets. The datasets were preprocessed and the models were trained and tested on them. The task was to
- Preprocess the data
    - Remove outliers
    - Handle missing values
    - Handle imbalanced data
    - Scale the data
    - Encode the categorical data
- Implement logistic regression and adaboost from scratch
- Train and test the models on the datasets
    - With Logistic Regression
    - With AdaBoost (Ensemble of weak Logistic Regression models)
- Compare the results with the results Between the Logistic Regression and AdaBoost model

## Datasets
- [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [Adult](https://archive.ics.uci.edu/dataset/2/adult)
- [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
### How to set up the dataset

The dataset should be arranged in the following way:

```
1805074.pdf
1805074.py
datasets
├──adult.data
├──adult.names
├──adult.test
├──creditcard.csv
├──Index
├──old.adult.names
├──WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Telco Customer Churn Dataset Performance
| Name | Value |
|----------|----------|
| Gradent Descent strategy    | Vanilla    |
| Decay strategy    | inverse    |
| Error Threshold    | 0.5    |
| Epochs    | 1000    |
| Top features    | 5    |
| Initial learning rate  | 0.1    |

### Weak Learner Performance
| Performance Measure | Training | Test |
|---------------------|----------|------|
| Accuracy            |    0.73813      |   0.73774   |
| Recall              |    0.01672      |   0.01604   |
| Specificity         |    0.99927      |   0.99903   |
| Precision           |    0.89286      |   0.85714   |
| FDR                 |    0.10714      |   0.14286   |
| F1 Score            |    0.03283      |   0.03150   |
                                     

### AdaBoost Performance
| Number of boosting rounds | Training | Test |
|-------------------|----------|------|
| K = 5             |  0.78293   |  0.78252   |
| K = 10            |  0.77387   |  0.78109   |
| K = 15            |  0.75484   |  0.75551   |
| K = 20            |  0.75751   |  0.75480   |

## Adult Dataset Performance
| Name | Value |
|----------|----------|
| Gradent Descent strategy    | Vanilla    |
| Decay strategy    | inverse    |
| Error Threshold    | 0.5    |
| Epochs    | 1000    |
| Top features    | 5    |
| Initial learning rate  | 0.1    |

### Weak Learner Performance
| Performance Measure | Training | Test |
|---------------------|------------|------|
| Accuracy            |  0.76298   |  0.76620  |
| Recall              |  0.05155   |  0.05189  |
| Specificity         |  0.99876   |  0.99886  |
| Precision           |  0.93253   |  0.93659  |
| FDR                 |  0.06747   |  0.06341  |
| F1 Score            |  0.09769   |  0.09834  |


### AdaBoost Performance
| Number of boosting rounds | Training | Test |
|---------------------|----------|------|
| K = 5             |   0.75280       |  0.75684    |
| K = 10            |   0.75280       |  0.75684    |
| K = 15            |   0.75280       |  0.75684    |
| K = 20            |   0.77850       |  0.78035    |

## Credit card Dataset Performance
| Name | Value |
|----------|----------|
| Gradent Descent strategy    | Vanilla    |
| Decay strategy    | inverse    |
| Error Threshold    | 0.5    |
| Epochs    | 1000    |
| Top features    | 5    |
| Initial learning rate  | 0.1    |

### Weak Learner Performance
| Performance Measure | Training | Test |
|---------------------|-----------------|-------------|
| Accuracy            |   0.99504       |   0.99430   |
| Recall              |   0.80856       |   0.76842   |
| Specificity         |   0.99975       |   0.99975   |
| Precision           |   0.98769       |   0.98649   |
| FDR                 |   0.01231       |   0.01351   |
| F1 Score            |   0.88920       |   0.86391   |


### AdaBoost Performance
| Number of boosting rounds | Training | Test |
|-------------------|-------------|-----------|
| K = 5             |  0.99504    |  0.99430  |
| K = 10            |  0.96660    |  0.96580  |
| K = 15            |  0.92675    |  0.92169  |
| K = 20            |  0.93661    |  0.93804  |

## What I had to learn
* When to use min-max scaling and when to use standard scaling
* When to use one hot encoding and when to use label encoding
    * if test data does not hold a categorical value that is present in the training data, one hot encoding will fail
      as is the case with {'native-country_Holand-Netherlands'} in the adult dataset
* Removing outliers and handling imbalanced data 
* The outliers in the Credit Card dataset should not be removed because they are **most likely the frauds** that we are trying to detect



## useful links:
- [Solvers for library models](https://stackoverflow.com/questions/38640109/logistic-regression-python-solvers-definitions/52388406#52388406)
- [Numeric feature scaling](https://scikit-learn.org/stable/modules/preprocessing.html)