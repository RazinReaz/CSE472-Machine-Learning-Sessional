# CSE472 Machine Learning Sessional
## Summary 
This repository contains the solutions to the assignments of the course CSE472: Machine Learning Sessional. Please refer to the [table of contents](#table-of-contents) for the assignments and **click on the assignment titles** to be guided to the respective READMEs containing the detailed results.
## Table of Contents
- [Assignment 1: *Transformation matrices*, *Eigen Decomposition* and *Low Rank Approximation* using SVD](#assignment-1-transformation-matrices-eigen-decomposition-and-low-rank-approximation-using-svd)
- [Assignment 2: *Data Preprocessing*, *Logistic Regression* and *AdaBoost* from scratch](#assignment-2-data-preprocessing-logistic-regression-and-adaboost-from-scratch)
- [Assignment 3: *Feed Forward Neural Network* from scratch](#assignment-3-feed-forward-neural-network-from-scratch)
- [Assignment 4: *Principal Component Analysis* and *Expectation Maximization (EM) Algorithm* visualization](#assignment-4-principal-component-analysis-and-expectation-maximization-em-algorithm-visualization)

## [Assignment 1: *Transformation matrices*, *Eigen Decomposition* and *Low Rank Approximation* using SVD](offline-1-linear-algebra/README.md)
### Transformation matrices
In this task, we derived the transformation matrix that would transform 2 vectors to 2 other vectors.
### Eigen Decomposition
We generated random invertible matrices and random symmetric invertible matrices and reconstructed them from their eigen decomposition.

### Low Rank Approximation using SVD
We reconstructed a greyscale image from its low rank approximation using SVD. The resulting images with varying K values are shown below:
![Image reconstruction](offline-1-linear-algebra/image_reconstruction.jpg)


## [Assignment 2: *Data Preprocessing*, *Logistic Regression* and *AdaBoost* from scratch](offline-2-LR-classifier-adaboost/README.md)
In this assignment, we had to implement logistic regression and adaboost **from scratch** and train them on 3 different datasets. The datasets were preprocessed and the models were trained and tested on them. The task was to
- Preprocess the data
- Implement logistic regression and adaboost **from scratch**
- Train and test the models on the datasets
    - With Logistic Regression
    - With AdaBoost (Ensemble of weak Logistic Regression models)
- Compare the results with the results Between the Logistic Regression and AdaBoost model

### Datasets
- [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [Adult](https://archive.ics.uci.edu/dataset/2/adult)
- [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

### Telco Customer Churn Dataset Performance
| Name | Value |
|----------|----------|
| Gradent Descent strategy    | Vanilla    |
| Decay strategy    | inverse    |
| Error Threshold    | 0.5    |
| Epochs    | 1000    |
| Top features    | 5    |
| Initial learning rate  | 0.1    |

#### Weak Learner Performance
| Performance Measure | Training | Test |
|---------------------|----------|------|
| Accuracy            |    0.73813      |   0.73774   |
| Recall              |    0.01672      |   0.01604   |
| Specificity         |    0.99927      |   0.99903   |
| Precision           |    0.89286      |   0.85714   |
| FDR                 |    0.10714      |   0.14286   |
| F1 Score            |    0.03283      |   0.03150   |
                                     

#### AdaBoost Performance
| Number of boosting rounds | Training | Test |
|-------------------|----------|------|
| K = 5             |  0.78293   |  0.78252   |
| K = 10            |  0.77387   |  0.78109   |
| K = 15            |  0.75484   |  0.75551   |
| K = 20            |  0.75751   |  0.75480   |



## [Assignment 3: *Feed Forward Neural Network* from scratch](offline-3-fnn/README.md)
In this assignment we had to implement a neural network from scratch and train it on the MNIST alphabet dataset. The trained model architecture and weights had to be saved for later use. The task was to
- Implement a feed forward neural network from scratch with support for
    - Dense Layers
    - Activation layers (ReLU, Sigmoid, Softmax, Tanh)
    - Initializers (Xavier, He, Lecun)
    - Dropout layers
    - Optimizers
- Build different Architectures and train them on the EMNIST dataset
- Report the performance of the models (Accuracy, Precision, Recall, F1 Score, Confusion Matrix)
- Save the model architecture and weights for later use

### Dataset
To download the training dataset, the following code was used:
```python
train_validation_dataset = ds.EMNIST(root='./data', split='letters',
train=True,
transform=transforms.ToTensor(),
download = True)
```
for downloading the test dataset, the following code was used:
```python
independent_test_set = ds.EMNIST(root='./data', split='letters',
train=False,
transform=transforms.ToTensor(),
download = True)
```
### Results 
#### Architecture used
```
Dense Layer: (784,464)
Initializer:Xavier
Activation: Sigmoid
Dropout: 0.7

Dense Layer: (464,348)
Initializer:He
Activation: ReLU
Dropout: 0.52

Dense Layer: (348,26)
Initializer:Xavier
Activation: Softmax

Loss: Cross Entropy Loss
```

#### Performance of the model
| Performance Measure | Value |
|---------------------|-------|
| Accuracy            | 95.29%|
| Validation Accuracy | 92.28%|
| Training Loss       | 0.135 |
| Validation Loss     | 0.233 |
| Validation Macro F1 | 0.922 |

![Accuracy vs epoch](offline-3-fnn/report/1/accuracy.png)
![Loss vs epoch](offline-3-fnn/report/1/loss.png)
![Confusion Matrix](offline-3-fnn/report/1/Training%20Confusion%20Matrix.png)
![Validation Confusion Matrix](offline-3-fnn/report/1/Validation%20Confusion%20Matrix.png)


## [Assignment 4: *Principal Component Analysis* and *Expectation Maximization (EM) Algorithm* visualization](offline-4-pca/README.md)
In this problem, we are given a dataset of **2, 3, 6 and 100 dimensional data** that was generated from a **gaussian mixture model**. \
The task was to 
- **Apply Principal Component Analysis (PCA)** to the dataset and visualize the data in 2D. 
- **Apply Expectation Maximization (EM) algorithm** on the 2 dimensional data to estimate the gaussian mixture model parameters used to generate the data.
    - The EM algorithm was implemented from scratch.
    - For each dataset, a guess was made for how many gaussian components were used to generate the data *from 3 to 8*
    - for each guessed number of components, the EM algorithm was run *5 times for 100 iterations* and the model with the highest log likelihood was chosen.
- **Visualize the EM algorithm** by plotting the data and the estimated gaussian mixture model parameters at each iteration.

### Results

#### 6 Dimensional data
The dataset is in [this](offline-4-pca/data/6D_data_points.txt) folder.\
The PCA reduced 6D data points are shown below:
![6 Dimensional Data Reduced](offline-4-pca/assets/plots/6D_data_points-reduced-plot.jpg)
The visualization of the EM algorithm for *5 components* is shown below:
![Visualization of EM Algorithm](offline-4-pca/assets/gifs/6D_data_points-gmm-5.gif)
the resulting clusters after 100 iterations are shown below:
![5 clusters on this Dataset](offline-4-pca/assets/plots/6D_data_points-gmm-5.jpg)


The other generated plots can be found in these links : 
1. [gifs](offline-4-pca/assets/gifs/).
2. [plots](offline-4-pca/assets/plots/).
3. [log-likehoods](offline-4-pca/assets/log_likelihoods/)
