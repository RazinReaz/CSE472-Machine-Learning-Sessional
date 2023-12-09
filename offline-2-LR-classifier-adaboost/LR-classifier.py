
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def label_encode_features(dataFrame, features, label_encoder = LabelEncoder()):
    for feature in features:
        dataFrame[feature] = label_encoder.fit_transform(dataFrame[feature])
    return dataFrame

def one_hot_encode_features(dataFrame, feature_names, one_hot_encoder = OneHotEncoder(sparse_output=False)):
    for feature in feature_names:
        encoded = one_hot_encoder.fit_transform(dataFrame[feature].values.reshape(-1, 1)).astype(np.int64)
        encoded_df = pd.DataFrame(encoded)
        encoded_df.columns = [feature + '_' + str(i) for i in range(encoded.shape[1])]
        encoded_df.index = dataFrame.index
        dataFrame = dataFrame.drop(feature, axis=1)
        dataFrame = pd.concat([dataFrame, encoded_df], axis=1)
    return dataFrame

def remove_outliers(dataFrame, columns):
    for column in columns:
        Q1 = np.percentile(dataFrame[column], 25)
        Q3 = np.percentile(dataFrame[column], 75)
        IQR = Q3 - Q1
        step = 1.5 * IQR
        # Remove the outliers
        dataFrame = dataFrame[(dataFrame[column] >= Q1 - step) & (dataFrame[column] <= Q3 + step)]
    return dataFrame

def preprocess_telco_data():
    data = pd.read_csv('datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    data.drop('customerID', axis=1, inplace=True)

    data['TotalCharges'] = data['TotalCharges'].replace(' ', np.nan)
    data.dropna(inplace=True)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])

    label_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', "Churn"]
    one_hot_features = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']


    data = label_encode_features(data, label_features)
    data = one_hot_encode_features(data, one_hot_features)
    data = remove_outliers(data, numeric_features)

    X = data.drop(['Churn'], axis=1)
    y = pd.DataFrame(data['Churn'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=74, stratify=y)

    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.fit_transform(X_test[numeric_features])

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values
    return X_train, X_test, y_train, y_test

def preprocess_adult_data():

    dataset_path = 'datasets/'
    data_column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
                        'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    data_train = pd.read_csv(dataset_path + 'adult.data', names=data_column_names)
    data_test = pd.read_csv(dataset_path + 'adult.test', names=data_column_names)
    data_test = data_test.drop(0, axis=0)

    # remove spaces from the beginning and end of the values
    data_train = data_train.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    data_test = data_test.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # see if there are any missing values
    # for column in data.columns:
    #     if data[column].isnull().sum() > 0:
    #         print('There are missing values in column ' + column)


    # for column in data.columns:
    #     print(column, '\t', type(data[column].unique()[0]), '\n', data[column].unique(), '\n')

    # remove the rows with '?' values
    data_train = data_train[(data_train != '?').all(axis=1)]
    data_test = data_test[(data_test != '?').all(axis=1)]

    numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    one_hot_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
    label_features =   ['sex', 'income']

    # convert numeric feature to np.float64
    data_train[numeric_features] = data_train[numeric_features].astype(np.float64)
    
    data_train = pd.get_dummies(data_train, columns=one_hot_features, drop_first=True, dtype=np.int64)
    data_test = pd.get_dummies(data_test, columns=one_hot_features, drop_first=True, dtype=np.int64)
    
    data_train = data_train.drop('native-country_Holand-Netherlands', axis=1)

    data_train = label_encode_features(data_train, label_features)
    data_test = label_encode_features(data_test, label_features)

    data_test['age'] = pd.to_numeric(data_test['age'])
    # remove outliers of age and education-num column. other columns dont seem to have outliers
    # data_train = remove_outliers(data_train, ['age', 'education-num'])
    X_train = data_train.drop('income', axis=1)
    X_test = data_test.drop('income', axis=1)
    y_train = pd.DataFrame(data_train['income'])
    y_test = pd.DataFrame(data_test['income'])

    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values
    
    return X_train, X_test, y_train, y_test

def preprocess_credit_data():
    data = pd.read_csv('datasets/creditcard.csv')
    data = data.drop('Time', axis = 1)

    X = data.drop('Class', axis = 1)
    y = pd.DataFrame(data['Class'])

    scalar = StandardScaler()
    scalar = scalar.fit(X)

    ros = RandomUnderSampler(random_state=0, sampling_strategy=0.025)
    X_sample, y_sample = ros.fit_resample(X, y)

    X_sample = pd.DataFrame(X_sample, columns=X.columns)
    y_sample = pd.DataFrame(y_sample, columns=y.columns)


    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state = 74)

    X_train_scaled = scalar.transform(X_train)
    X_test_scaled = scalar.transform(X_test)

    y_train = y_train.values
    y_test = y_test.values

    return X_train_scaled, X_test_scaled, y_train, y_test

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegressionModel():
    def __init__(self, 
                 top_features_k = 10, 
                 GD_threshold = 0.5, 
                 initial_learning_rate = 0.1, 
                 total_epochs=1000, 
                 verbose=False,
                 strategy='vanilla',
                 decay_strategy='inverse',
                 select_features=False):
        self.init_lr = initial_learning_rate
        self.top_features_k = top_features_k
        self.GD_threshold = GD_threshold
        self.total_epochs = total_epochs
        self.verbose = verbose
        self.strategy = strategy
        self.decay_strategy = decay_strategy
        self.weights = None # shape (n_features + 1, 1), this contains the bias too

        self.selected_feature_indices = None
        self.fitted = False
        self.select_features = select_features

    def loss(self, y, y_predictions):
        epsilon = 1e-5
        return -np.mean(y * np.log(y_predictions + epsilon) + (1 - y) * np.log(1 - y_predictions + epsilon))
    
    def entropy(labels):
        class_counts = Counter(labels)
        num_samples = len(labels)
        entropy = 0
        for count in class_counts.values():
            if count == 0: continue
            p = count / num_samples
            entropy -= p * np.log2(p)
        return entropy
    
    def verbose_print(self, message):
        if self.verbose:
            print(message)

    def learning_rate_scheduler(self, epoch):
        if self.decay_strategy == 'exponential':
            decay_rate = 2
            return self.init_lr * np.exp(-decay_rate * epoch)
        elif self.decay_strategy == 'inverse':
            return self.init_lr / (1 + self.init_lr * epoch)
        elif self.decay_strategy == 'step':
            decay_factor = 0.5
            decay_epochs = 10
            if epoch % decay_epochs == 0 and epoch:
                return self.init_lr * decay_factor
            return self.init_lr
        else:
            raise NotImplementedError(f"Decay strategy: {self.decay_strategy} is not implemented")
        
    def select_top_features(self, X, y, k):
        if not self.select_features:
            return X
        mutual_info_scores = mutual_info_classif(X, y.ravel())
        selector = SelectKBest(mutual_info_classif, k=k)
        X_top_features = selector.fit_transform(X, y.ravel())
        self.selected_feature_indices = selector.get_support(indices=True)
        return X_top_features

    def fit(self, X, y):
        self.verbose_print("Top {} features are selected".format(self.top_features_k))
        self.verbose_print("GD strategy: {}".format(self.strategy))
        self.verbose_print("Decay strategy: {}".format(self.decay_strategy))

        X = self.select_top_features(X, y, self.top_features_k)
        
        number_of_datapoints, number_of_features = X.shape
        X = np.concatenate((np.ones((number_of_datapoints, 1)), X), axis=1)
        
        if self.strategy == 'vanilla':
            self.weights = np.zeros((number_of_features + 1, 1))
            for epoch in range(self.total_epochs):
                z = X.dot(self.weights)
                y_predictions = sigmoid(z)
                loss = self.loss(y, y_predictions)
                if loss < self.GD_threshold:
                    break
                dw = (1 / number_of_datapoints) * X.T.dot(y_predictions - y)
                self.weights -= self.learning_rate_scheduler(epoch) * dw

        elif self.strategy == 'random':
            best_of = 20
            min_loss = np.inf
            best_weights = None
            self.verbose_print("\nFinding best weights from {} random initializations".format(best_of))
            for _ in range(best_of):
                weights = np.random.randn(number_of_features + 1, 1)
                for epoch in range(self.total_epochs):
                    y_predictions = sigmoid(X.dot(weights))
                    loss = self.loss(y, y_predictions)
                    dw = (1 / number_of_datapoints) * X.T.dot(y_predictions - y)
                    weights -= self.learning_rate_scheduler(epoch) * dw
                    
                if loss < min_loss:
                    self.verbose_print("Found better weights with loss: {}".format(loss))
                    min_loss = loss
                    best_weights = weights
            self.weights = best_weights
            
        elif self.strategy == 'mini':
            batch_size = 16
            self.weights = np.zeros((number_of_features + 1, 1))
            self.verbose_print("Batch size: {}".format(batch_size))
            for epoch in range(self.total_epochs):
                y_predictions = sigmoid(X.dot(self.weights))
                loss = self.loss(y, y_predictions)
                if loss < self.GD_threshold:
                    break
                for batch in range(0, number_of_datapoints, batch_size):
                    X_batch = X[batch:batch+batch_size]
                    y_batch = y[batch:batch+batch_size]
                    y_pred_batch = y_predictions[batch:batch+batch_size]
                    dw = (1 / batch_size) * X_batch.T.dot(y_pred_batch - y_batch)
                    self.weights -= self.learning_rate_scheduler(epoch) * dw
        else:
            raise NotImplementedError(f"Strategy: {self.strategy} is not implemented. Please choose from 'vanilla', 'random' or 'mini'")
        self.fitted = True

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Model is not trained yet")
        if self.select_features:
            X = X[:, self.selected_feature_indices]            
        number_of_datapoints, number_of_features = X.shape
        X = np.concatenate((np.ones((number_of_datapoints, 1)), X), axis=1)
        y_predictions = sigmoid(X.dot(self.weights))
        y_predictions = np.where(y_predictions > 0.5, 1, 0)
        return y_predictions
    
    def accuracy_score(self, y_pred, y):
        return np.mean(y_pred == y)
    
    def report_performance_metrics(self, y_pred, y, mode):
        if y_pred.shape != y.shape:
            raise ValueError("y_pred and y should have the same shape")
        if self.fitted == False:
            raise ValueError("Model is not trained yet")
        
        tp = np.sum(np.logical_and(y_pred == 1, y == 1), axis=0)[0]
        tn = np.sum(np.logical_and(y_pred == 0, y == 0), axis=0)[0]
        fp = np.sum(np.logical_and(y_pred == 1, y == 0), axis=0)[0]
        fn = np.sum(np.logical_and(y_pred == 0, y == 1), axis=0)[0]

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        f1_score = 2 * precision * recall / (precision + recall)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        FDR = fp / (fp + tp)

        print(f"Accuracy\tRecall\t\tSpecificity\tPrecision\tFDR\tF1 Score")
        print("{:.5f}\t\t{:.5f}\t\t{:.5f}\t\t{:.5f}\t\t{:.5f}\t{:.5f}\t\t{}".format(accuracy, recall, specificity, precision, FDR, f1_score, mode))
    def get_weights(self):
        if not self.fitted:
            raise ValueError("Model is not fitted yet")
        return self.weights
    
class Ensemble():
    def __init__(self, n_estimators=10, learning_rate=1.0, verbose=False):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.classifiers = []
        self.classifiers_weights = []
        self.verbose = verbose
        self.boosted = False

    def resample(self, X, y, weights):
        if len(weights) != X.shape[0]:
            raise ValueError("Number of weights should be equal to number of datapoints")
        number_of_datapoints, number_of_features = X.shape
        indices = np.random.choice(number_of_datapoints, size=number_of_datapoints, replace=True, p=weights)
        return X[indices], y[indices]

    def normalize(self, weights):
        return weights / np.sum(weights)
    
    def verbose_print(self, message):
        if self.verbose:
            print(message)

    def AdaBoost(self, X, y, hypotheses):
        if len(hypotheses) != self.n_estimators:
            raise ValueError(f"Number of hypotheses should be equal to {self.n_estimators}")
        number_of_datapoints, number_of_features = X.shape  
        data_weights = np.ones(number_of_datapoints) / number_of_datapoints
        
        data = X.copy()

        i = 0
        while len(self.classifiers) < self.n_estimators:
            if i > len(hypotheses) - 1:
                hypothesis = LogisticRegressionModel(strategy='vanilla',
                                                    decay_strategy='inverse',
                                                    initial_learning_rate=0.1,
                                                    total_epochs=100,
                                                    GD_threshold=0.6,
                                                    top_features_k=5)
            else:
                hypothesis = hypotheses[i]
            if i != 0: 
                data, y = self.resample(data, y, data_weights)
            hypothesis.fit(data, y)
            error = 0
            for j in range(number_of_datapoints):
                if hypothesis.predict(np.expand_dims(data[j], axis = 0)) != y[j]:
                    error += data_weights[j]
            if error > 0.5:
                continue
            error = max(error, 1e-10)

            for j in range(number_of_datapoints):
                if hypothesis.predict(np.expand_dims(data[j], axis = 0)) == y[j]:
                    data_weights[j] *= error / (1 - error)
            data_weights = self.normalize(data_weights)

            self.classifiers_weights.append(0.5 * np.log((1 - error) / error))
            self.classifiers.append(hypothesis)
            i += 1

        self.classifiers_weights = self.normalize(self.classifiers_weights)
        # print("Classifiers weights: {}".format(self.classifiers_weights))
        self.boosted = True
    
    def predict(self, X):
        if not self.boosted:
            raise ValueError("Ensemble model is not boosted yet")
        number_of_datapoints, number_of_features = X.shape
        y_predictions = np.zeros((number_of_datapoints, 1))
        for i, classifier in enumerate(self.classifiers):
            prediction = classifier.predict(X)
            prediction = np.where(prediction == 0, -1, prediction)
            y_predictions += self.classifiers_weights[i] * prediction
        y_predictions = np.where(y_predictions > 0, 1, 0)
        return y_predictions
    
    def accuracy_score(self, y_pred, y):
        return np.mean(y_pred == y)
    
    def report_performance_metrics(self, y_pred, y, mode):
        if y_pred.shape != y.shape:
            raise ValueError("y_pred and y should have the same shape")
        if self.boosted == False:
            raise ValueError("Model is not fitted yet")
        
        tp = np.sum(np.logical_and(y_pred == 1, y == 1), axis=0)[0]
        tn = np.sum(np.logical_and(y_pred == 0, y == 0), axis=0)[0]
        fp = np.sum(np.logical_and(y_pred == 1, y == 0), axis=0)[0]
        fn = np.sum(np.logical_and(y_pred == 0, y == 1), axis=0)[0]

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        f1_score = 2 * precision * recall / (precision + recall)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        FDR = fp / (fp + tp)

        print(f"Accuracy\tRecall\t\tSpecificity\tPrecision\tFDR\tF1 Score\tmode")
        print("{:.5f}\t\t{:.5f}\t\t{:.5f}\t\t{:.5f}\t\t{:.5f}\t{:.5f}\t\t{}\t{}".format(accuracy, recall, specificity, precision, FDR, f1_score, mode, self.n_estimators))

if __name__ == '__main__':
    dataset = 'telco'
    X_train, X_test, y_train, y_test = None, None, None, None
    if dataset == 'telco':
        X_train, X_test, y_train, y_test = preprocess_telco_data()
    elif dataset == 'adult':
        X_train, X_test, y_train, y_test = preprocess_adult_data()
    elif dataset == 'credit':
        X_train, X_test, y_train, y_test = preprocess_credit_data()
    else:
        raise ValueError("Dataset should be either 'telco', 'adult' or 'credit'")

    print("Dataset: {}".format(dataset))
    classifiers = LogisticRegressionModel(strategy='vanilla', 
                                            decay_strategy='inverse',
                                            initial_learning_rate=0.1,
                                            total_epochs=1000,
                                            GD_threshold=0.5,
                                            top_features_k=5)
    classifiers.fit(X_train, y_train)
    y_pred = classifiers.predict(X_train)
    classifiers.report_performance_metrics(y_pred, y_train, "Train")
    y_pred = classifiers.predict(X_test)
    classifiers.report_performance_metrics(y_pred, y_test, "Test")

    for K in [5, 10, 15, 20]:
        classifiers = []
        for k in range(K):
            classifier = LogisticRegressionModel(strategy='vanilla', 
                                            decay_strategy='inverse',
                                            initial_learning_rate=0.1,
                                            total_epochs=100,
                                            GD_threshold=0.6,
                                            top_features_k=5)
            classifiers.append(classifier)

        ensemble = Ensemble(n_estimators=K, verbose=True)
        ensemble.AdaBoost(X_train, y_train, classifiers)

        y_pred = ensemble.predict(X_train)
        ensemble.report_performance_metrics(y_pred, y_train, "Train")
        y_pred = ensemble.predict(X_test)
        ensemble.report_performance_metrics(y_pred, y_test, "Test")

    
    

        

    
