
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

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

    # No missing values in this dataset.
    # TotalCharges is a numeric column, but it is stored as an object. We need to remove the rows with spaces
    # label encoder and one hot encoders before splitting the data
    # need to scale 3 features (done after splitting)

    data['TotalCharges'] = data['TotalCharges'].replace(' ', np.nan)
    data.dropna(inplace=True)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])


    # for col in data.columns:
    #     print(col, data[col].unique())

    label_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', "Churn"]
    one_hot_features = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']


    data = label_encode_features(data, label_features)
    data = one_hot_encode_features(data, one_hot_features)
    data = remove_outliers(data, numeric_features)

    # # Test and Training split
    X = data.drop(['Churn'], axis=1)
    y = pd.DataFrame(data['Churn'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=74, stratify=y)

    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.fit_transform(X_test[numeric_features])
    # remove column names from the dataframes
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values


    return X_train, X_test, y_train, y_test

def preprocess_adult_data():

    dataset_path = 'datasets/adult/'
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



if __name__ == '__main__':
    # X_train, X_test, y_train, y_test = preprocess_telco_data()
    # X_train, X_test, y_train, y_test = preprocess_adult_data()
    X_train, X_test, y_train, y_test = preprocess_credit_data()

    # training the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train.ravel())

    # evaluating the model
    y_pred = model.predict(X_train)
    print(accuracy_score(y_train, y_pred))

    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))

