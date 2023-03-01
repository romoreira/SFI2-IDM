import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for data visualization
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.neural_network import MLPClassifier
import time
from sklearn.model_selection import cross_val_score
import pickle

def csv_loader():
    data = 'dataset/dataset.csv'
    df = pd.read_csv(data)
    #print(df.shape)
    #print(df)
    #print(df.columns)
    return df
def get_correlation_graph(df):
    correlation = df.corr()
    plt.figure(figsize=(10,8))
    plt.title('Correlation of Attributes with Class variable')
    a = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
    a.set_xticklabels(a.get_xticklabels(), rotation=90)
    a.set_yticklabels(a.get_yticklabels(), rotation=30)
    plt.show()

def get_X_y(df):
    df['class_int'] = pd.Categorical(df[' Label']).codes
    df = clean_dataset(df)
    X = df.drop(['class_int'], axis=1)
    y = df['class_int']
    return X, y
def get_train_test(df):
    df['class_int'] = pd.Categorical(df[' Label']).codes
    #print(df['class_int'].unique())
    #print(df[' Label'].unique())
    #exit()
    df = clean_dataset(df)
    #X = df.drop([' Label', 'Flow ID', ' Source IP', ' Destination IP', ' Timestamp', 'SimillarHTTP', 'Flow Bytes/s'], axis=1)
    X = df.drop(['class_int'], axis=1)
    y = df['class_int']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    return X_train, X_test, y_train, y_test

def clean_dataset(df):
    df = df.drop([' Label', 'Flow ID', ' Source IP', ' Destination IP', ' Timestamp', 'SimillarHTTP', 'Flow Bytes/s'], axis=1)

    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep].astype(np.float64)

def feature_reduction(X, Y):
    # Feature extraction
    #test = SelectKBest(score_func=chi2, k=4)
    #fit = test.fit(X, Y)
    test = SelectKBest(f_classif, k=5)
    fit = test.fit(X, Y)

    # Summarize scores
    np.set_printoptions(precision=3)
    print(fit.scores_)

    features = fit.transform(X)
    # Summarize selected features
    print(features[0:5, :])
    exit()

def save_model(model):
    filename = 'saved_model/knn.pth'
    pickle.dump(model, open(filename, 'wb'))
    print("Model saved successfully!")

def load_model(model_name):
    # load the model from disk
    knn = pickle.load(open(model_name, 'rb'))
    return knn


X_train, X_test, y_train, y_test = get_train_test(csv_loader())
#print(X_train.dtypes)
#print(X_test.dtypes)
#print(y_train.dtypes)
#print(y_test.dtypes)


cols = X_train.columns
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])


X, y = get_X_y(csv_loader())
#cross_validation("KNN", X, y)
#cross_validation("RandomForest", X, y)
#cross_validation("SVM", X, y)
#cross_validation("MLP", X, y)

print("\n### KNN ###")
knn = KNeighborsClassifier(n_neighbors=7)

start = time.time()
knn.fit(X_train, y_train)
end = time.time()
print(f'KNN Time: {end - start}')

y_pred = knn.predict(X_test)
print(y_pred)
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
#y_pred_train = knn.predict(X_train)
#print('Training set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
#print('Test set accuracy score: {:.4f}'.format(knn.score(X_test, y_test)))
target_names = ['BENIGN', 'DrDoS-DNS', 'DrDoS_MSSQL', 'DrDoS_NetBIOS', 'DrDoS_SNMP', 'DrDoS_UDP','Syn', 'TFTP', 'UDP-lag']
#print(classification_report(y_test, y_pred, target_names=target_names))
scores = classification_report(y_test, y_pred, target_names=target_names)
file = open("Results/KNN_one.txt", "w")
a = file.write("Time: "+str(end - start)+str("\n"))
a = file.write(str(scores))
file.close()
print(a)

save_model(knn)