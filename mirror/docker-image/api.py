from flask import (
    Flask, jsonify, render_template, request
)
import subprocess
from subprocess import run

app = Flask(__name__)
import threading
import requests
import json
import logging

#For machine learning pourposes
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for data visualization
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle

prediction_ip = ""
prediction_service = "Down"
knn = ""
rf = ""
mlp = ""
svm = ""


def load_model(model_name):
    # load the model from disk
    knn = pickle.load(open(model_name, 'rb'))
    return knn

def clean_dataset(df):
    df = df.drop(['src_ip', 'dst_ip', 'timestamp', 'flow_byts_s'], axis=1)

    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep].astype(np.float64)

def do_predict_svm(json_string):
    #print("JSON RECEBIDO NO DO_PREDICT:" +str(json_string))
    #dict = json.loads(json_string)
    dict = json_string
    #print("Colunas recebidas no JSON: "+str(dict['columns']))
    df = pd.DataFrame([], columns=dict['columns'])
    list = dict['data'][0]
    df = df.append(pd.DataFrame([list], columns=dict['columns']), ignore_index=True)

    src_ip = df['src_ip'].values[0]
    dst_ip = df['dst_ip'].values[0]

    X_test = clean_dataset(df)
    global svm
    y_pred = svm.predict(X_test)
    #probability = knn.predict_proba(X_test)
    probability = 1


    return y_pred, src_ip, int(X_test['src_port'].values[0]), dst_ip, int(X_test['dst_port'].values[0]), probability

def do_predict_rf(json_string):
    #print("JSON RECEBIDO NO DO_PREDICT:" +str(json_string))
    #dict = json.loads(json_string)
    dict = json_string
    #print("Colunas recebidas no JSON: "+str(dict['columns']))
    df = pd.DataFrame([], columns=dict['columns'])
    list = dict['data'][0]
    df = df.append(pd.DataFrame([list], columns=dict['columns']), ignore_index=True)

    src_ip = df['src_ip'].values[0]
    dst_ip = df['dst_ip'].values[0]

    X_test = clean_dataset(df)
    global rf
    y_pred = rf.predict(X_test)
    #probability = knn.predict_proba(X_test)
    probability = 1


    return y_pred, src_ip, int(X_test['src_port'].values[0]), dst_ip, int(X_test['dst_port'].values[0]), probability

def do_predict_mlp(json_string):
    #print("JSON RECEBIDO NO DO_PREDICT:" +str(json_string))
    #dict = json.loads(json_string)
    dict = json_string
    #print("Colunas recebidas no JSON: "+str(dict['columns']))
    df = pd.DataFrame([], columns=dict['columns'])
    list = dict['data'][0]
    df = df.append(pd.DataFrame([list], columns=dict['columns']), ignore_index=True)

    src_ip = df['src_ip'].values[0]
    dst_ip = df['dst_ip'].values[0]

    X_test = clean_dataset(df)
    global mlp
    y_pred = mlp.predict(X_test)
    #probability = knn.predict_proba(X_test)
    probability = 1


    return y_pred, src_ip, int(X_test['src_port'].values[0]), dst_ip, int(X_test['dst_port'].values[0]), probability

def do_predict_knn(json_string):
    #print("JSON RECEBIDO NO DO_PREDICT:" +str(json_string))
    #dict = json.loads(json_string)
    dict = json_string
    #print("Colunas recebidas no JSON: "+str(dict['columns']))
    df = pd.DataFrame([], columns=dict['columns'])
    list = dict['data'][0]
    df = df.append(pd.DataFrame([list], columns=dict['columns']), ignore_index=True)

    src_ip = df['src_ip'].values[0]
    dst_ip = df['dst_ip'].values[0]

    X_test = clean_dataset(df)
    global knn
    y_pred = knn.predict(X_test)
    #probability = knn.predict_proba(X_test)
    probability = 1


    return y_pred, src_ip, int(X_test['src_port'].values[0]), dst_ip, int(X_test['dst_port'].values[0]), probability


@app.route('/start_prediction/<predictor_ip>', methods=['GET'])
def start_prediction_server(predictor_ip):
    global prediction_ip
    prediction_ip = predictor_ip
    global prediction_service
    prediction_service = "Up"
    return {'Prediction Status': "started"}, 200

@app.route('/prediction_status', methods=['GET'])
def get_prediction_status():
    return {'Prediction Service': str(prediction_service), 'Preditor IP': str(prediction_ip)}, 200

@app.route('/load_models', methods=['GET'])
def load_models():
    global knn
    global mlp
    global rf
    global svm
    knn = load_model("/home/saved_model/knn.pth")
    mlp = load_model("/home/saved_model/mlp.pth")
    rf = load_model("/home/saved_model/rf.pth")
    svm = load_model("/home/saved_model/svm.pth")

    return {'Models Loaded': 'OK'}, 200

@app.route('/prediction/svm', methods=['POST'])
def predition_task_svm():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):

        json_string = request.get_json()

        prediction, src_ip, src_port, dst_ip, dst_port, probability = do_predict_svm(json_string)
        features = ['BENIGN', 'DrDoS-DNS', 'DrDoS_MSSQL', 'DrDoS_NetBIOS', 'DrDoS_SNMP', 'DrDoS_UDP','Syn', 'TFTP', 'UDP-lag']
        return {'Predicted Class': features[int(prediction)]}, 200
        #features = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']

        #json_string = json.dumps({'src_ip': str(src_ip), 'src_port': str(src_port), 'dst_ip': str(dst_ip), 'dst_port': str(dst_port), 'result': str(features[int(prediction)]), 'probability': str(probability[0][int(prediction)])}, indent=5)

        #print("JSON STING 2 AFTER AGGREGATION: " + str(json_string))

        #r = requests.post('http://'+str(prediction_ip)+':3000/predictions', json=json_string)
        #print("Posted Prediction to list: "+str(r.status_code))
        #return json_string
    else:
        return 'Content-Type not supported!'

@app.route('/prediction/rf', methods=['POST'])
def predition_task_rf():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        
        json_string = request.get_json()

        prediction, src_ip, src_port, dst_ip, dst_port, probability = do_predict_rf(json_string)
        features = ['BENIGN', 'DrDoS-DNS', 'DrDoS_MSSQL', 'DrDoS_NetBIOS', 'DrDoS_SNMP', 'DrDoS_UDP','Syn', 'TFTP', 'UDP-lag']
        return {'Predicted Class': features[int(prediction)]}, 200
        #features = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']

        #json_string = json.dumps({'src_ip': str(src_ip), 'src_port': str(src_port), 'dst_ip': str(dst_ip), 'dst_port': str(dst_port), 'result': str(features[int(prediction)]), 'probability': str(probability[0][int(prediction)])}, indent=5)

        #print("JSON STING 2 AFTER AGGREGATION: " + str(json_string))

        #r = requests.post('http://'+str(prediction_ip)+':3000/predictions', json=json_string)
        #print("Posted Prediction to list: "+str(r.status_code))
        #return json_string
    else:
        return 'Content-Type not supported!'

@app.route('/prediction/mlp', methods=['POST'])
def predition_task_mlp():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):

        json_string = request.get_json()

        prediction, src_ip, src_port, dst_ip, dst_port, probability = do_predict_mlp(json_string)
        features = ['BENIGN', 'DrDoS-DNS', 'DrDoS_MSSQL', 'DrDoS_NetBIOS', 'DrDoS_SNMP', 'DrDoS_UDP','Syn', 'TFTP', 'UDP-lag']
        return {'Predicted Class': features[int(prediction)]}, 200
        #features = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']

        #json_string = json.dumps({'src_ip': str(src_ip), 'src_port': str(src_port), 'dst_ip': str(dst_ip), 'dst_port': str(dst_port), 'result': str(features[int(prediction)]), 'probability': str(probability[0][int(prediction)])}, indent=5)

        #print("JSON STING 2 AFTER AGGREGATION: " + str(json_string))

        #r = requests.post('http://'+str(prediction_ip)+':3000/predictions', json=json_string)
        #print("Posted Prediction to list: "+str(r.status_code))
        #return json_string
    else:
        return 'Content-Type not supported!'

@app.route('/prediction/knn', methods=['POST'])
def predition_task_knn():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):

        json_string = request.get_json()

        prediction, src_ip, src_port, dst_ip, dst_port, probability = do_predict_knn(json_string)
        features = ['BENIGN', 'DrDoS-DNS', 'DrDoS_MSSQL', 'DrDoS_NetBIOS', 'DrDoS_SNMP', 'DrDoS_UDP','Syn', 'TFTP', 'UDP-lag']
        return {'Predicted Class': features[int(prediction)]}, 200
        #features = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']

        #json_string = json.dumps({'src_ip': str(src_ip), 'src_port': str(src_port), 'dst_ip': str(dst_ip), 'dst_port': str(dst_port), 'result': str(features[int(prediction)]), 'probability': str(probability[0][int(prediction)])}, indent=5)

        #print("JSON STING 2 AFTER AGGREGATION: " + str(json_string))

        #r = requests.post('http://'+str(prediction_ip)+':3000/predictions', json=json_string)
        #print("Posted Prediction to list: "+str(r.status_code))
        #return json_string
    else:
        return 'Content-Type not supported!'


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
