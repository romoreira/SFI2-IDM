from flask import (
    Flask, jsonify, render_template, request
)
import subprocess
from subprocess import run

app = Flask(__name__)
import threading
import requests
import json
#from model_trainer import load_model

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

def do_predct(json_string):
    print("JSON RECEBIDO NO DO_PREDICT:" +str(json_string))
    #dict = json.loads(json_string)
    dict = json_string
    df = pd.DataFrame([], columns=dict['columns'])
    list = dict['data'][0]
    df = df.append(pd.DataFrame([list], columns=dict['columns']), ignore_index=True)
    X_test = clean_dataset(df)
    knn = load_model("saved_model/knn.pth")
    y_pred = knn.predict(X_test)
    print(y_pred)
    return y_pred



@app.route('/prediction', methods=['POST'])
def predition_task():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json_string = request.get_json()
        print("JSON Recebido pelo prediction: "+str(json_string))
        return {'result': 'ok'}
    elif request.data:
        print('Data:' + str(request.data))
        json_string = request.get_json()
        prediction_response = do_predct(json_string)

        json_string = '{"src_ip": "SOURCE_IP", "src_port": "SOURCE_PORT", "dst_ip": "DST_IP", "dst_port": "DST_PORT", "result": [0], "probability": [[0]]}'
        json_string = json.loads(json_string)
        print("Posting json_string to: "+str(json_string))
        r = requests.post('http://68.219.96.1:3000/predictions', json=json_string)
        print("Posted Prediction to list: "+str(r.status_code))
        return json_string
    else:
        return 'Content-Type not supported!'


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
