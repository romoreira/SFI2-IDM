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

@app.route('/prediction', methods=['POST'])
def predition_task():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json_string = request.json
        print("JSON Recebido pelo prediction: "+str(json_string))
        return {'result': 'ok'}
    elif request.data:
        print('Data:' + str(request.data))
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
