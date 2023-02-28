from flask import (
    Flask, jsonify, render_template, request
)
import subprocess
from subprocess import run

app = Flask(__name__)
import threading



class ThreadedPrediction(threading.Thread):

    def __init__(self):
        super(ThreadedPrediction, self).__init__()
    def run(self):
        print("Running Prediction")
        process = subprocess.run(['cicflowmeter', '-i', 'eth0', '-c', 'flows.csv', '--url', 'http://0.0.0.0:8080/prediction'], check=True, stdout=subprocess.PIPE, universal_newlines=True)
        output = process.stdout
        return str(output)

@app.route('/prediction', methods=['POST'])
def predition_task():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        print("JSON Recebido pelo prediction: "+str(json))
        return {'result': 'ok'}
    elif request.data:
        print('Data:' + str(request.data))
        return {"src_ip": "SOURCE_IP", "src_port": "SOURCE_PORT", "dst_ip": "DST_IP", "dst_port": "DST_PORT", "result": [0], "probability": [[0]]}
    else:
        return 'Content-Type not supported!'


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
