from flask import (
    Flask, jsonify, render_template, request
)
import subprocess
from subprocess import run

app = Flask(__name__)
import threading


class NewThreadedTask(threading.Thread):
    def __init__(self):
        super(NewThreadedTask, self).__init__()

    def run(self):
        # run some code here
        print('Threaded task has been completed')

class ThreadedTCPDump(threading.Thread):

    amount = 0

    def __init__(self, amount):
        super(ThreadedTCPDump, self).__init__()
        self.amout = amount
    def run(self):
        print("Running TCPDump with amout: "+str(self.amout))
        process = subprocess.run(['tcpdump', '-c', '10', '-i', 'eth0', '-vvv'], check=True, stdout=subprocess.PIPE, universal_newlines=True)
        output = process.stdout
        return str(output)

class ThreadedFlowMeter(threading.Thread):

    def __init__(self, predictor_ip):
        super(ThreadedFlowMeter, self).__init__()
        self.predictor_ip = predictor_ip
    def run(self):
        print("Running FlowMeter and sending Flow to "+str(self.predictor_ip))
        process = subprocess.run(['cicflowmeter', '-i', 'eth0', '-c', 'flows.csv', '-u', 'http://'+str(self.predictor_ip)+':8080/prediction'], check=True, stdout=subprocess.PIPE, universal_newlines=True)
        output = process.stdout
        return str(output)

@app.route('/start/<predictor_ip>', methods=['GET'])
def start_flowmeter(predictor_ip):
    flow_meter = ThreadedFlowMeter(predictor_ip)
    flow_meter.start()
    #flow_meter.join()
    return {'status': 'ok', 'result': str("FlowMeter is Running")}, 200

@app.route('/sample', methods=['GET'])
def flow_meter():
    new_thread = NewThreadedTask()
    new_thread.start()
    # optionally wait for task to complete
    new_thread.join()
    return {'status': 'ok'}, 200

@app.route('/read/<amount>', methods=['GET'])
def read(amount):
    tcpdump = ThreadedTCPDump(amount)
    result = tcpdump.run()
    return {'status': 'ok', 'result': result}
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
