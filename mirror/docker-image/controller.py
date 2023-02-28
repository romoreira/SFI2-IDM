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
        output = process.stdout.read()
        print(str(output))
        return str(output)

class ThreadedFlowMeter(threading.Thread):

    predictor_ip = ""
    interface = ""

    def __init__(self, predictor_ip, interface):
        super(ThreadedFlowMeter, self).__init__()
        self.predictor_ip = predictor_ip
        self.interface = interface
    def run(self):

        command = 'cicflowmeter -i '+str(self.interface)+' -c flows.csv -u http://'+str(self.predictor_ip)+':8080/prediction'

        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)

        while True:
            print("While true ever")
            line = proc.stdout.read()
            print("Line: "+str(line))
            if not line:
                break
            else:
                print("doing some stuff with...", line)
                print("done for this line!")

        #print("Running FlowMeter and sending Flow to "+str(self.predictor_ip) +" on interface: "+str(self.interface))
        #process = subprocess.run(['cicflowmeter', '-i', str(self.interface), '-c', 'flows.csv', '-u', 'http://'+str(self.predictor_ip)+':8080/prediction'], check=True, stdout=subprocess.PIPE, universal_newlines=True)
        #output = process.stdout.read()
        #print(str(output))
        #return str(output)

@app.route('/start/<predictor_ip>/<interface>', methods=['GET'])
def start_flowmeter(predictor_ip, interface):
    flow_meter = ThreadedFlowMeter(predictor_ip, interface)
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
