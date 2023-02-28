from flask import Flask
from subprocess import run
app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return "Hello World!"

@app.route('/read/<amount>', methods=['GET'])
def read(amount):
    data = run("tcpdump -c "+str(amount)+" -i eth0 -vvv",capture_output=True,shell=True)
    return str(data.stdout)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True)
