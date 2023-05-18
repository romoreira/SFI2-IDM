## Configurations for Network Flows Monitoring Using [FlowMeter](https://pypi.org/project/cicflowmeter/)

Steps:

1. `sudo apt-get install python3-pip` for Python3 PIP installation
2. `pip3 install cicflowmeter` for FlowMeter installation through PIP
3. `sudo cp flow.py` and `sudo cp flow_session.py` replace the default file (installed from PIP) by the file inside this repo.
4. `sudo cicflowmeter -i eth0 -c output.csv` run the FlowMeter
