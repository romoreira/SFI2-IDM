## SFI2 Security Enhancements

kubectl delete deployment,svc mysql
kubectl delete pvc mysql-pv-claim
kubectl delete pv mysql-pv-volume
kubectl delete secret mysql-secret

kubectl exec --stdin --tty shell-demo -- /bin/bash

Based on: https://phoenixnap.com/kb/kubernetes-mysql


### Installing FlowMeter
pip3 install cicflowmeter