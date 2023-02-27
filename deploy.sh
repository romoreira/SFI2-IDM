#!/bin/bash
kubectl apply -f keyrock-service.yaml 
kubectl apply -f keyrock-deployment.yaml
kubectl apply -f mirror/mlagent.yaml 
kubectl apply -f mirror/virtual-service.yaml 
