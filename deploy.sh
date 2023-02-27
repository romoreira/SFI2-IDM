#!/bin/bash
kubectl apply -f keyrock-service.yaml 
kubectl apply -f keyrock-deployment.yaml
kubectl apply -f mlagent.yaml 
kubectl apply -f virtual-service.yaml 
