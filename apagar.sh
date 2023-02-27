#!/bin/bash
kubectl delete deployment keyrock
kubectl delete deployment mlagent
kubectl delete virtualservice keyrock
kubectl delete destinationrule keyrock
kubectl delete svc keyrock
