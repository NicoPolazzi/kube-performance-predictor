# Kube Performance Predictor

This project represents my Master's thesis project in Sofware: Science and Technology at the University of Florence.

## Requirements

Ensure that you have the following tools installed:
* [Poetry](https://python-poetry.org/)
* [Minikube](https://minikube.sigs.k8s.io/docs/start/)
* [Istioctl](https://istio.io/latest/docs/setup/getting-started/)


## Experiment

The tests were performed on a local cluster using Minikube with a deployed [microservices-demo](https://github.com/GoogleCloudPlatform/microservices-demo) application.

### Setup

Create the cluster:

```shell
 minikube start --cpus=4 --memory 4096 --disk-size 32g
 ```

 Install Istio and enable sidecar injection:

 ```shell
istioctl install --set profile=minimal -y
kubectl label namespace default istio-injection=enabled
```

Minikube may lack specific Custom Resource Definitions (CRDs) required for the Gateway API. Install them manually if they are missing:

```shell
kubectl get crd gateways.gateway.networking.k8s.io &> /dev/null || \
{ kubectl kustomize "github.com/kubernetes-sigs/gateway-api/config/crd?ref=v1.3.0" | kubectl apply -f -; }
```
Clone the demo repository and deploy the application with the Istio service mesh component using Kustomize:

```shell
git clone --depth 1 --branch v0 https://github.com/GoogleCloudPlatform/microservices-demo.git
cd microservices-demo/kustomize
kustomize edit add component components/service-mesh-istio
kubectl apply -k .
```

Once all pods are running, expose the promethues server:

```shell
kubectl port-forward -n istio-system service/prometheus 9090:9090
```

### Run

You can run the experiment with:

```shell
poetry run python kpp/collector.py 
```
