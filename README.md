# Kube Performance Predictor

This project represents my Master's thesis in Sofware: Science and Technology at the University of Florence.

## Requirements

Ensure that you have the following tools installed:
* [Poetry](https://python-poetry.org/)
* [Minikube](https://minikube.sigs.k8s.io/docs/start/)
* [Istioctl](https://istio.io/latest/docs/setup/getting-started/)
* [Kustomize](https://github.com/kubernetes-sigs/kustomize)


## Setup

The tests were performed on a local cluster using Minikube with the deployed [microservices-demo](https://github.com/GoogleCloudPlatform/microservices-demo) application.

In order to replicate the experiment on your device, you need to follow these setup steps.


### Create the cluster

Initialize a Minikube cluster with sufficient resources:

```shell
 minikube start --cpus=4 --memory 4096 --disk-size 32g
 ```

 ### Install Istio and Addons

Install the Istio minimal profile and enable sidecar injection for the default namespace:

 ```shell
istioctl install --set profile=minimal -y
kubectl label namespace default istio-injection=enabled
```

**Important**: The minimal profile does not include Prometheus by default. Install it manually to enable metrics collection:

```shell
kubectl apply -f [https://raw.githubusercontent.com/istio/istio/master/samples/addons/prometheus.yaml](https://raw.githubusercontent.com/istio/istio/master/samples/addons/prometheus.yaml)
```

### Install Gateway API CRDs

Minikube may lack specific Custom Resource Definitions (CRDs) required for the Gateway API. Install them if missing:

```shell
kubectl get crd gateways.gateway.networking.k8s.io &> /dev/null || \
{ kubectl kustomize "github.com/kubernetes-sigs/gateway-api/config/crd?ref=v1.3.0" | kubectl apply -f -; }
```

### Deploy the application

Clone the demo repository and deploy the application with the Istio service mesh component:

```shell
git clone --depth 1 --branch v0 https://github.com/GoogleCloudPlatform/microservices-demo.git
cd microservices-demo/kustomize
kustomize edit add component components/service-mesh-istio
kubectl apply -k .
```

Wait for all pods to be in the `Running` state before proceeding.


## Configuration

The application uses Environment Variables for configuration. You can set these in your shell or create a `.env` file in the root directory of the project. For example:

```
PROMETHEUS_URL=http://localhost:9090
EXPERIMENT_DURATION_SECONDS=600
QUERY_SAMPLE_DURATION_SECONDS=60

TEST_USER_START=10
TEST_USER_END=100
TEST_USER_STEP=5
```

## Usage

### Expose Prometheus

Before running the experiment, you must expose the Prometheus server so the collector can access it:

```shell
kubectl port-forward -n istio-system service/prometheus 9090:9090
```

### Run the experiment

In a new terminal window, execute the collector script:

```shell
poetry run python kpp/collector.py 
```
