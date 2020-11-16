[![CodeFactor](https://www.codefactor.io/repository/github/aiforgoodsimulator/model-server/badge)](https://www.codefactor.io/repository/github/aiforgoodsimulator/model-server)
[![codecov](https://codecov.io/gh/AIforGoodSimulator/model-server/branch/dev/graph/badge.svg?token=TK647J6ZUC)](https://codecov.io/gh/AIforGoodSimulator/model-server)
[![wemake-python-styleguide](https://img.shields.io/badge/style-wemake-000000.svg)](https://github.com/wemake-services/wemake-python-styleguide)
[![Testing](https://github.com/AIforGoodSimulator/model-server/workflows/AIforGood%20ModelServer%20Tests/badge.svg)](https://github.com/AIforGoodSimulator/model-server/actions?query=workflow%3A%22AIforGood+ModelServer+test+workflow%22)
[![Dev-Build-Deploy](https://github.com/AIforGoodSimulator/model-server/workflows/DEV%20Build%20Deploy/badge.svg)](https://github.com/AIforGoodSimulator/model-server/actions?query=workflow%3A%22DEV+Build+Deploy%22)
[![UAT-Build-Deploy](https://github.com/AIforGoodSimulator/model-server/workflows/UAT%20Build%20Deploy/badge.svg)](https://github.com/AIforGoodSimulator/model-server/actions?query=workflow%3A%22UAT+Build+Deploy%22)



### Install Azure ClI

Below is for Ubuntu, for other environments and details see - https://docs.microsoft.com/en-us/cli/azure/install-azure-cli-apt

```bash
$ sudo apt-get update
$ sudo apt-get install ca-certificates curl apt-transport-https lsb-release gnupg


$ curl -sL https://packages.microsoft.com/keys/microsoft.asc |
    gpg --dearmor |
    sudo tee /etc/apt/trusted.gpg.d/microsoft.gpg > /dev/null



$ AZ_REPO=$(lsb_release -cs)
$ echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ $AZ_REPO main" |
    sudo tee /etc/apt/sources.list.d/azure-cli.list


$ sudo apt-get update
$ sudo apt-get install azure-cli
```

### Install Helm 

Below is for Ubuntu, for other environments and details see - https://helm.sh/docs/intro/install/

```bash
$ curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3
$ chmod 700 get_helm.sh
$ ./get_helm.sh
```

### Install kubectl

Below is for Ubuntu, for other environments and details see - https://kubernetes.io/docs/tasks/tools/install-kubectl/

```bash
$ sudo apt-get update && sudo apt-get install -y apt-transport-https gnupg2 curl
$ curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
$ echo "deb https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee -a /etc/apt/sources.list.d/kubernetes.list
$ sudo apt-get update
$ sudo apt-get install -y kubectl


```

### Upgrade the pods 

Follow the steps below to authenticate and run commands to upgrade all kubernetes pods (dask workers) with the latest code libraries

```bash
$ az login 
```

You will be rediercted to a broswer and will need to login with your azure account.

Get the credentials for the Dask Kubernetes service for ai4good and update helm repositories:
```bash
az aks get-credentials --resource-group AI4Good --name DaskAks
helm repo add dask https://helm.dask.org/ 
helm repo update   
```

Now make changes to the dask-config.yml file (add python libraries, change workers, etc) and run the following to deploy to all workers:

```bash
cd into the model-server directory
helm upgrade dask-service dask/dask -f dask-config.yml        
```
                                      

### Other useful commands

* Get kube services and show your external IP
```bash
$ kubectl get services
NAME                     TYPE           CLUSTER-IP    EXTERNAL-IP     PORT(S)                       AGE
dask-service-scheduler   LoadBalancer   10.0.115.71   EXTERNAL_IP     8786:30339/TCP,80:30897/TCP   3d21h
kubernetes               ClusterIP      10.0.0.1      <none>          443/TCP                       3d23h
```

* List the running pods.  This can be useful after running the 'helm update' command to see the status of updating pods.
```bash
$ kubectl get pods
NAME                                      READY   STATUS    RESTARTS   AGE
dask-service-scheduler-7d985fbbb5-mxqzm   1/1     Running   0          10h
dask-service-worker-6d7c454445-2p5k5      1/1     Running   0          10h
dask-service-worker-6d7c454445-4mjmb      1/1     Running   0          10h
dask-service-worker-6d7c454445-65prp      1/1     Running   0          10h
dask-service-worker-6d7c454445-8cxrl      1/1     Running   0          10h
dask-service-worker-6d7c454445-8gwc4      1/1     Running   0          10h
dask-service-worker-6d7c454445-8r8kw      1/1     Running   0          10h
dask-service-worker-6d7c454445-99vs9      1/1     Running   0          10h
dask-service-worker-6d7c454445-9mpmz      1/1     Running   0          10h
dask-service-worker-6d7c454445-b7j66      1/1     Running   0          10h
dask-service-worker-6d7c454445-g657v      1/1     Running   1          10h
dask-service-worker-6d7c454445-n8xhg      1/1     Running   0          10h
dask-service-worker-6d7c454445-pxjjp      1/1     Running   0          10h
dask-service-worker-6d7c454445-wr7tb      1/1     Running   0          10h
dask-service-worker-6d7c454445-x5d7n      1/1     Running   0          10h
dask-service-worker-6d7c454445-xkqnj      1/1     Running   0          10h
```

* Examine the logs of a particular pod.  Can be useful for troubleshooting
```bash
$ kubectl logs dask-service-worker-6d7c454445-xkqnj 
+ '[' '' ']'
+ '[' -e /opt/app/environment.yml ']'
+ echo 'no environment.yml'
no environment.yml
+ '[' 'unzip python==3.7' ']'
+ echo 'EXTRA_CONDA_PACKAGES environment variable found.  Installing.'
+ /opt/conda/bin/conda install -y unzip python==3.7
EXTRA_CONDA_PACKAGES environment variable found.  Installing.
Collecting package metadata (current_repodata.json): ...working... done
...
...
...
```

* Install fresh ***WARNING : This is for new installs ONLY.  This will create a NEW cluster and it will have a NEW external IP***

```bash
helm delete dask-service
helm install dask-service dask/dask -f dask-config.yml      
```
