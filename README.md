## OpenClip Train-Inference-Deploy ##

##### The repo has two sub-dirs in following structure #####

```
openclip-train-inference-deploy/
  -openclip-docker-training/
      --README.md
      -- Dockerfile
      -- train.py
  -openclip-docker-inference/
      --README.md
      -- Dockerfile (for sagemaker use only)
      -- sagemaker_inference.py
```


#### openclip-docker-training focuses on remote / local training, logging,metrics monitoring, model storage , notifications ####
#### openclip-docker-inference focuses on deploying trained model on AWS Sagemaker endpoint  ####

### Logging Model metrics ###

![mlflow_ui_model_metrics1.png](openclip-docker-training/mlflow_ui_model_metrics1.png)
![mlflow_ui_model_metrics.png](openclip-docker-training/mlflow_ui_model_metrics.png)

### Deploying in AWS Sagemaker endpoint ###
![sagemaker-endpoint.png](openclip-docker-training/sagemaker-endpoint.png)

### Model storing & versioning in S3 ###
![models3_storage.png](openclip-docker-training/models3_storage.png)

### Training notifications / warning via AWS SNS email publish ###

![notification&warning.png](openclip-docker-training/notification%26warning.png)
