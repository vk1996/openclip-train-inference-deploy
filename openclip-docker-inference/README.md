## OpenCLIP Inference with AWS Sagemaker deploy ##


### Documentation ###

### The inference model runs remotely in a AWS sagemaker ml.c6i.2xlarge endpoint ###

##### Note: The Dockerfile in folder openclip-docker-inference is for AWS ECR to build Sagemaker endpoint. #####
##### The below command executes all necessary steps like building docker for Sagemaker endpoint, #####
##### pushing to AWS ECR, creating model endpoint #####
##### invoking endpoint and predicting image vectors & text vectors #####
##### The below cmd deploys trained OpenCLIP chkpt in AWS Sagemaker, can be configured for autoscaling & serveless inference #####

```
$python3 sagemaker_inference.py
```

##### The below command can be used for testing the Sagemaker endpoint without building it's docker. Also doesn't require any AWS credentials #####
##### The lambda handler in the below script connects REST API with my existing Sagemaker endpoint ##### 

```
$python3 lambda_inference.py
```

