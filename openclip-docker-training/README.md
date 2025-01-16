## OpenCLIP training with docker & AWS ##


### Documentation ###

#### Run the below commands to build container for training desired EC2 instances ####
#### Make sure to edit training data bucket in config.py ####

```
$sudo docker build -t clip-aws-training .
```

```
$sudo docker run -e AWS_ACCESS_KEY_ID=$access_key_id -e AWS_SECRET_ACCESS_KEY=$aws_secret -e AWS_DEFAULT_REGION=$aws-region -p 5000:5000 clip-aws-training
```
### Tools used ###
#### Docker ####
#### AWS S3 for model storage ####
#### MLflow for logging model metrics & system usage metrics & model versioning ####
#### Custom callback for model versioning & saving best checkpoint ####
#### AWS SNS for email notification & warning for model training  ####

#### Model metrics like train_loss & val_loss, hyperparams like lr,epoch_num,  ####
#### batch_size are logged against each epoch . refer attached mlflow ui screenshots ####

#### train.py downloads client images zip & description json from configured bucket ####
#### the docker can be configured to run as a monthly cron-job for the clients ####
#### to receive model each month . the train data bucket needs to be parameterized for this ####


