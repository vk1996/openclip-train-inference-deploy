'''

local script to send img/text payload to Sagemaker endpoint
Also builds new sagemaker endpoint & it's docker if Sagemaker endpoint is unavailable

'''
import boto3
import os
from botocore.exceptions import ClientError
from sagemaker import Model
from deploy_config import model_location,role,inference_repository_uri,endpoint_name,version,region,account_id,delete_endpoint_after_prediction,build_endpoint_docker
import io
import torch
from sagemaker import Predictor

def infer(input_data,mode):
    if mode=="image":
        with open(input_data, 'rb') as f:
            img_bytes = f.read()
        payload = bytearray(img_bytes )

    elif mode=="text":
        payload=input_data
    else:
        return

    data = predictor.predict(payload)
    data = io.BytesIO(data)
    data = torch.load(data, weights_only=True)
    print('output:',data.shape)
    return data

def check_endpoint_exists(endpoint_name):
    sagemaker_client = boto3.client('sagemaker')
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        print("Endpoint exists:", response)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            print(f"Endpoint '{endpoint_name}' does not exist.")
            return False
        else:
            raise

def build_sagemaker_docker_and_push_to_ecr():
    os.system(f"sudo docker build -t {endpoint_name} .")
    os.system(f"sudo docker tag {endpoint_name} {inference_repository_uri}")
    os.system(f"aws ecr get-login-password --region {region} | sudo docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com")
    os.system(f"aws ecr create-repository --repository-name {endpoint_name}")
    os.system(f"sudo docker push {account_id}.dkr.ecr.{region}.amazonaws.com/{endpoint_name}:test{version}")


def delete_sagemaker_endpoint(endpoint_name):
    """
    Deletes a SageMaker endpoint and its associated endpoint configuration.

    Args:
        endpoint_name (str): The name of the endpoint to delete.
    """
    # Create a SageMaker client
    sagemaker_client = boto3.client('sagemaker')

    try:
        # Delete the endpoint
        print(f"Deleting endpoint: {endpoint_name}...")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"Endpoint '{endpoint_name}' deleted successfully.")

        # Optionally delete the endpoint configuration
        print(f"Deleting endpoint configuration for: {endpoint_name}...")
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print(f"Endpoint configuration for '{endpoint_name}' deleted successfully.")

    except Exception as e:
        print(f"Error occurred while deleting the endpoint: {e}")



if __name__=="__main__":

    if not check_endpoint_exists(endpoint_name):

    ############# Creates new Sagemaker endpoint if not available #############
        if build_endpoint_docker:
            build_sagemaker_docker_and_push_to_ecr()
        sagemaker_model = Model(
            model_data=model_location,
            role=role,
            image_uri=inference_repository_uri,
            name=endpoint_name.lower())
        try:
            sagemaker_model.deploy(initial_instance_count=1, instance_type='ml.c6i.2xlarge',
                                   endpoint_name=endpoint_name)
        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                endpoint_name = endpoint_name + 'a'
                sagemaker_model.deploy(initial_instance_count=1, instance_type='ml.c6i.2xlarge',
                                       endpoint_name=endpoint_name)

    ############## Creates new Sagemaker endpoint if not available ############

    predictor = Predictor(endpoint_name=endpoint_name)



    path='test.png'
    img_features=infer(path,mode="image")
    img_features = img_features / img_features.norm(dim=-1, keepdim=True)

    #test description
    description="A bald man with a long gray beard smiles a faint and dreamy smile while carrying on his back a little girl who wears his hat.An old man with a long gray beard and tattered clothes wakes up in the forest and, while still sitting, grabs hold of his rifle and checks his head"
    text_features=infer(description,mode="text")
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    cosine_sim_score=100.0 * img_features @ text_features.T
    print('Cosine similarity score:',cosine_sim_score.detach().numpy()[0][0])

    if delete_endpoint_after_prediction:
        delete_sagemaker_endpoint(endpoint_name)



