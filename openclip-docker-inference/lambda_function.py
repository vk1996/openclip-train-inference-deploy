'''

handler used in AWS Lambda function

'''
import os
import boto3
import json
import base64


# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):

    payload = event['data']
    mode=event['mode']

    if mode=="image":
        payload = base64.b64decode(payload)
        payload = bytearray(payload)

    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       #ContentType='application/json',
                                       Body=payload)
    response=response['Body'].read()
    tensor_bytearray = bytearray(response)


    # Encode the bytearray into a Base64 string
    encoded_tensor = base64.b64encode(tensor_bytearray).decode('utf-8')

    # Create a JSON object
    json_data = {
        "tensor": encoded_tensor
    }

    # Serialize the JSON object to a string
    json_string = json.dumps(json_data)


    return json_string
