'''

AWS Lambda to connect REST API with OpenClip sagemaker endpoint doesn't require AWS credentials for testing
The configured Sagemaker endpoint might be turned off for cost $$ reasons , email vignesh.kotteeswaran@gmail.com before testing

'''

import requests
import json
import base64
import io
import torch




def infer_from_lambda(input_data,mode):
    if mode=="image":
        with open("test.png", 'rb') as f:
            img_bytes = f.read()

        img_bytes=base64.b64encode(img_bytes).decode('utf-8')
        data = {"data":img_bytes,"mode":mode}
    elif mode=="text":
        data = {"data":input_data,"mode":mode}

    else:
        return []

    response = requests.post(lambda_url,json=data)



    if response.status_code == 200:
        #print("Success:")
        loaded_json = json.loads(response.json())
        # Decode the Base64 string back into a bytearray
        decoded_tensor = bytearray(base64.b64decode(loaded_json["tensor"]))
        decoded_tensor = io.BytesIO(decoded_tensor)
        decoded_tensor = torch.load(decoded_tensor, weights_only=True)
        print("Decoded Tensor:", decoded_tensor.shape)
        return decoded_tensor

    else:
        print(f"Failed with status code {response.status_code}")
        return []


if __name__=="__main__":

    try:
        # Your API Gateway URL
        lambda_url = "https://c5jorm9b11.execute-api.us-east-1.amazonaws.com/test"
        img_path="test.png"
        description="A bald man with a long gray beard smiles a faint and dreamy smile while carrying on his back a little girl who wears his hat.An old man with a long gray beard and tattered clothes wakes up in the forest and, while still sitting, grabs hold of his rifle and checks his head"
        img_features=infer_from_lambda(img_path,"image")
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        text_features=infer_from_lambda(description,"text")
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        cosine_sim_score=100.0 * img_features @ text_features.T
        print('Cosine similarity score:',cosine_sim_score.detach().numpy()[0][0])

    except Exception as e:
        print(e)
