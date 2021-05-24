import boto3
from sagemaker.predictor import Predictor
from PIL import Image
import numpy as np
import json
import os
import requests
from requests_aws4auth import AWS4Auth
from data.fashion_mnist import FashionMnistDataset

ROOT_DIR = os.getcwd()
endpoint = 'dev-fmnist-v4-deployment'
endpoint_url = 'https://runtime.sagemaker.ap-southeast-1.amazonaws.com/endpoints/dev-fmnist-v4-deployment/invocations'
local_url = 'http://127.0.0.1:5000/predict'
auth = AWS4Auth('AKIAVR5S2FYGFLZAG2Z5',
                'V4qkYxGEtsG1j2Bo+v6UQi8rBw7UAoz70kToJOsK', 'ap-southeast-1', 'sagemaker')
headers = {'Content-type': 'image/png'}
runtime = boto3.Session().client(service_name='sagemaker-runtime')


# set the object categories array
object_categories = ['AnkleBoot', 'Bag', 'Coat', 'Dress',
                     'Pullover', 'Sandal', 'Shirt', 'Sneaker', 'TShirtTop', 'Trouser']


pil_image = Image.open(ROOT_DIR + '/src/test_output_34_9.png')
image_bytes = pil_image.tobytes()  # from PIL.Image

files = {
    "image": ("test_output_34_9.png", image_bytes),
    "image": ("test_output_34_9.png", image_bytes),
}

response = requests.post(local_url, headers=headers, auth=auth, files=files)
print(response.content)


# AWS

# Call your model for predicting which object appears in this image.
# response = runtime.invoke_endpoint(
#     EndpointName=endpoint,
#     ContentType='image/png',
#     Body=bytearray([test_X[0].squeeze(-1)])
# )

# read the prediction result and parse the json
# result = response['Body'].read()
# result = json.loads(result)

# print(result)

# which category has the highest confidence?
# pred_label_id = np.argmax(result)
# print("Result: label - " +
#       object_categories[pred_label_id] + ", probability - " + str(result[pred_label_id]))


