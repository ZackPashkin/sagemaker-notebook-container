import boto3
import numpy as np
import cv2


runtime_client = boto3.client('runtime.sagemaker')

test_image = '002_0096.jpg'
img = cv2.imread(test_image)
img = cv2.resize(img, (224,224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.reshape(img,(1,224,224,3))


# calling deployed endpoint   
response = runtime_client.invoke_endpoint(EndpointName='mobile-endpoint',
                                       ContentType='application/json',
                                       Body=json.dumps(img.tolist()))
response = eval(response['Body'].read().decode('utf-8'))

print(response)
print(np.argmax(response['predictions']))
