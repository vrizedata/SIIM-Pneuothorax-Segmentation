import numpy as np
import requests, json, cv2,os
from utils import encode_img,decode_img
import matplotlib.pyplot as plt

def input_gen(image_name):
    """
    makes data in proper input format
    """
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(type(image))
    image = encode_img(image)
    data = {
    "image" : image     
            }
    data= json.dumps(data)
    return data

def output_gen(resp):
    """
    process output data
    """
    data = resp.json()
    image = data['data']['image']
    image = decode_img(image)
    return image

img_path = os.path.join('data','img.png')
data = input_gen(img_path)
# print(data)
print("-----------------")
# print(data.shape)
print(type(data))
resp = requests.post("http://localhost:4001/predict", data)
print(type(resp))
out = output_gen(resp)
# display image on screen
plt.imshow(out,cmap='gray')
plt.show()


