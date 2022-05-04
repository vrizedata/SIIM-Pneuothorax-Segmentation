#from flask import Flask, request, jsonify
#from black import out
from utils import *
import yaml, json
import cv2,os
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image


import numpy as np
import requests, json, cv2,os
from utils import encode_img,decode_img
import matplotlib.pyplot as plt



def input_gen(image_file):
    """
    makes data in proper input format
    """
    print("image type - ",type(image_file))
    print("image  below ")
    print(image_file)
    # image = cv2.imread(image_name)
    # img_str = image_name
    # #nparr = np.fromstring(img_str, np.uint8)
    # #image = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)
    # #image = image_name
    #image = np.fromstring(img_str, np.uint8).reshape( h, w, nb_planes )
    #image = Image.open(image_file)
    #st.image(image, caption='Input', use_column_width=True)
    #img_array = np.array(image)
    #cv2.imwrite('out.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

        # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
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
    #data = resp.json()
    #image = data['data']['image']
    image = resp["image"]
    image = decode_img(image)
    return image

with open('config.yaml') as yaml_file:
    config = yaml.safe_load(yaml_file)

config = config['CONFIGS']
model_path = config['MODEL_PATH']
img_size = config['IMG_SIZE']
device = config['DEVICE']
mask_thresh = config['MASK_THRESHOLD']
pred_thresh = config['PRED_THRESHOLD']
min_area = config['MIN_AREA']

MODEL = ort.InferenceSession(model_path)
transform = get_transforms()

def load_image(image_file):
	img = Image.open(image_file)
	return img




st.title("Lung Pneumothorax Segmentation")



def get_prediction(image):
    """
    model prediction
    returns mask,probability
	
    """
    image = image.transpose(2,0,1)
    batch = np.expand_dims(image,axis=0)
    onix_input = {"in_image": batch}
    mask,prob = MODEL.run(None, onix_input)
    return mask,prob

def predict_img(data):
            data = json.loads(data)
            #data = json.load(data)
            bs64_image = data["image"]
            #bs64_image = data.read()
                #try:
            image = decode_img(bs64_image)
            # preprocessing
            #image = data
            #print(image.shape) 
            out = transform(image=image)
            image = out["image"]
            mask,prob = get_prediction(image)
            print("result: ------------")
            #print(type(mask))
            mask,prob = sigmoid(mask[0][0]),sigmoid(prob[0][0])
            mask = post_process_mask(mask,prob,mask_thresh=mask_thresh,min_area=min_area,pred_thres=pred_thresh)
            mask = encode_img(mask)
            print(type(mask))
            data = {
                "image": mask,
                "probability": str(prob)
            }

            return data



uploaded_image = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
if uploaded_image is not None:
        #st.image(load_image(uploaded_image),width=250)
        			  #Saving upload
        with open(os.path.join("data","input_image.png"),"wb") as f:
                        f.write((uploaded_image).getbuffer())
        
        data = input_gen(uploaded_image)
        #print(data)
        print("------------------")
        print(type(data))
        print("-----------")
        #print(type(image_file))
        #print(image_file)
        res = predict_img(data)
        output = output_gen(res)
        print("**********************")
        print(type(output))
        #image_out = Image.fromarray(output)
        image_file_path = "data/output_image.png"
        #image_out.save(image_file_path)
        
        #plt.imshow(output,cmap='gray')
        #plt.show()
        plt.imsave(image_file_path,output, cmap='gray')
        # st.image(,width=250)
        col1, col2 = st.columns(2)
        col1.header("Original image")
        col1.image(load_image("data/input_image.png"), use_column_width=True)
        col2.header("Segmented image")
        col2.image(load_image(image_file_path), use_column_width=True)
        #st.image([load_image("data/input_image.png"), load_image(image_file_path)],width=250)


    

