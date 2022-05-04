import numpy as np
from albumentations import (Normalize, Resize, Compose)
import base64,cv2
from io import BytesIO
from PIL import Image

def decode_img(img):
  """
  decode img
  """
  # img = img.decode("utf-8")
  try:
    img = base64.b64decode(img)
    img = Image.open(BytesIO(img))
    img = np.array(img)
    return img
  except Exception as e:
    raise e

def encode_img(img):
  """
  encode image
  """
  try:
    ret, frame = cv2.imencode('.png', img)
    im_bytes = frame.tobytes()
    img_txt = base64.b64encode(im_bytes)
    img_txt = img_txt.decode("utf-8")
    return img_txt

  except Exception as e:
    raise e


def sigmoid(x):
    """
    returns sigmoid x
    """
    return 1/(1 + np.exp(-x))

def get_transforms(size=512, mean= (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):
    """
    normalize and resize input image
    """

    list_trfms = Compose([
            Resize(size, size),
            Normalize(mean=mean, std=std, p=1),
        ])
    return list_trfms

def post_process_area(probability, threshold, min_size):
    """
    count the number of components present. 
    mask is nn empty only if size of component > min_size
    """
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((1024, 1024), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def post_process_mask(mask,prob,mask_thresh=0.5,min_area=2500,pred_thres=0.5):
    """
    post process pipeline
    mask: mask
    mask_thresh : mask probability threshold
    prob: probability to be empty mask
    pred_thresh: threshold of empty mask
    min_area: minimum area required for a componnent to consider it as valid
    """
    if mask.shape != (1024,1024):
      mask = cv2.resize(mask, dsize=(1024,1024), interpolation=cv2.INTER_LINEAR)
    mask, num_of_predictions = post_process_area(mask,mask_thresh,min_area)
    if num_of_predictions == 0 or prob >= pred_thres:
        mask = np.zeros(shape=(1024,1024))
    return mask