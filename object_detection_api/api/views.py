from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import json

#Loading the saved_model
import tensorflow as tf
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import random
import matplotlib.pyplot as plt
import os

PATH_TO_SAVED_MODEL="/home/bruno-rg/Documents/object-detection-api/mobilenet/saved_model"
print('Loading model... \n', end='')

# Load saved model and build the detection function
detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)
print('Done!')

#Loading the label_map
category_index=label_map_util.create_category_index_from_labelmap("/home/bruno-rg/Documents/object-detection-api/label_map.pbtxt",use_display_name=True)

def load_image_into_numpy_array(path):
    return np.array(Image.open(path))

# Create your views here.
def detect(request, image):
    """image_path = '/home/bruno-rg/Documents/object-detection-api/images/AX_7170.jpeg'

    print('Running inference for {}... '.format(image_path), end='')

    image_np = load_image_into_numpy_array(image_path)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes']"""
    print(image)
    return HttpResponse("OK")