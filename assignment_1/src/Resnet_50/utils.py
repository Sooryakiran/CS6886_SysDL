import cv2
import numpy as np

def load_image(path):
    image = cv2.imread(path)
    image = hwc_to_chw(image)
    return image

def hwc_to_chw(image):
    return np.moveaxis(image, -1, 0)

def label_list(path):
    classes = None
    with open(path) as f:
        classes = [line.strip() for line in f.readlines()]
    return classes
