import cv2
import numpy as np

def load_image(path):
    image = cv2.imread(path)
    image = hwc_to_chw(image)
    return image

def hwc_to_chw(image):
    return np.moveaxis(image, -1, 0)
