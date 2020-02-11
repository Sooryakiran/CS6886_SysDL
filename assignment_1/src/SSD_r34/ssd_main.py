import numpy as np
import torch
# import utils
import json
from torchvision import transforms, models
from ssd_r34 import SSD_R34 as ssd
from PIL import Image

model = ssd()
model.eval()
transform = transforms.Compose([
    transforms.Resize(1200),
    # transforms.CenterCrop(1200),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_image_path = "../../images/traffic.jpg"
image = Image.open(input_image_path)

image_tensor = transform(image).unsqueeze(0)
output = model(image_tensor)
print(output)
