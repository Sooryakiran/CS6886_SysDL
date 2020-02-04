import numpy as np
import torch
import utils
import torchvision.models as models

resnet_50 = models.resnet50(pretrained = True)

input_image_path = "../images/phone.jpg"
image = utils.load_image(input_image_path)

image = np.expand_dims(image, 0)
image_tensor = torch.tensor(image).float()
output = resnet_50(image_tensor)
print(output.shape)
