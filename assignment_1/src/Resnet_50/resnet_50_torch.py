import numpy as np
import torch
import utils
import json
from torchvision import transforms, models
from PIL import Image

resnet_50 = models.resnet50(pretrained = True)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_image_path = "../../images/cobra.jpg"
image = Image.open(input_image_path)

# class_idx = json.loads("imagenet_class_index.json")
# labels = [class_idx[str(k)][1] for k in range(len(class_idx))]
with open('imagenet_classes.txt') as f:
  labels = [line.strip() for line in f.readlines()]

image_tensor = transform(image).unsqueeze(0)
output = resnet_50(image_tensor).detach()
_, indices = torch.sort(output, descending=True)
percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
[print(idx.item(), labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
print(percentage[85])
