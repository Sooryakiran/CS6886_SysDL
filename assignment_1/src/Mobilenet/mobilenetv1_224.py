import numpy as np
import torch
import utils
import json
import copy
from torchvision import transforms, models
from mobilenet import mobilenet_v1 as MobileNet
from PIL import Image

model = MobileNet()
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
])

input_image_path = "../../images/cobra.jpg"
image = Image.open(input_image_path)

# print(model)
state = torch.load("mobilenet_v1_1.0_224.pth")
state_copy = copy.deepcopy(state)
for k, v in state.items():
    if "Conv2d_" in k:
        state_copy.pop(k)
        parts = k.split("Conv2d_")
        rest = parts[1].split(".")
        new_rest =  parts[0] + rest[0] + ".conv"
        # print(parts[0])
        for st in rest[1:]:
            if "pointwise" in st:
                st = "1"
            elif "depthwise" in st:
                st = "0"
            if not "conv" in st:
                new_rest = new_rest + "." + st
        state_copy.update({new_rest: v})
        # print(k, new_rest)
# print(state_copy)
model.load_state_dict(state_copy)
model.eval()

# class_idx = json.loads("imagenet_class_index.json")
# labels = [class_idx[str(k)][1] for k in range(len(class_idx))]
with open('imagenet_classes.txt') as f:
  labels = [line.strip() for line in f.readlines()]

image_tensor = transform(image).unsqueeze(0)
output = model(image_tensor).detach()
_, indices = torch.sort(output, descending=True)
percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
[print(idx.item(), labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
print(percentage[85])
