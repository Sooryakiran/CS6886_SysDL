import torch
from PIL import Image
from torchvision import transforms
from ssd_r34 import Encoder
import ssd_r34 as ssd
import numpy as np
import plot_detections

saved_path = "resnet34-ssd1200.pytorch"

if torch.cuda.is_available():
    model = torch.load(saved_path)
else:
    model = torch.load(saved_path, map_location = torch.device('cpu'))

model.eval()
print(model)
dboxes = ssd.dboxes_R34_coco([1200, 1200],[3,3,2,2,2,2])
enc = Encoder(dboxes)
transform = transforms.Compose([
    transforms.Resize((1200, 1200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_image_path = "../../images/people.jpg"
image = Image.open(input_image_path)

image_tensor = transform(image).unsqueeze(0)
output = model(image_tensor)
bboxes = output[0][0]
classes = output[1][0]
conf = output[2][0]
plot_detections.ssd_plot(bboxes, conf, classes, image)


"""
Convert into ONNX format

"""
x = torch.randn(1, 3, 1200, 1200, requires_grad=True)
torch_out = model(x)
torch.onnx.export(model,                    # model being run
                  x,                            # model input (or a tuple for multiple inputs)
                  "ssd1200.onnx",          # where to save the model (can be a file or file-like object)
                  export_params=True,           # store the trained parameter weights inside the model file
                  opset_version=10,             # the ONNX version to export the model to
                  do_constant_folding=True,     # whether to execute constant folding for optimization
                  input_names = ['input'],      # the model's input names
                  output_names = ['bboxes', 'classes', 'conf'],    # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'bboxes' : {0 : 'batch_size'},
                                'classes' : {0 : 'batch_size'},
                                'conf' : {0 : 'batch_size'}})


print("ONNX Model Saved")
exit()
"""
Run a sample test on onnx and compare with torch outputs
"""
import onnxruntime
ort_session = onnxruntime.InferenceSession("ssd_r34.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

"""
Compute ONNX Runtime output prediction

"""
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)
#
# np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
# print("Exported model has been tested with ONNXRuntime, and the result looks good!")
