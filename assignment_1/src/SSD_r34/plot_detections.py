import numpy as np
import matplotlib.pyplot as plt

with open("coco-labels-2014_2017.txt", 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def ssd_plot(boxes, conf, obj, image):
    image = np.array(image)
    top_k=10
    plt.figure(figsize=(10,10))
    colors = plt.cm.hsv(np.linspace(0, 1, 81)).tolist()
    plt.imshow(image)  # plot the image for matplotlib
    currentAxis = plt.gca()
    w, h, c = image.shape
    print(w, h)
    # scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    for i in range(conf.size(0)):
        if conf[i].item() >= 0.6:
            object = obj[i].item()
            score = conf[i].item()*100
            label_name = labels[object - 1]
            display_txt = '%s: %.2f'%(label_name, score)
            print(display_txt)
            pt = boxes[i].detach().cpu().numpy()
            pt[0], pt[1] = pt[0]*w, pt[1]*w
            pt[2], pt[3] = pt[2]*h, pt[3]*h

            # pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
            color = colors[object-1]
            print(pt)
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})

    plt.show()
