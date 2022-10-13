import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, xywh2xyxy
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from pathlib import Path
import os
import time

class YOLOv7pose:
    def __init__(self, config, device):

        self.device = device
        self.config = config
        print('Importing YOLOv7-POSE from ', config['path_weights_pose'])

        weigths = torch.load(config['path_weights_pose'], map_location=device)
        self.model = weigths['model']
        _ = self.model.float().eval()
        self.half = config['half_prec_pose']

        if self.half == True and torch.cuda.is_available():
            model.half().to(device)

        # import yolo functions
        sys.path.append(str(Path(config['path_model_pose'])))
        from utils.datasets import letterbox
        from utils.general import non_max_suppression_kpt, xywh2xyxy
        from utils.plots import output_to_keypoint, plot_skeleton_kpts


model = YOLOv7pose(load)

path = Path(r'C:\Users\scharton\Desktop\in_car_test_set\images')
paths = [f.path for f in os.scandir(path) if '.jpg' in f.name]



path = paths[23]
image = cv2.imread(str(path))

t0 = time.time()
image = letterbox(image, 640, stride=64, auto=True)[0]
image = transforms.ToTensor()(image)
image = torch.tensor(np.array([image.numpy()])) ### <---- ? HÄ

if torch.cuda.is_available():
    image = image.half().to(device)

with torch.no_grad():
    prediction = model(image)[0]
    output = non_max_suppression_kpt(prediction, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)

output

with torch.no_grad():
    output = output_to_keypoint(output)
nimg = image[0].permute(1, 2, 0) * 255
nimg = nimg.cpu().numpy().astype(np.uint8)
nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
for idx in range(output.shape[0]):
    plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)


%matplotlib inline
plt.figure(figsize=(8,8))
plt.axis('off')
plt.imshow(nimg)
plt.show()
print(time.time()-t0)
print(output.shape)

output
boxes = xywh2xyxy(output[:,2:6])
confs = output[:,6]
poses = output[:,7:]


nimg = cv2.rectangle(nimg, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,0,0), 2)
plt.figure(figsize=(8,8))
plt.axis('off')
plt.imshow(nimg)
plt.show()
