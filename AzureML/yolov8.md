# Yolov8 in Azure Machine learning

## Yolo V8 in Azure Machine Learning

## Pre-requisites

- Azure Machine Learning Service
- Azure Storage Account

## Code

- Create a new Azure Machine Learning Service Workspace
- Create a notebook
- Pick the kernel as python with Tensorflow and Pytorch
- Now clone the repo from github
- Change conda environment to azureml_py38_TF_PY

```
source active azureml_py38_TF_PY
```

- clone the repo
  
```
git clone https://github.com/ultralytics/ultralytics.git
```

- Open Terminal and run the following command
- cd to the ultralytics folder
- cd into yolov8

```
pip install -r requirements.txt
```

- Now install Ultralytics Yolov8

```
pip install ultralytics
```

- Test in terminal

```
yolo task=detect mode=predict model=yolov8n.pt source="bus.jpg"
```

- output

```
yolo task=detect mode=predict model=yolov8n.pt source="bus.jpg"
Ultralytics YOLOv8.0.5 🚀 Python-3.8.5 torch-1.12.1 CPU
Fusing layers...
YOLOv8n summary: 168 layers, 3151904 parameters, 0 gradients, 8.7 GFLOPs
image 1/1 /mnt/batch/tasks/shared/LS_root/mounts/clusters/devbox1/code/Users/babal/yolov8/ultralytics/bus.jpg: 640x480 4 persons, 1 bus, 1 stop sign, 752.6ms
Speed: 1.3ms pre-process, 752.6ms inference, 301.7ms postprocess per image at shape (1, 3, 640, 640)
```

- Create a notebook with kernel as python with Tensorflow and Pytorch

```
# Load YOLOv8n, train it on COCO128 for 3 epochs and predict an image with it
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model
model.train(data='coco128.yaml', epochs=3)  # train the model
model('https://ultralytics.com/images/bus.jpg')  # predict on an image  
```

- Run the cell
- output

```
Found https://ultralytics.com/images/bus.jpg locally at bus.jpg
Ultralytics YOLOv8.0.5 🚀 Python-3.8.5 torch-1.12.1 CPU
Fusing layers... 
Model summary: 168 layers, 3151904 parameters, 0 gradients, 8.7 GFLOPs
image 1/1 /mnt/batch/tasks/shared/LS_root/mounts/clusters/devbox1/code/Users/babal/yolov8/ultralytics/bus.jpg: 640x480 4 persons, 1 bus, 1 stop sign, 51.3ms
Speed: 2.4ms pre-process, 51.3ms inference, 0.9ms postprocess per image at shape (1, 3, 640, 640)
```

- Now segementation

```
# Load YOLOv8n-seg, train it on COCO128-seg for 3 epochs and predict an image with it
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  # load a pretrained YOLOv8n segmentation model
model.train(data='coco128-seg.yaml', epochs=3)  # train the model
model('https://ultralytics.com/images/bus.jpg')  # predict on an image
```

- Output

```
Found https://ultralytics.com/images/bus.jpg locally at bus.jpg
Ultralytics YOLOv8.0.5 🚀 Python-3.8.5 torch-1.12.1 CPU
Fusing layers... 
YOLOv8n-seg summary: 195 layers, 3404320 parameters, 0 gradients, 12.6 GFLOPs
image 1/1 /mnt/batch/tasks/shared/LS_root/mounts/clusters/devbox1/code/Users/babal/yolov8/ultralytics/bus.jpg: 640x480 4 persons, 1 bus, 1 skateboard, 75.6ms
Speed: 0.6ms pre-process, 75.6ms inference, 2.4ms postprocess per image at shape (1, 3, 640, 640)
```