#pip install roboflow
#pip install -q git+https://github.com/THU-MIG/yolov10.git
#pip install -q roboflow
#wget -P -q https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10n.pt
#wget -P -q https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10n.pt
#wget -P -q https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10s.pt
#wget -P -q https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10m.pt
#wget -P -q https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10b.pt
#wget -P -q https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10x.pt
#wget -P -q https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10l.pt
from roboflow import Roboflow
from ultralytics import YOLOv10

rf = Roboflow(api_key="opq7XA6Ou6Na0SaSxlQ3")
project = rf.workspace("shopvision-sbz6n").project("shopvisonv2")
version = project.version(2)
dataset = version.download("yolov8")

model='/content/-q/yolov10n.pt' \
data='/content/Fire-Detection-1/data.yaml'
#yolo task=detect mode=train epochs=30 batch=32 plots=True \

#for testing for videofile 
model_path = '/content/runs/detect/train/weights/best.pt'
model = YOLOv10(model_path)
results = model(source='name of your video file here', conf=0.25,save=True)
