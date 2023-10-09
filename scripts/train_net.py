import os
from YOLOv8.ultralytics.models import YOLO

# load pretrained model checkpoint
ckpt_filepath = os.getcwd() + "/checkpoints/yolov8l.pt"
model = YOLO(ckpt_filepath) 

# data yaml path
data_yaml_path = os.getcwd() + "/PlantDoc_ObjectDetection/plantdoc_objectdetection.yaml"

# Train the model with 2 GPUs
results = model.train(data=data_yaml_path, epochs=100, imgsz=640, device=[0, 1])

# print training results
print(f"Final Training Results:\n\n{results}")
