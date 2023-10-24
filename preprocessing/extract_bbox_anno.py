import os 
import sys
sys.path.insert(0, f"{os.getcwd()}")
from preprocessing.YOLOv8.ultralytics.models import YOLO

if __name__ == "__main__":
    root = "/home/vislab-001/Jared/CS534-Team-3-AI-Project"

    # load pretrained model checkpoint
    model = YOLO(f"{root}/runs/detect/downscaled_train_100_epochs_plantdoc_od/weights/best.pt") 

    # data yaml path
    data_yaml_path = f"{root}/PlantDoc_ObjectDetection/downscaled_plantdoc_od.yaml"

    results = model.predict(source=data_yaml_path, imgsz=640, save_conf=True, save_txt=True)