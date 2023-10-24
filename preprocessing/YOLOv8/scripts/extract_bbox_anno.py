import os
from preprocessing.YOLOv8.ultralytics.models import YOLO

if __name__ == "__main__":
    root = "/home/vislab-001/Jared/CS534-Team-3-AI-Project"

    # load pretrained model checkpoint
    model = YOLO(f"{root}/runs/detect/downscaled_train_100_epochs_plantdoc_od/weights/best.pt") 

    # data path
    data_dir = f"{root}/PlantDoc_ObjectDetection/test"

    for img_anno_file in os.listdirdata_dir 

    results = model.predict(source=data_yaml_path, imgsz=640, save_conf=True, save_txt=True)