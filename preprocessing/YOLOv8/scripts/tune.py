from ultralytics.models import YOLO

if __name__ == "__main__":
    # load pretrained model checkpoint
    model = YOLO("yolov8l.pt") 
    # data yaml path
    data_yaml_path = "/home/vislab-001/Jared/CS534-Team-3-AI-Project/data/PlantDoc_ObjectDetection/downscaled_plantdoc_od.yaml"
    # run hyperparameter tuning via Genetic algorithm
    model.tune(data=data_yaml_path, epochs=30, iterations=100, optimizer='AdamW', imgsz=640, device=[0, 1], plots=False, save=False, val=False)