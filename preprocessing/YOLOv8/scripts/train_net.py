from ultralytics.models import YOLO

if __name__ == "__main__":
    # load pretrained model checkpoint
    model = YOLO("yolov8l.pt") 

    # data yaml path
    data_yaml_path = "/home/vislab-001/Jared/CS534-Team-3-AI-Project/data/PlantDoc_ObjectDetection/downscaled_plantdoc_od.yaml"

    # Train the model with 2 GPUs
    results = model.train(data=data_yaml_path, epochs=100, imgsz=640, device=[0, 1])
