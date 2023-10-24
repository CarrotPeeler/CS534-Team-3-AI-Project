import os
import pandas as pd
from preprocessing.YOLOv8.ultralytics.models import YOLO

if __name__ == "__main__":
    root = "/home/vislab-001/Jared/CS534-Team-3-AI-Project"

    # load pretrained model checkpoint
    model = YOLO(f"{root}/runs/detect/downscaled_train_100_epochs_plantdoc_od/weights/best.pt") 

    # data path
    data_dir = f"{root}/PlantDoc_ObjectDetection/test"

    # get all annotation file names without the .txt extension
    raw_anno_names = list(map(lambda x:x.rpartition('.')[0], os.listdir(data_dir)))

    # columns
    img = []
    target_cls = []
    target_xywh = []
    bbox_xyxy = []
    bbox_conf = []
    top1 = []
    top5 = []
    top1_conf = []
    top5_conf = []

    for img_name in raw_anno_names:
        # get path to img
        anno_path = data_dir + '/' + img_name + '.txt'
        img_path = data_dir + '/' + img_name + '.jpg'

        # grab target class and bbox xywh
        with open(anno_path) as f:
            labels = f.readlines()
            for label in labels:    
                cls, x, y, w, h = f.read().split()
                target_cls.append(int(cls))
                target_xywh.append()

        result = model.predict(source=img_path, imgsz=640)
        for r in result:
            boxes = r.boxes.cpu()
            probs = r.probs.cpu()
        
        # append results
        img.append(img_name)
        bbox_xyxy.append(boxes.xywh)
        bbox_conf.append(boxes.conf)
        top1.append(probs.top1)
        top5.append(probs.top5)
        top1_conf.append(probs.top1conf)
        top5_conf.append(probs.top5conf)

    results = pd.DataFrame()
    results["image"] = img
    results["bbox_xywh"] = bbox_xyxy
    results["bbox_conf"] = bbox_conf
    results["top1"] = top1
    results["top5"] = top5
    results["top1_conf"] = top1_conf
    results["top5_conf"] = top5_conf
    results.to_csv(f"{root}/yolo_results.csv", index=False)
    

        