import os
import numpy as np
import pandas as pd
from preprocessing.YOLOv8.ultralytics.models import YOLO

if __name__ == "__main__":
    root = "/home/vislab-001/Jared/CS534-Team-3-AI-Project"
    # file path to the annotations for each image
    data_dir = f"{root}/data/PlantDoc_ObjectDetection/test"
    img_dir = f"{data_dir}/images"
    label_dir = f"{data_dir}/labels"

    # load pretrained model checkpoint
    model = YOLO(f"{root}/runs/detect/downscaled_train_100_epochs_plantdoc_od/weights/best.pt") 

    # get all annotation file names without the .txt extension
    raw_anno_names = list(map(lambda x:x.rpartition('.')[0], os.listdir(label_dir)))

    # columns
    img = []
    target_cls = []
    target_xywh = []
    bbox_xywh = []
    bbox_conf = []
    bbox_cls = []

    for img_name in raw_anno_names:
        # get path to img
        anno_path = label_dir + '/' + img_name + '.txt'
        img_path = img_dir + '/' + img_name + '.jpg'

        # grab target class and bbox xywh from label annotation file for image
        with open(anno_path) as f:
            tmp_clss = []
            tmp_xywhs = []
            labels = f.readlines()
            # for each line, gather class and bbox labels
            for label in labels:    
                cls, x, y, w, h = label.split()
                tmp_clss.append(int(cls))
                tmp_xywhs.append(list(map(lambda x:float(x), [x,y,w,h])))
            target_cls.append(tmp_clss)
            target_xywh.append(tmp_xywhs)
            
        # get bbox predictions from image
        result = model(source=img_path, imgsz=640)
        # gather results from each input image (only 1 input here)
        for r in result:
            boxes = r.boxes.cpu()
            # append results
            img.append(img_name)
            bbox_xywh.append(boxes.xywh.tolist())
            bbox_conf.append(boxes.conf.tolist())
            bbox_cls.append(boxes.cls.tolist())

    # create output csv file with target and predicted values
    results = pd.DataFrame()
    results["image"] = img
    results["target_xywh"] = target_xywh
    results["target_cls"] = target_cls
    results["xywh"] = bbox_xywh
    results["cls"] = bbox_cls
    results["conf"] = bbox_conf
    results.to_csv(f"{root}/yolo_results.csv", index=False)