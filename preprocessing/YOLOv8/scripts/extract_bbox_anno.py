import os
import pandas as pd
from ultralytics.models import YOLO

if __name__ == "__main__":
    root = "/home/vislab-001/Jared/CS534-Team-3-AI-Project"
    # file path to the annotations for each image
    data_dir = f"{root}/data/PlantDoc_ObjectDetection/test"
    img_dir = f"{data_dir}/images"
    label_dir = f"{data_dir}/labels"
    label_fine_dir = f"{data_dir}/labels_old"

    # load pretrained model checkpoint
    model = YOLO(f"{root}/runs/detect/downscaled_train_100_epochs_plantdoc_od/weights/best.pt") 

    # get all annotation file names without the .txt extension
    raw_anno_names = list(map(lambda x:x.rpartition('.')[0], os.listdir(label_dir)))

    # columns
    img_paths = []
    target_species = []
    target_disease = []
    target_xywh = []
    bbox_xywh = []
    bbox_conf = []
    bbox_cls = []

    for img_name in raw_anno_names:
        # get path to img
        anno_path = label_dir + '/' + img_name + '.txt'
        anno_fine_path = label_fine_dir + '/' + img_name + '.txt'
        img_path = img_dir + '/' + img_name + '.jpg'

        # grab species target class and bbox xywh from label annotation file for image
        with open(anno_path) as f:
            tmp_clss = []
            tmp_xywhs = []
            labels = f.readlines()
            # for each line, gather class and bbox labels
            for label in labels:    
                cls, x, y, w, h = label.split()
                tmp_clss.append(int(cls))
                tmp_xywhs.append(list(map(lambda x:float(x), [x,y,w,h])))
            target_species.append(tmp_clss)
            target_xywh.append(tmp_xywhs)
            
        # grab fine-grained, disease/condition of species in image
        with open(anno_fine_path) as f_fine:
            tmp_fine_clss = []
            labels = f_fine.readlines()
            # for each line, gather class and bbox labels
            for label in labels:  
                cls_fine, x, y, w, h = label.split()
                tmp_fine_clss.append(int(cls_fine))
            target_disease.append(tmp_fine_clss)
            
        # get bbox predictions from image
        result = model(source=img_path, imgsz=640)
        # gather results from each input image (only 1 input here)
        for r in result:
            boxes = r.boxes.cpu()
            # append results
            img_paths.append(img_path)
            # x,y center coord of bbox; w,h are width, height of box
            bbox_xywh.append(boxes.xywh.tolist())
            # confidence scores/probabilites of how correct predicted classes may be
            bbox_conf.append(boxes.conf.tolist())
            # predicted classes
            bbox_cls.append(boxes.cls.tolist())

    # create output csv file with target and predicted values
    results = pd.DataFrame()
    results["image_path"] = img_paths
    results["target_xywh"] = target_xywh
    results["target_species"] = target_species
    results["target_disease"] = target_disease
    results["xywh"] = bbox_xywh
    results["cls"] = bbox_cls
    results["conf"] = bbox_conf
    results.to_csv(f"{root}/yolo_results.csv", index=False)