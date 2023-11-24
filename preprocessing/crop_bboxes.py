import os
import cv2
import ast
import pandas as pd
import imutils
import torch
from ultralytics.utils.plotting import save_one_box

if __name__ == "__main__":
    # directory where cropped images will be saved to 
    crop_save_dir = "/home/vislab-001/Jared/CS534-Team-3-AI-Project/data/Cropped_PlantDoc"
    # path to csv which has yolo results including bbox xywh coordinates
    bbox_anno_csv_path = "/home/vislab-001/Jared/CS534-Team-3-AI-Project/yolo_results.csv"

    # make directory if it doesn't already exist
    os.makedirs(crop_save_dir, exist_ok=True)

    # read in bbox annotation file from inference
    csv_df = pd.read_csv(bbox_anno_csv_path)

    # prepare dataframe/csv for output cropped image information
    anno_df = pd.DataFrame()
    img_paths = []
    target_class = []
    pred_xywh = []
    target_xywh = []
    conf = []

    # iterate through yolo results for each image in PlantDoc test dataset
    for idx, row in csv_df.iterrows():
        # read in image from file path
        img_path = row["image_path"]
        img = cv2.imread(img_path)
        height, width, rgb = img.shape
        # extract csv information for image
        pred_bboxes = ast.literal_eval(row["xyxy"])
        target_bboxes = ast.literal_eval(row["target_xywh"])
        # make sure to unnormalize the target bboxes for easier comparison w/ unnormalized pred boxes
        for target_bbox in target_bboxes:
            t_x, t_y, t_w, t_h = target_bbox
            target_bbox = [t_x * width, t_y*height, t_w*width, t_h*height]

        target_disease = ast.literal_eval(row["target_disease"]) 
        obj_conf = ast.literal_eval(row["conf"])

        for i in range(len(pred_bboxes)):
            # generate 4 copies of cropped image, rotated in 90 deg increments
            for angle in [0, 90, 180, 270]:
                save_path = f"{crop_save_dir}/{img_path.rpartition('/')[-1].rpartition('.jpg')[0]}_bbox_{i}_rotated_{angle}.jpg"
                # crop image
                xyxy = torch.tensor(pred_bboxes[i])
                cropped_img = save_one_box(xyxy, img, BGR=True, save=False)

                # save crop if not bad image; also resize and rotate
                try:
                    # resize crop
                    cropped_img = cv2.resize(cropped_img, (256,256))
                    # rotate
                    cropped_img = imutils.rotate(cropped_img, angle)
                    cv2.imwrite(save_path, cropped_img)
                except:
                    continue
                # append csv information for this particular crop
                img_paths.append(save_path)
                target_class.append(target_disease)
                conf.append(obj_conf)
                pred_xywh.append(pred_bboxes)
                target_xywh.append(target_bboxes)

    # create columns of dataframe
    anno_df["image_path"] = img_paths
    anno_df["target_class"] = target_class
    anno_df["pred_xywh"] = pred_xywh
    anno_df["target_xywh"] = target_xywh
    # save dataframe as csv
    anno_df.to_csv("crop_results.csv", index=False)


