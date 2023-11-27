import pandas as pd
import torch
from class_mappings import *
from ultralytics.utils.ops import xywh2xyxy
from ast import literal_eval
from train.utils import process_batch

if __name__ == "__main__":
    # compute final metrics (mAP50, mAP50-95, recall, precision, F1-score)
    # Since we use rotation augmentation to increase inference redundancy of a single cropped image,
    # we will ensemble results via mean

    # set annotation csv path for scnn results
    anno_path = "/home/vislab-001/Jared/CS534-Team-3-AI-Project/train/scnn_results.csv"

    anno_df = pd.read_csv(anno_path)
    
    # filter out images with no target information
    anno_df.drop(anno_df.index[anno_df["target_class"] == "[]"], inplace=True)

    # create subgroups for analysis based on rotation aug
    df_0 = anno_df.loc[anno_df["image_path"].str.rpartition('_rotated_')[2].str.rpartition('.jpg')[0] == "0"]
    df_90 = anno_df.loc[anno_df["image_path"].str.rpartition('_rotated_')[2].str.rpartition('.jpg')[0] == "90"]
    df_180 = anno_df.loc[anno_df["image_path"].str.rpartition('_rotated_')[2].str.rpartition('.jpg')[0] == "180"]
    df_270 = anno_df.loc[anno_df["image_path"].str.rpartition('_rotated_')[2].str.rpartition('.jpg')[0] == "270"]

    stats = []
    for idx, row in df_0.iterrows():
        # if idx < 100:
            # reconstruct detection data matrix but include new pred class
            x1,y1,x2,y2 = literal_eval(row["pred_xyxy"])[0]
            conf = literal_eval(row["conf"])[0]
            cls = literal_eval(row["pred_class"])[0]
            detections = [[x1,y1,x2,y2,conf,cls]]
            print(cls)
            
            # reconstruct label data matrix
            labels = []
            target_xywh = literal_eval(row["target_xywh"])
            target_class = literal_eval(row["target_class"])
            for i in range(len(target_xywh)):
                t_xyxy = xywh2xyxy(torch.tensor(target_xywh[i]))
                tx1,ty1,tx2,ty2 = t_xyxy
                tcls = target_class[i]
                # remap to NewPlantDiseases class idx
                tcls = cls_mapping[plantdoc_classes[tcls]]
                label = [tcls,tx1,ty1,tx2,ty2]
                labels.append(label)
                print("targ_npd",tcls)

            detections = torch.tensor(detections)
            labels = torch.tensor(labels)
            print(row["image_path"])
            correct_bboxes = process_batch(detections, labels)
            stats.append((correct_bboxes, detections[:, 4], detections[:, 5], labels[:, 0]))

