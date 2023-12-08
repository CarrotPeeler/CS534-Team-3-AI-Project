import pandas as pd
import numpy as np
import torch
from class_mappings import *
from ultralytics.utils.ops import xywh2xyxy
from ast import literal_eval
from train.utils import process_batch, ap_per_class, mp, mr, map50, map50_95
from typing import List, Tuple


def get_stats(df:pd.DataFrame, train_dataset:str) -> List[Tuple]:
    """
    For each predicted image, parse binary true-positive array for iou 0.5:0.95, 
    detection confidence, detection class, and target class

    params:
        df: dataframe with columns [image_path,target_class,pred_xyxy,target_xywh,conf,pred_class]
        train_dataset: either "NewPlantDiseases" or "PlantDoc"; used for class idx remapping
    returns:
        list of tuples (true_pos_array, confidence, pred_class, target_class)
    """
    stats = []
    for idx, row in df.iterrows():
        # construct detection data matrix but include new pred class
        x1,y1,x2,y2 = literal_eval(row["pred_xyxy"])[0]
        conf = literal_eval(row["conf"])[0]
        cls = literal_eval(row["pred_class"])[0]
        detections = [[x1,y1,x2,y2,conf,cls]]
        # print(cls)
        
        # construct label data matrix
        labels = []
        target_xywh = literal_eval(row["target_xywh"])
        target_class = literal_eval(row["target_class"])
        for i in range(len(target_xywh)):
            t_xyxy = xywh2xyxy(torch.tensor(target_xywh[i]))
            tx1,ty1,tx2,ty2 = t_xyxy
            tcls = target_class[i]

            if train_dataset == "NewPlantDiseases":
                # remap PlantDoc Object Detection class idx to NewPlantDiseases class idx
                tcls = plantdoc_od_to_npds_mapping[plantdoc_od_idxs_mapping[tcls]]
            elif train_dataset == "PlantDoc":
                # remap PlantDoc Object Detection class idx to PlantDoc Classification class idx
                tcls = plantdoc_cls_name_mapping[plantdoc_od_idxs_mapping[tcls]]

            label = [tcls,tx1,ty1,tx2,ty2]
            labels.append(label)
            # print("targ_npd",tcls)

        detections = torch.tensor(detections)
        labels = torch.tensor(labels)
        # print(row["image_path"])
        correct_bboxes = process_batch(detections, labels)
        stats.append((correct_bboxes, detections[:, 4], detections[:, 5], labels[:, 0]))
    return stats


def get_metrics(stats:List[Tuple]) -> Tuple:
    """
    Computes mean precision, mean recall, mAP50, and mAP50-95

    params:
        stats: list of tuples (true_pos_array, confidence, pred_class, target_class)
    returns:
        tuple of arrays (precision, recall, map50, map50-95)
    """
    # Zip tuple columns together to create four numpy arrays
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)] 

    # parse out individual arrays
    tp, conf, pred_cls, target_cls = stats

    # get tp, fp, p, r, f1, ap
    results = ap_per_class(tp, conf, pred_cls, target_cls)
    
    # compute mean precision, recall, map50, and map50-95
    precision = mp(results[2])
    recall = mr(results[3])
    map_50 = map50(results[5])
    map_50_95 = map50_95(results[5])
    return precision, recall, map_50, map_50_95


if __name__ == "__main__":
    # compute final metrics (mAP50, mAP50-95, recall, precision)
    # Since we use rotation augmentation to increase inference redundancy of a single cropped image,
    # we will ensemble results via mean

    # set annotation path to the csv for the model you would like to obtain metrics for
    anno_path = "/home/vislab-001/Jared/CS534-Team-3-AI-Project/train/mobilenet_results.csv"
    # train dataset (either "PlantDoc" or "NewPlantDiseases"); use PlantDoc for finetuned models and NewPlantDiseases otherwise
    # train_dataset = "PlantDoc"
    train_dataset = "NewPlantDiseases"

    anno_df = pd.read_csv(anno_path)
    
    # filter out images with no target information
    anno_df.drop(anno_df.index[anno_df["target_class"] == "[]"], inplace=True)

    # create subgroups for analysis based on rotation augment
    df_0 = anno_df.loc[anno_df["image_path"].str.rpartition('_rotated_')[2].str.rpartition('.jpg')[0] == "0"]
    df_90 = anno_df.loc[anno_df["image_path"].str.rpartition('_rotated_')[2].str.rpartition('.jpg')[0] == "90"]
    df_180 = anno_df.loc[anno_df["image_path"].str.rpartition('_rotated_')[2].str.rpartition('.jpg')[0] == "180"]
    df_270 = anno_df.loc[anno_df["image_path"].str.rpartition('_rotated_')[2].str.rpartition('.jpg')[0] == "270"]

    dfs = [df_0, df_90, df_180, df_270]

    # record metrics for each augmented batch of predictions
    all_metrics = []

    # gather metrics for each augmented batch of predictions
    for df in dfs:
        stats = get_stats(df, train_dataset)
        metrics = get_metrics(stats)
        all_metrics.append(metrics)
    
    # perform mean ensemble of results
    ensemble_metrics = list(zip(*all_metrics))
    ensemble_metrics = list(map(lambda x:np.array(x).mean(), ensemble_metrics))
    
    # print final metrics
    print(f"Precision: {ensemble_metrics[0]:0.3f},\
            Recall: {ensemble_metrics[1]:0.3f},\
            mAP50: {ensemble_metrics[2]:0.3f},\
            mAP50-95: {ensemble_metrics[3]:0.3f}")
