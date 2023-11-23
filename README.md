# CS534-Team-3-AI-Project
Something something here...

## Dependencies
- idk

## Datasets Used
Download the following datasets:
- [Augmented PlantVillage Dataset for Image Classification](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/)
  - NOTE: PlantVillage uses plant images taken in a labratory, unlike PlantDoc which uses photos of plants out in nature; PlantVillage provides more data overall, however
- [PlantDoc Datset for Image Classification](https://github.com/pratikkayal/PlantDoc-Dataset.git)
- [PlantDoc Datset for Object Detection](https://universe.roboflow.com/joseph-nelson/plantdoc) (NOTE: cURL is recommended for downloading)

## Running YOLOv8
### Data Preprocessing
Reformat annotations to only reflect species and not disease
```
python3 preprocessing/downscale_annotations.py
```
Reformat test data to only contain classes with multiple diseases (to evaluate disease detection and not plant detection)
```
python3 preprocessing/reformat_test_data.py
```

### Training
Train **YOLOv8** in the background: 
```
cd preprocessing/YOLOv8
tmux
python3 scripts/train_net.py < /dev/null > train_log.txt 2>&1 &
```
Can also run optional hyperparameter tuning running the following:
```
cd preprocessing/YOLOv8
tmux
python3 scripts/tune.py < /dev/null > tune_log.txt 2>&1 &
```

### Obtaining Bounding Boxes
```
cd preprocessing/YOLOv8
python3 scripts/extract_bbox_anno.py 
```

### Cropping Bounding Boxes
```
cd preprocessing/YOLOv8
python3 crop_bboxes.py
```

## Running Fine-grained Classification

