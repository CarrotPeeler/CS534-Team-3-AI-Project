# CS534-Team-3-AI-Project
Project on developing a pipeline for Plant Disease Detection in natural environments. 

## Dependencies
- Imutils
- PyTorch
- cv2 (Open-CV)
- Keras/Tensorflow
- Pandas
- Ultralytics
- Numpy

## Datasets Used
Download the following datasets:
- [Augmented PlantVillage Dataset for Image Classification](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/)
  - NOTE: PlantVillage uses plant images taken in a labratory, unlike PlantDoc which uses photos of plants out in nature; PlantVillage provides more data overall, however
- [PlantDoc Datset for Image Classification](https://github.com/pratikkayal/PlantDoc-Dataset.git)
- [PlantDoc Datset for Object Detection](https://universe.roboflow.com/joseph-nelson/plantdoc) (NOTE: cURL is recommended for downloading)

## Demo
For a quick demo on how the pipeline works, first download the SCNN pretrained weights from [here](https://wpi0-my.sharepoint.com/:u:/g/personal/jchan3_wpi_edu/EesXPbRY35VKn2RfM50LG3oBOqgrZSKBrZEOmJm1uymoPA?e=Da4Qq0) (Must have access to a WPI email account).

Place the checkpoint file in the demo/SCNN folder => demo/SCNN/SCNN9epoch.h5

Run YOLOv8 over the single test image:
```
python3 demo/SCNN/demo_yolo.py
```
This script will generate bounding boxes for the single image and then crop/preprocess them accordingly. After preprocessing, the saved images will appear in the preprocessed_images/class_samples folder.

Then, run the Sequential CNN for fine-grained, disease classification:
```
python3 demo/SCNN/demo_scnn.py
```
The terminal should display the predicted classes for each photo in order. There should be 16 photos and predictions total. For reference, the correct target class for the potato blight photo given is class 12.

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

