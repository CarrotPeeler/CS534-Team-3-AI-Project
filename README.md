# CS534-Team-3-AI-Project
Something something here...

### Dependencies
- idk

### Datasets Used
Download the following datasets:
- [Augmented PlantVillage Dataset for Image Classification](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
  - NOTE: PlantVillage uses plant images taken in a labratory, unlike PlantDoc which uses photos of plants out in nature; PlantVillage provides more data overall, however
- [PlantDoc Datset for Image Classification](https://github.com/pratikkayal/PlantDoc-Dataset.git)
- [PlantDoc Datset for Object Detection](https://universe.roboflow.com/joseph-nelson/plantdoc)

### Data Preprocessing
Run YOLOv8 inference over PlantDoc Object Detection test split:
```
python3  
```

### Training
Train **YOLOv8** in the background: 
```
cd preprocessing/YOLOv8
tmux
python3 scripts/train_net.py < /dev/null > train_log.txt 2>&1 &
```

### Inference

