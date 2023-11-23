# Removes classes with no diseased samples 
# (to make evaluation only for disease detection, not plant detection)

import os

if __name__ == "__main__":
    filtered_classes = [1,2,3,4,5,7,8,9,10]
    root = "/home/vislab-001/Jared/CS534-Team-3-AI-Project"
    # file path to the annotations for each image
    data_dir = f"{root}/data/PlantDoc_ObjectDetection/test"
    img_dir = f"{data_dir}/images"
    label_dir = f"{data_dir}/labels"

    filtered_img_dir = f"{data_dir}/filtered_images"
    filtered_label_dir = f"{data_dir}/filtered_labels"

    os.makedirs(filtered_img_dir, exist_ok=True)
    os.makedirs(filtered_label_dir, exist_ok=True)

    imgs_to_move = []
    labels_to_move = []

    # check for any images/labels that are apart of the classes to be filtered
    for label_name in os.listdir(label_dir):
        # reconstruct label and corresponding image path
        label_path = f"{label_dir}/{label_name}"
        img_path = f"{img_dir}/{label_name.rpartition('.txt')[0]}.jpg"
        # iterate through each bounding box in label file and check if class matches those to be filtered
        with open(label_path) as f:
            for label in f.readlines():
                class_idx = int(label.partition(' ')[0])
                if class_idx in filtered_classes:
                    imgs_to_move.append(img_path)
                    labels_to_move.append(label_path)
                    break

    # move filtered images and labels to different folders
    assert len(imgs_to_move) == len(labels_to_move), "Error: length mismatch for filtered images and labels"

    for i in range(len(labels_to_move)):
        # fetch old paths and construct new paths for images and labels
        label_path = labels_to_move[i]
        label_new_path = f"{filtered_label_dir}/{labels_to_move[i].rpartition('/')[-1]}"

        img_path = imgs_to_move[i]
        img_new_path = f"{filtered_img_dir}/{imgs_to_move[i].rpartition('/')[-1]}"

        # move images and labels to new paths
        os.rename(label_path, label_new_path)
        os.rename(img_path, img_new_path) 

