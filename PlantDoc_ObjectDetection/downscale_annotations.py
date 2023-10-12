# Script to group fine-grained subclasses into umbrella classes for YOLO bbox detections

import os
import yaml

def group_fine_grained_classes(data):
    """
    Groups fine-grained/similar classes together

    params:
        data: dict with yaml data
    returns:
        dict (key = class group label, value = list of classes in a group)
    """
    class_groups_dict = {}
    for class_idx, class_name in enumerate(data["names"]):
        # Annotation error, Soyabean and Soybean should be same class
        if class_name == "Soyabean leaf":
            umbrella_class_name = "Soybean"
        else:
            umbrella_class_name = class_name.partition(" ")[0].strip()
        class_groups_dict.setdefault(umbrella_class_name, []).append((class_name, class_idx))
    return class_groups_dict


def remap_class_idxs(label_dirs, class_groups_dict):
    """
    Re-label images based on prior grouping constraints

    params:
        label_dirs: list of string paths for each split dir (train, val, test)
        class_group_dict: dict (key = class group label, value = list of classes in group)
    """
    for label_dir in label_dirs:
        # rename label dir to old since a new label dir will be made
        dir_rename = label_dir.rpartition('/')[0] + "/labels_old"
        os.rename(label_dir, dir_rename)

         # grab paths to old annotations
        label_anno_paths = list(map(lambda x: dir_rename + '/' + x, os.listdir(dir_rename)))

        # create dir for updated annos
        os.mkdir(label_dir)

        # iterate through each annotation file and update the class label according to remapping
        for anno_path in label_anno_paths:
            anno_name = anno_path.rpartition('/')[-1]
            # go through old anno file and parse out data
            new_lines = ""
            with open(anno_path) as f_anno:
                # read old lines and parse out label
                for line in f_anno.readlines():
                    class_idx = int(line.partition(' ')[0])
                    # check mapping for new label
                    for group_idx, (key, value) in enumerate(class_groups_dict.items()):
                        if class_idx in list(map(lambda x:x[1], value)):
                            # rewrite line with updated class label
                            new_line = f"{group_idx} {line.partition(' ')[-1]}"
                            new_lines += new_line

            # write updated data to new annotation file
            with open(label_dir + '/' + anno_name, "w+") as f_new_anno:
                f_new_anno.seek(0)
                f_new_anno.write(new_lines)
                        

def fix_anno(yaml_filepath):
    """
    Groups fine-grained/similar classes together under a new umbrella class index

    params:
        yaml_filepath: path to yaml file for the data
    returns:
        Returns an array, where each index represents the new class index;
        each sub list groups old class indices together under a new umbrella class index
    """
    with open(yaml_filepath) as f:
        data = yaml.safe_load(f)

    # get remapped classes
    class_groups_dict = group_fine_grained_classes(data)

    # get train, test, val label directory
    yaml_root = yaml_filepath.rpartition('/')[0]
    label_dirs = [yaml_root + "/" + split + "/labels" for split in ["train", "valid", "test"]]

    remap_class_idxs(label_dirs, class_groups_dict)
    

if __name__ == "__main__":
    yaml_path = "/home/vislab-001/Jared/CS534-Team-3-AI-Project/PlantDoc_ObjectDetection/plantdoc_objectdetection.yaml"
    fix_anno(yaml_path)