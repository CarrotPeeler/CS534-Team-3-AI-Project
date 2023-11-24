import os
import cv2
import imutils
from ultralytics.models import YOLO
from ultralytics.utils.plotting import save_one_box

if __name__ == "__main__":
    # specify root dir of project
    root = "/home/vislab-001/Jared/CS534-Team-3-AI-Project"

    # directory where preprocessed images are saved 
    crop_save_dir = "/home/vislab-001/Jared/CS534-Team-3-AI-Project/demo/SCNN/preprocessed_images"
    os.makedirs(crop_save_dir, exist_ok=True)

    # directory where YOLOv8 results are saved
    yolo_results_save_dir = f"{root}/demo/SCNN/yolo_results"
    os.makedirs(yolo_results_save_dir, exist_ok=True)

    # load pretrained model checkpoint
    model = YOLO(f"{root}/runs/detect/downscaled_train_100_epochs_plantdoc_od/weights/best.pt") 

    # path to single image for demo 
    img_path = f"{root}/demo/SCNN/orig_image/backus-056-potato-blight_jpg.rf.70d8ec2e2694286e3c6093b405d2c7e5.jpg"

    # get bbox predictions from image
    result = model(source=img_path, imgsz=640, project=yolo_results_save_dir, save=True, save_crop=False)

    # process bounding box results
    for r in result:
        boxes = r.boxes.cpu()
        # x,y center coord of bbox; w,h are width, height of box
        bboxes_xyxy = boxes.xyxy

    # read in the original image for preprocessing steps
    img = cv2.imread(img_path)

    # perfom preprocessing (crop, rotate, resize) on each bbox in image
    for i, xyxy in enumerate(bboxes_xyxy):
        print(xyxy)
        cropped_img = save_one_box(xyxy, img, BGR=True, save=False)

        # generate 4 copies of cropped image, rotated in 90 deg increments
        for angle in [0, 90, 180, 270]:
            save_path = f"{crop_save_dir}/{img_path.rpartition('/')[-1].rpartition('.jpg')[0]}_{i}_{angle}.jpg"

            # save crop if not bad image; also resize and rotate
            try:
                # resize crop
                cropped_img = cv2.resize(cropped_img, (256,256))
                # rotate
                cropped_img = imutils.rotate(cropped_img, angle)
                # save
                cv2.imwrite(save_path, cropped_img)
            except:
                print("Failed to preprocess image.")
        
    

