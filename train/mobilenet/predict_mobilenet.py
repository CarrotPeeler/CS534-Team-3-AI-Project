import numpy as np
import pandas as pd

from tensorflow import keras
from keras.utils import load_img, img_to_array

if __name__ == "__main__":
    # set annotation csv path for cropped input images 
    anno_path = "/home/vislab-001/Jared/CS534-Team-3-AI-Project/crop_results.csv"

    output_anno_path = "/home/vislab-001/Jared/CS534-Team-3-AI-Project/train/mobilenet_results.csv"

    anno_df = pd.read_csv(anno_path)

    # set directory to checkpoint path (if .pb file, set path to entire parent folder of .pb; else if .h5, set path to .h5 file)
    checkpoint_path = "/home/vislab-001/Jared/CS534-Team-3-AI-Project/runs/classify/mobilenet"

    # set the model hyperparameters needed to build it
    num_classes = 25

    # set height and width of input image.
    img_width, img_height = 224, 224
    input_shape = (img_width, img_height, 3)

    # load model and saved weights 
    model = keras.models.load_model(checkpoint_path)
    print(model.summary())

    # predict all cropped images
    preds = []
    for idx, row in anno_df.iterrows():
        # load image
        img_path = row["image_path"]
        img = load_img(img_path, target_size=(img_width, img_height))
        #convert to array 
        img = img_to_array(img)
        #normalize image
        img = img/255.0
        #transform image array to a tensor
        img = np.expand_dims(img, axis=0)
        # Prediction step
        predictions = model.predict(img, batch_size=1)
        # get predicted class
        cls_pred = np.argmax(predictions, axis=1)
        preds.append(cls_pred)
    anno_df["pred_class"] = preds
    anno_df.to_csv(output_anno_path, index=False)
    

    