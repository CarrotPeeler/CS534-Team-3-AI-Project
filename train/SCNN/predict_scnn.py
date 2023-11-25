import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import load_img, img_to_array

if __name__ == "__main__":
    # set annotation csv path for cropped input images 
    anno_path = "/home/vislab-001/Jared/CS534-Team-3-AI-Project/crop_results.csv"

    output_anno_path = "/home/vislab-001/Jared/CS534-Team-3-AI-Project/train/SCNN/scnn_results.csv"

    anno_df = pd.read_csv(anno_path)

    # set directory to checkpoint path
    checkpoint_path = "/home/vislab-001/Jared/CS534-Team-3-AI-Project/demo/SCNN/SCNN9epoch.h5"

    # set the model hyperparameters needed to build it
    num_classes = 25

    # set height and width of input image.
    img_width, img_height = 256, 256
    input_shape = (img_width, img_height, 3)

    # Build the Sequential CNN backbone
    model = Sequential()

    model.add(Conv2D(32, (5, 5), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    # load pretrained model weights
    model.load_weights(checkpoint_path)

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
    anno_df.to_csv("./train/scnn_results.csv", index=False)
    

    