# Import Libraries

import os
import glob
import statistics as st
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

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
model.load_weights('SCNN9epoch.h5')

# set the directory where test sampls are located
predict_dir = "SCNN_predict"

def get_files(directory):
    if not os.path.exists(directory):
        return 0
    count = 0
    for current_path, dirs, files in os.walk(directory):
        for dr in dirs:
            count += len(glob.glob(os.path.join(current_path, dr + "/*")))
    return count

# get the test images
predict_samples = get_files(predict_dir)

print(predict_samples, "Prediction images")

# Get normalization preprocessing function
predict_datagen = ImageDataGenerator(rescale=1 / 255)

# Perform preprocessing over all test images and get dataset object
predict_generator = predict_datagen.flow_from_directory(predict_dir, shuffle=False,
                                                        target_size=(img_width, img_height))
# Prediction step
predictions = model.predict(predict_generator)
classes = np.argmax(predictions, axis=1)
print(classes)

classes = np.array(classes)
print(st.mode(classes))
