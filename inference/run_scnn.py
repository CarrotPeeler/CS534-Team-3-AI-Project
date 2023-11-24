# Import Libraries
import warnings

warnings.filterwarnings("ignore")

import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Keras API
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# from keras.utils import np_utils

# Get directories

test_dir = "valid"
train_dir = "train"


# function to get count of images

def get_files(directory):
    if not os.path.exists(directory):
        return 0
    count = 0
    for current_path, dirs, files in os.walk(directory):
        for dr in dirs:
            count += len(glob.glob(os.path.join(current_path, dr + "/*")))
    return count


train_samples = get_files(train_dir)
num_classes = len(glob.glob(train_dir + "/*"))
test_samples = get_files(test_dir)
print(num_classes, "Classes")
print(train_samples, "Train images")
print(test_samples, "Test images")

# Preprocessing data, Data Augmentation
train_datagen = ImageDataGenerator(rescale=1 / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1 / 255)
# print(np.size(train_datagen))

# set height and width of input image.
img_width, img_height = 256, 256
input_shape = (img_width, img_height, 3)
batch_size = 32

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size)
test_generator = test_datagen.flow_from_directory(test_dir, shuffle=False,
                                                  target_size=(img_width, img_height),
                                                  batch_size=batch_size)

# The name of the 15 diseases.
print(train_generator.class_indices)

# CNN building.

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
model.summary()

# model_layers = [layer.name for layer in model.layers]
# print('layer name : ', model_layers)

# validation data.
validation_generator = train_datagen.flow_from_directory(
    train_dir,  # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size)

# @title
# Model building to get trained with parameters.
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy()])
train = model.fit_generator(train_generator,
                            epochs=9,
                            steps_per_epoch=train_generator.samples // batch_size,
                            validation_data=validation_generator,
                            verbose=1)

acc = train.history['categorical_accuracy']
val_acc = train.history['val_categorical_accuracy']
loss = train.history['loss']
val_loss = train.history['val_loss']

epochs = range(1, len(acc) + 1)
# Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")

plt.figure()
# Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss Magnitude")
plt.legend(loc="upper right")
plt.show()

# Get the true labels and predicted labels for the test data
true_labels = test_generator.classes  # Updated line
predicted_labels = np.argmax(model.predict(test_generator), axis=1)

# Create a confusion matrix
confusion = confusion_matrix(true_labels, predicted_labels)

# Plot the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=test_generator.class_indices,
            yticklabels=test_generator.class_indices)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("./mobilenet_train_confmat")

# Generate and print the classification report
class_names = list(test_generator.class_indices.keys())
report = classification_report(true_labels, predicted_labels, target_names=class_names)
print(report)

score, accuracy = model.evaluate(test_generator, verbose=1)
print("Test score is {}".format(score))
print("Test accuracy is {}".format(accuracy))

# Save entire model with optimizer, architecture, weights and training configuration.
model.save('SCNN2epoch.h5')

plant_model = tf.keras.models.load_model('SCNN10epoch.h5')
plant_model.summary()
