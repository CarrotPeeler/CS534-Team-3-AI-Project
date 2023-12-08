import os
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
#Add Callbacks, e.g. ModelCheckpoints, earlystopping, csvlogger.
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.applications import DenseNet121
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization, GlobalAveragePooling2D, AveragePooling2D


if __name__ == "__main__":
    # params for dataset directory and image size
    base_dir = os.getcwd() + "/data/New_Plant_Disease_Dataset"
    # dir for saving checkpoints and results
    save_dir = "/home/vislab-001/Jared/CS534-Team-3-AI-Project/runs/classify/densenet"

    # hyperparameters
    image_size = 224
    batch_size = 32
    epochs = 9
    init_lr = 1e-3 # 1E-3

    os.makedirs(save_dir, exist_ok=True)

    # create train images preprocesser (includes augmentation step, etc.)
    train_datagen = ImageDataGenerator(
                            rescale = 1/255.0,
                            rotation_range=45,          # Randomly rotates images in range [-45, 45] degrees
                            width_shift_range=0.1,
                            height_shift_range=0.1, 
                            shear_range=0.2, 
                            zoom_range=0.2,
                            horizontal_flip=True,
                            vertical_flip=True,
                            fill_mode="nearest"         # fill newly created missing pixels during the image transformation
                        )
    # create test images preprocesser (only normalizes pixel values)
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.0)

    # create datatsets for train and test images automatically by scraping folders with samples
    # each folder pertains to a unique class
    train_data = train_datagen.flow_from_directory(os.path.join(base_dir,"train"),
                                                shuffle=True,
                                                target_size=(image_size,image_size),
                                                batch_size=batch_size,
                                                class_mode="categorical"                                               
                                                )

    test_data = test_datagen.flow_from_directory(os.path.join(base_dir,"valid"),
                                                shuffle=False,
                                                target_size=(image_size,image_size),
                                                batch_size=batch_size,
                                                class_mode="categorical"                                               
                                                )
    # print class corresponding indices and image shape
    print(f"Number of classes {len(train_data.class_indices)}")
    print(train_data.class_indices)
    
    densenet = DenseNet121(weights='imagenet', include_top=False)

    input = keras.Input(shape=(image_size, image_size, 3))
    x = Conv2D(3, (3, 3), padding='same')(input)
    
    x = densenet(x)
    
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    output = Dense(25,activation = 'softmax', name='root')(x)

    model = Model(input,output)

    # @title
    # Model building to get trained with parameters.
    opt = Adam(
        learning_rate=init_lr, 
        beta_1=0.9, 
        beta_2=0.999, 
        epsilon=0.1, 
        decay=init_lr / epochs
    )
    annealer = ReduceLROnPlateau(
        monitor='val_accuracy', 
        factor=0.5, 
        patience=5, 
        verbose=1, 
        min_lr=1e-3
    )

    # compile final model with adam optimizer and loss function (cross entropy)
    model.compile(optimizer=keras.optimizers.Adam(),
                            loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                            metrics=[keras.metrics.CategoricalAccuracy()])

    # EarlyStopping callback.
    early_stop = EarlyStopping(monitor='val_loss', 
                            patience=3, 
                            verbose=1)

    callbacks_list = [early_stop]

    # train model for 5 epochs
    history = model.fit(train_data,
                        steps_per_epoch=300,  
                        validation_data=test_data,
                        epochs=epochs,
                        validation_steps=300,
                        callbacks=callbacks_list)
    # final evaluation after training
    model.evaluate(test_data)
    # save model weights to checkpoint
    model.save(save_dir)

    # Learning Curves.
    # Plot Loss vs Accuracy graphs.
    print(history)
    plt.figure(figsize=(18,7))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label = 'Training Loss')
    plt.plot(history.history['val_loss'], label = 'Validation Loss')
    plt.grid(False)
    plt.xlabel('Epochs')
    plt.ylabel('Loss Magnitude')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['categorical_accuracy'], label = 'Training Accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label = 'validation Accuracy')
    plt.grid(False)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(f"{save_dir}/mobilenet_train_results.png")

    # Get the true labels and predicted labels for the test data
    true_labels = test_data.classes  # Updated line
    predicted_labels = np.argmax(model.predict(test_data), axis=1)

    # Create a confusion matrix
    confusion = confusion_matrix(true_labels, predicted_labels)

    # Plot the confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=test_data.class_indices, yticklabels=test_data.class_indices)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f"{save_dir}/mobilenet_train_confmat")

    # Generate and print the classification report
    class_names = list(test_data.class_indices.keys())
    report = classification_report(true_labels, predicted_labels, target_names=class_names)
    print(report)

    # save report to .txt file
    with open(f"{save_dir}/report.txt", "w+") as f:
        f.writelines(report)




