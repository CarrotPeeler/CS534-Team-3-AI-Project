import os
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
#Add Callbacks, e.g. ModelCheckpoints, earlystopping, csvlogger.
from keras.callbacks import EarlyStopping


if __name__ == "__main__":
    # params for dataset directory and image size
    base_dir = os.getcwd() + "/data/PlantDoc_Classification"
    # dir for saving checkpoints and results
    save_dir = "/home/vislab-001/Jared/CS534-Team-3-AI-Project/runs/classify/mobilenet_finetuned"

    # set directory to checkpoint path (if .pb file, set path to entire parent folder of .pb; else if .h5, set path to .h5 file)
    checkpoint_path = "/home/vislab-001/Jared/CS534-Team-3-AI-Project/runs/classify/mobilenet"

    # hyperparameters
    image_size = 224
    batch_size = 32
    epochs = 30

    os.makedirs(save_dir, exist_ok=True)

    # create train images preprocesser (includes augmentation step, etc.)
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.0,
                                                                shear_range = 0.2,
                                                                zoom_range = 0.2,
                                                                width_shift_range = 0.2,
                                                                height_shift_range = 0.2,
                                                                fill_mode="nearest",
                                                                horizontal_flip=True,
                                                                rotation_range=90)
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

    test_data = test_datagen.flow_from_directory(os.path.join(base_dir,"test"),
                                                shuffle=False,
                                                target_size=(image_size,image_size),
                                                batch_size=batch_size,
                                                class_mode="categorical"                                               
                                                )
    steps_per_epoch = train_data.n//batch_size
    validation_steps = test_data.n//batch_size

    # print class corresponding indices and image shape
    print(f"Number of classes {len(train_data.class_indices)}")
    print(train_data.class_indices)

    # load pretrained model and saved weights 
    pretrained_model = keras.models.load_model(checkpoint_path)

    # remove last dense layer to create base model
    base_model = keras.Model(pretrained_model.inputs, pretrained_model.layers[-2].output, name="base_model")
    print(base_model.summary())

    # Freeze the base_model parameters
    base_model.trainable = False
    
    # Create new model on top 
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    outputs = keras.layers.Dense((len(train_data.class_indices)),activation="softmax")(x)
    mobilenet_model = keras.Model(inputs, outputs, name='leaf_disease_model_mobilenet')
    print(mobilenet_model.summary())
    
    # compile final model with adam optimizer and loss function (cross entropy)
    mobilenet_model.compile(optimizer=keras.optimizers.Adam(),
                            loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                            metrics=[keras.metrics.CategoricalAccuracy()])

    # EarlyStopping callback.
    early_stop = EarlyStopping(monitor='val_loss', 
                            patience=3, 
                            verbose=1)

    callbacks_list = [early_stop]

    # train model for 5 epochs
    history = mobilenet_model.fit(train_data,
                                steps_per_epoch=steps_per_epoch,  
                                validation_data=test_data,
                                epochs=epochs,
                                validation_steps=validation_steps,
                                callbacks=callbacks_list)
    # final evaluation after training
    mobilenet_model.evaluate(test_data)
    # save model weights to checkpoint
    mobilenet_model.save(save_dir)

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
    predicted_labels = np.argmax(mobilenet_model.predict(test_data), axis=1)

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




