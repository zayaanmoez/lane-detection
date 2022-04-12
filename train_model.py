from tabnanny import verbose
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
from dataset import DataSet
from lanenet import LaneNet
from process_data import process_dataset
from utils import get_config

TRAIN_DATASET = "train"
EX_TRAIN_DATASET = "ex_train"


print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))

###############################################################################
# Metrics

fit_metrics = ['loss', 'binary_segmentation_loss', 'instance_segmentation_loss', 
        'val_loss', 'val_binary_segmentation_loss', 'val_instance_segmentation_loss']


# Plot training data
def plot(fit_history):
    for key in fit_metrics:
        plt.plot(fit_history[key])
        plt.title('model ' + key)
        plt.ylabel(key)
        plt.xlabel('epoch')
        plt.show()
    plt.show()

###############################################################################
# Training Script

def train_model(dataset, config):
    path_root =  config.path.root[dataset]

    # Create the dataset if it doesn't exist
    if (os.path.isdir(path_root) and path_root+dataset+"_data.txt"
        and os.path.isdir(path_root+config.path.image.src)
        and os.path.isdir(path_root+config.path.image.binary)
        and os.path.isdir(path_root+config.path.image.instance)): 
        print("Dataset already exists. Skipping dataset creation.")
    else:
        print("Creating dataset.")
        process_dataset(dataset)
    
    # Load the dataset
    train_dataset = DataSet(dataset, config)
    train_dataset.load_dataset()

    # Create the model
    lanenet = LaneNet(training=True, params=config.params)
    model = lanenet._build_model((config.params.input_height, config.params.input_width, 3))
    model.summary()

    # Record the training history
    fit_history = dict((metric, []) for metric in fit_metrics)

    epochs = config.params.ex_train.epochs if (dataset == EX_TRAIN_DATASET) else config.params.epochs
    for epoch in range(epochs):
        src_imgs, binary_imgs, instance_imgs = train_dataset.next_batch(config.params.batch_size)
        src_valid, binary_valid, instance_valid = train_dataset.next_batch(config.params.batch_size)

        X_train = np.array(src_imgs)
        Y_train = [np.array(binary_imgs), np.array(instance_imgs)]

        X_valid = np.array(src_valid)
        Y_valid = [np.array(binary_valid), np.array(instance_valid)]

        if epoch % 100 == 0:
            checkpoint_filepath = "./tmp/cp.ckpt"

            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                verbose=1,
                save_freq=1,
                monitor='val_loss')

            np.save('./tmp/fit_history.txt', fit_history)

            # Model weights are saved if it's the best seen so far.
            history = model.fit(X_train, Y_train,
                validation_data=(X_valid, Y_valid),
                batch_size=config.params.batch_size,
                verbose='auto',
                callbacks=[model_checkpoint_callback])
        else:
            history = model.fit(X_train, Y_train, 
                validation_data=(X_valid, Y_valid),
                batch_size=config.params.batch_size,
                verbose='auto')

        for key in history.history.keys():
            fit_history[key].append(history.history[key][0])

    # Save the model
    model.save("models/model#")

    # Plot the training data    
    plot(fit_history)

    return

if __name__ == "__main__":
    config = get_config()
    train_model(TRAIN_DATASET, config)