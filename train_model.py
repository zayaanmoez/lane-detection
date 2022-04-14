import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from dataset import DataSet
from lanenet import LaneNet
from process_data import process_dataset
from utils import get_config
import seaborn as sns

sns.set_theme()

TRAIN_DATASET = "train"
EX_TRAIN_DATASET = "ex_train"


# print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))

###############################################################################
# Metrics

fit_metrics = ['loss', 'binary_segmentation_loss', 'instance_segmentation_loss', 
        'val_loss', 'val_binary_segmentation_loss', 'val_instance_segmentation_loss',
        'accuracy', 'val_accuracy']


# Plot training data
def plot(fit_history):
    for key in fit_metrics:
        plt.plot(fit_history[key][0], fit_history[key][1])
        plt.title('model ' + key)
        plt.ylabel(key.replace('_', ' '))
        plt.xlabel('epoch')
        plt.show()
    plt.show()

###############################################################################
# Training Script

def train_model(dataset, config, model_name):
    # Configure gpu
    gpus = tf.config.list_physical_devices('GPU')
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


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
    fit_history = dict((metric, [[], []]) for metric in fit_metrics)

    epochs = config.params.ex_train.epochs if (dataset == EX_TRAIN_DATASET) else config.params.epochs
    for epoch in range(1, epochs):
        src_imgs, binary_imgs, instance_imgs = train_dataset.next_batch(config.params.batch_size)

        X_train = np.array(src_imgs)
        Y_train = [np.array(binary_imgs), np.array(instance_imgs)]

        # Validate and save model checkpoints every 100 epochs
        if epoch % 100 == 0:
            
            checkpoint_filepath = "./tmp/cp.ckpt"

            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                verbose=1,
                save_freq=1,
                monitor='val_loss')

            np.save('./tmp/fit_history', fit_history)

            # Validation data
            src_valid, binary_valid, instance_valid = train_dataset.next_batch(config.params.batch_size)
            X_valid = np.array(src_valid)
            Y_valid = [np.array(binary_valid), np.array(instance_valid)]

            # Model weights are saved if it's the best seen so far.
            history = model.fit(X_train, Y_train,
                validation_data=(X_valid, Y_valid),
                batch_size=config.params.batch_size,
                epochs=1,
                verbose='auto',
                callbacks=[model_checkpoint_callback])
            history = history.history
        else:
            history = model.train_on_batch(X_train, Y_train,
                return_dict=True,
                reset_metrics=(True if epoch % 10 == 1 else False))

        if epoch % 10 == 0:
            if epoch % 100 != 0: 
                print("Epoch: ", epoch)
                print(history)
            for key in history.keys():
                acc_metrics = ['binary_segmentation_accuracy', 'val_binary_segmentation_accuracy']
                value = history[key][0] if epoch % 100 == 0 else history[key]
                if key in acc_metrics:
                    fit_history[key.replace('binary_segmentation_', '')][0].append(epoch)
                    fit_history[key.replace('binary_segmentation_', '')][1].append(value)
                else:
                    fit_history[key][0].append(epoch)
                    fit_history[key][1].append(value)

        if epoch == 5000 or epoch == 10000 or epoch == 15000:
            plot(fit_history)

    # Save the model
    model.save(config.path.models+model_name)

    # Plot the training data    
    plot(fit_history)

    return

if __name__ == "__main__":
    config = get_config()
    train_model(EX_TRAIN_DATASET, config, model_name="model_test")

    # Load and Save checkpoint
    # checkpoint_filepath = "./tmp/cp.ckpt"
    
    # lanenet = LaneNet(training=True, params=config.params)
    # model = lanenet._build_model((config.params.input_height, config.params.input_width, 3))

    # model.load_weights(checkpoint_filepath)
    # model.save(config.path.models+"model#2")