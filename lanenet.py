import tensorflow as tf
import keras.layers as layers
import numpy as np
from collections import OrderedDict
from compute_loss import accuracy, binary_segmentation_loss, instance_segmentation_loss
from keras.regularizers import l2

class LaneNet():
    """
    LaneNet Implementaion based on 
    Neven, Davy & Brabandere, Bert & Georgoulis, Stamatios & Proesmans, Marc & Van Gool, Luc. (2018). 
    Towards End-to-End Lane Detection: an Instance Segmentation Approach. 286-291. 10.1109/IVS.2018.8500547. 
    """

    def __init__(self, training, params):
        self.training = training
        self.params = params
        self.branches = ["binary", "instance"]
        self.encoder_states = OrderedDict()
        self.intializer = tf.keras.initializers.he_normal()

    # Shared Encoder for lanenet based on VGG16 architecture
    def _sharedEncoder(self, input_tensor):
        # Define the encoder model
        x = layers.Conv2D(64, kernel_size=3, activation='relu', padding="same", name="conv1_1", 
            kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, kernel_size=3, activation='relu', padding="same", name="conv1_2", 
            kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(x)
        x = layers.BatchNormalization()(x)
        self.encoder_states["shared_1"] = x
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="pool1")(x)
        x = layers.Conv2D(128, kernel_size=3, activation='relu', padding="same", name="conv2_1", 
            kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, kernel_size=3, activation='relu', padding="same", name="conv2_2", 
            kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(x)
        x = layers.BatchNormalization()(x)
        self.encoder_states["shared_2"] = x
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="pool2")(x)
        x = layers.Conv2D(256, kernel_size=3, activation='relu', padding="same", name="conv3_1", 
            kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, kernel_size=3, activation='relu', padding="same", name="conv3_2", 
            kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, kernel_size=3, activation='relu', padding="same", name="conv3_3", 
            kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(x)
        x = layers.BatchNormalization()(x)
        self.encoder_states["shared_3"] = x
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="pool3")(x)
        x = layers.Conv2D(512, kernel_size=3, activation='relu', padding="same", name="conv4_1", 
            kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(512, kernel_size=3, activation='relu', padding="same", name="conv4_2", 
            kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(512, kernel_size=3, activation='relu', padding="same", name="conv4_3", 
            kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(x)
        x = layers.BatchNormalization()(x)
        self.encoder_states["shared_4"] = x
        shr_encoder_output = layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="pool4")(x)
        

        return shr_encoder_output

    # Seperate encoder stage for Binary and Instance segmentation
    def _encoderStageBranch(self, branch, shr_encoder_output):
        conv_1 = layers.Conv2D(512, kernel_size=3, activation='relu', padding="same", 
            name="conv_"+branch+"_1", kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(shr_encoder_output)
        bn_1 = layers.BatchNormalization()(conv_1)
        conv_2 = layers.Conv2D(512, kernel_size=3, activation='relu', padding="same", 
            name="conv_"+branch+"_2", kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(bn_1)
        bn_2 = layers.BatchNormalization()(conv_2)
        conv_3 = layers.Conv2D(512, kernel_size=3, activation='relu', padding="same", 
            name="conv_"+branch+"_3", kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(bn_2)
        output = layers.BatchNormalization()(conv_3)
        # output = layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="pool_"+branch)(bn_3)
        self.encoder_states["encoder_"+branch] = output

        return output

    # VGG16 based encoder with shared stages and branches for Binary and Instance segmentation in the final stage
    def _vgg16_encoder(self, encoder_input):
        shared_encoder = self._sharedEncoder(encoder_input)
        binary_encoder = self._encoderStageBranch("binary", shared_encoder)
        instance_encoder = self._encoderStageBranch("instance", shared_encoder)
        return [binary_encoder, instance_encoder]

    # VGG16 based decoder with seperate decoding branch for Binary and Instance segmentation
    def _vgg16_decode(self, encoder_output):
        decoder_output = []
        # Define the decoder model
        for idx, branch_name in enumerate(self.branches):
            x = layers.Conv2DTranspose(512, kernel_size=3, activation='relu', padding="same",
                name="deconv_"+branch_name+"_1_1", kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(encoder_output[idx])
            x = layers.BatchNormalization()(x)
            x = layers.Conv2DTranspose(512, kernel_size=3, activation='relu', padding="same",
                name="deconv_"+branch_name+"_1_2", kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(x)
            x = layers.BatchNormalization()(x)
            x = layers.UpSampling2D(size=(2, 2), name="upsample_"+branch_name+"_1")(x)
            y = layers.Conv2D(512, kernel_size=3, activation='relu', padding="same",
                    kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(self.encoder_states["shared_4"])
            x = layers.Add()([x, y])
            x = layers.Conv2DTranspose(256, kernel_size=3, activation='relu', padding="same",
                name="deconv_"+branch_name+"_2_1", kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2DTranspose(256, kernel_size=3, activation='relu', padding="same",
                name="deconv_"+branch_name+"_2_2", kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(x)
            x = layers.BatchNormalization()(x)
            x = layers.UpSampling2D(size=(2, 2), name="upsample_"+branch_name+"_2")(x)
            y = layers.Conv2D(256, kernel_size=3, activation='relu', padding="same",
                    kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(self.encoder_states["shared_3"])
            x = layers.Add()([x, y])
            x = layers.Conv2DTranspose(128, kernel_size=3, activation='relu', padding="same",
                name="deconv_"+branch_name+"_3_1", kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2DTranspose(128, kernel_size=3, activation='relu', padding="same",
                name="deconv_"+branch_name+"_3_2", kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(x)
            x = layers.BatchNormalization()(x)
            x = layers.UpSampling2D(size=(2, 2), name="upsample_"+branch_name+"_3")(x)
            y = layers.Conv2D(128, kernel_size=3, activation='relu', padding="same",
                    kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(self.encoder_states["shared_2"])
            x = layers.Add()([x, y])
            x = layers.Conv2DTranspose(64, kernel_size=3, activation='relu', padding="same",
                name="deconv_"+branch_name+"_4_1", kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(x)
            x = layers.BatchNormalization()(x)
            x = layers.UpSampling2D(size=(2, 2), name="upsample_"+branch_name+"_4")(x)
            y = layers.Conv2D(64, kernel_size=3, activation='relu', padding="same",
                    kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(self.encoder_states["shared_1"])
            x = layers.Add()([x, y])
            # x = layers.Conv2DTranspose(64, kernel_size=3, activation='relu', padding="same",
            #     name="deconv_"+branch_name+"_5", kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(x)
            # x = layers.BatchNormalization()(x)

            if branch_name == "binary":
                output = layers.Conv2D(2, kernel_size=3, activation='relu', padding="same",
                    name="final_"+branch_name, kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(x)
            elif branch_name == "instance":
                output = layers.Conv2DTranspose(64, kernel_size=3, activation='relu', padding="same",
                    name="final_"+branch_name, kernel_regularizer=l2(0.0001), kernel_initializer=self.intializer)(x)
            decoder_output.append(output)

        return decoder_output
    

    def _build_model(self, input_shape):
        # Build the model
        batch_size = self.params.batch_size if self.training else self.params.test.batch_size
        autoencoder_input = tf.keras.Input(shape=input_shape, batch_size=batch_size, 
            name="autoencoder_input")
        encoded = self._vgg16_encoder(autoencoder_input)
        decoded = self._vgg16_decode(encoded)


        binary_segmentation = layers.Softmax(name="binary_segmentation", axis=-1)(decoded[0])
        # binary_segmentation = ArgMax(input_dim=(32, 256, 512, 2), name="binary_segmentation")(binary_softmax)
        # binary_segmentation = layers.Lambda(lambda inputs: 
        #     tf.map_fn(lambda x: tf.cast(tf.argmax(x, axis=-1), tf.float32), inputs), 
        #     name="binary_segmentation")(binary_softmax)
        
        instance_bn = layers.BatchNormalization()(decoded[1])
        instance_relu = layers.ReLU()(instance_bn)
        instance_embedding = layers.Conv2D(4, kernel_size=1, padding="same",
            name="instance_segmentation", kernel_initializer=self.intializer)(instance_relu)

        model = tf.keras.Model(inputs=autoencoder_input, outputs=[binary_segmentation, instance_embedding], 
            name="vgg16_autoencoder")
        

        # Decay learning rate
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.params.learning_rate, self.params.decay_steps, 
        #     self.params.decay_rate, staircase=True)
        # model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9),
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.05, decay=0.01, momentum=0.9),
            loss={"binary_segmentation": binary_segmentation_loss, "instance_segmentation": instance_segmentation_loss},
            loss_weights={"binary_segmentation": 1, "instance_segmentation": 1},
            metrics={"binary_segmentation": accuracy})

        return model