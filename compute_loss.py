import tensorflow as tf
from discriminative_loss import discriminative_loss

def binary_segmentation_loss(y_true, y_pred):
    """
    Binary segmentation loss
    """
    
    y_pred = tf.map_fn(lambda x: tf.cast(tf.argmax(x, axis=-1), tf.float32), y_pred)

    tf.cast(y_true, tf.float32)
    tf.cast(y_pred, tf.float32)
    
    y_pred = tf.expand_dims(y_pred, axis=-1)
    y_true = tf.expand_dims(y_true, axis=-1)
    
    _, _, count = tf.unique_with_counts(
            tf.reshape(y_true, shape=[y_true.shape[0] * y_true.shape[1] * y_true.shape[2]]))

    inverse_weights = tf.divide(1.0, 
        tf.math.log(tf.add(tf.divide(tf.constant(1.0), tf.cast(count, tf.float32)), 
            tf.constant(1.02))))
    inverse_weights = tf.gather(inverse_weights, tf.cast(y_true, tf.int32))

    ce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return tf.reduce_mean(ce_loss(y_true, y_pred, sample_weight=inverse_weights))


def instance_segmentation_loss(y_true, y_pred):
    """
    Instance segmentation loss
    """

    tf.cast(y_true, tf.float32)
    tf.cast(y_pred, tf.float32)

    loss, _, _, _ = discriminative_loss(y_pred, y_true, 4, y_true.shape[1:], 0.5, 3.0, 1.0, 1.0, 0.001)

    return loss

def accuracy(y_true, y_pred):
    """
    Accuracy metric
    """

    y_pred = tf.map_fn(lambda x: tf.cast(tf.argmax(x, axis=-1), tf.float32), y_pred)

    y_pred = tf.expand_dims(y_pred, axis=-1)
    idx = tf.where(tf.equal(y_true, 1))
    pixels = tf.gather_nd(y_pred, idx)
    
    accuracy = tf.math.count_nonzero(pixels)

    return tf.math.divide(accuracy, tf.cast(tf.shape(pixels)[0], tf.int64))
