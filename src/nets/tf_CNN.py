import tensorflow as tf
import numpy as np

def tf_CNN(input_tensor, num_classes, is_training):
    conv1 = tf.layers.conv2d(inputs=input_tensor, filters=32, kernel_size=[5, 5], 
                             padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], 
                             padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)    
    pool2_flat = tf.reshape(pool2, [-1, pool2.shape[1]*pool2.shape[2]*pool2.shape[3]])

    dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense1, rate=0.5, training=is_training)

    dense2 = tf.layers.dense(inputs=dropout, units=256, activation=tf.nn.relu)

    logits = tf.layers.dense(inputs=dense2, units=num_classes)
    return logits