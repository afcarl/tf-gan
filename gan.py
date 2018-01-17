""" TensorFlow implementation of Generative Adversarial Networks """

import numpy as np
import tensorflow as tf
sess = tf.InteractiveSession()

from tensorflow.python.keras.datasets.mnist import load_data
(x_train, _), (x_test, _) = load_data()

# Squash the data to [0, 1]
x_train = x_train / 255.
x_test = x_test / 255.

with tf.name_scope("G"):
    with tf.name_scope("z"):
        z = tf.placeholder(tf.float32, [None, 32])
    with tf.name_scope("H_1"):
        W_1 = tf.Variable(tf.truncated_normal(shape=[32, 64], stddev=0.1), name="W_1")
        b_1 = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[64]), name="b_1")
        a_1 = tf.nn.relu(tf.matmul(tf.transpose(W_1), z) + b_1)
    with tf.name_scope("H_2"):
        W_2 = tf.Variable(tf.truncated_normal(shape=[64, 64], stddev=0.1), name="W_2")
        b_2 = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[64]), name="b_2")
        a_2 = tf.nn.relu(tf.matmul(tf.transpose(W_2), a_1) + b_2)
    with tf.name_scope("x"):
        W_3 = tf.Variable(tf.truncated_normal(shape=[64, 784], stddev=0.1), name="W_3")
        b_3 = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[64]), name="b_3")
        x = tf.matmul(tf.transpose(W_3), a_2) + b_3
    with tf.name_scope("Loss"):
        

    tf.summary.image("X", tf.reshape(X, [-1, 28, 28, 1]))



sess.close()
