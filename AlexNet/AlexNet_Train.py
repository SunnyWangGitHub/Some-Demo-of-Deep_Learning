#python3
# -*- coding: utf-8 -*-
# File  : AlexNet_Train.py
# Author: Wang Chao
# Date  : 2018/11/23

import os
import tensorflow as tf
import AlexNet_Inference
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# variable of cnn
BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.001
TRAINING_STEPS = 20000
MOVING_AVERAGE_DECAY = 0.99

# the path of save file and model
MODEL_SAVE_PATH = 'path_to_model'
MODEL_NAME = 'model.ckpt'


####用mnist 跑Alex net，书上用的数据是imagenet，准备用mnist来跑，需要reshap input tensor
def train(mnist):
    x = tf.placeholder(tf.float32,[None,784],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,10], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    y = AlexNet_Inference.inference(x,10)

    global_step = tf.Variable(0,trainable=False)

    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step= global_step)
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:xs, y_: ys})

            if i%1000 == 0:
                print('after %d training step(s),loss on training batch is %g.' % (step, loss_value))
                saver.save(
                    sess,
                    os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                    global_step= global_step
                )


def main(argv=None):
    mnist = input_data.read_data_sets('MNIST_data', one_hot= True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()

