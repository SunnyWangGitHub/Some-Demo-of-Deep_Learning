#python3
# -*- coding: utf-8 -*-
# File  : MNIST_Train.py
# Author: Wang Chao
# Date  : 2018/11/20

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#load variable and forward function which define in MNIST_Inferece.py
import MNIST_Inference

#variable of cnn
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

#the path of save file and model
MODEL_SAVE_PATH = 'path_to_model'
MODEL_NAME = 'model.ckpt'

def train(mnist):
    x = tf.placeholder(tf.float32,[None,MNIST_Inference.INPUT_NODE],name = 'x-input')
    y_ = tf.placeholder(tf.float32,[None,MNIST_Inference.OUTPUT_NODE],name = 'y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    #use forward in MNIST_Inference
    y = MNIST_Inference.inference(x,regularizer)
    global_step = tf.Variable(0,trainable = False)

    #define loss,learning_rate,Moving average.. and train
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variables_averages_op = variable_average.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y,labels = tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)
    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op = tf.no_op(name='train')

    #init tensorflow 持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        for i in range(TRAINING_STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})

            #save model each 1000 echo
            if i%1000 ==0:
                print('after %d training step(s),loss on training batch is %g.'%(step,loss_value))
                saver.save(
                    sess,
                    os.path.join(MODEL_SAVE_PATH,MODEL_NAME),
                    global_step = global_step
                )

def main(argv = None):
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()














