#python3
# -*- coding: utf-8 -*-
# File  : LeNet_eval.py
# Author: Wang Chao
# Date  : 2018/11/21

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import LeNet_Inference
import LeNet_Train
import numpy as np

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,
                           [mnist.test.num_examples,
                            LeNet_Inference.IMAGE_SIZE,
                            LeNet_Inference.IMAGE_SIZE,
                            LeNet_Inference.NUM_CHANNELS
                            ],
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [None, LeNet_Inference.OUTPUT_NODE], name='y-input')
        reshaped_xs = np.reshape(mnist.test.images, (mnist.test.num_examples, LeNet_Inference.IMAGE_SIZE, LeNet_Inference.IMAGE_SIZE, LeNet_Inference.NUM_CHANNELS))
        validate_feed = {x:reshaped_xs,
                         y_:mnist.test.labels}

        y = LeNet_Inference.inference(x,False,None)

        correct_prediction = tf.equal(tf.arg_max(y,1),tf.argmax(y_,1))
        accuarcy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        ###############???
        variable_average = tf.train.ExponentialMovingAverage(LeNet_Train.MOVING_AVERAGE_DECAY)
        variables_to_store = variable_average.variables_to_restore()
        saver = tf.train.Saver(variables_to_store)
        ##################

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(LeNet_Train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('-')[-1]
                    accuarcy_score = sess.run(accuarcy,feed_dict=validate_feed)
                    print("after %s training step(s),validation accuracy = %g"%(global_step,accuarcy_score))
                else:
                    print('no checkpoint file found')
                    return
                time.sleep(EVAL_INTERVAL_SECS)
def main(argv=None):
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
    evaluate(mnist)

if __name__ =='__main__':
    tf.app.run()