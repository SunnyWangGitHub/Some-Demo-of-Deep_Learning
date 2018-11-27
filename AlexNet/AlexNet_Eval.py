#python3
# -*- coding: utf-8 -*-
# File  : MNIST_eval.py
# Author: Wang Chao
# Date  : 2018/11/20

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import AlexNet_Inference
import AlexNet_Train

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,[None,784],name = 'x-input')
        y_ = tf.placeholder(tf.float32,[None,10],name = 'y-input')
        validate_feed = {x:mnist.validation.images,
                         y_:mnist.validation.labels}

        y = AlexNet_Inference.inference(x,10)

        correct_prediction = tf.equal(tf.arg_max(y,1),tf.argmax(y_,1))
        accuarcy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        ###############???
        variable_average = tf.train.ExponentialMovingAverage(AlexNet_Train.MOVING_AVERAGE_DECAY)
        variables_to_store = variable_average.variables_to_restore()
        saver = tf.train.Saver(variables_to_store)
        ##################

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(AlexNet_Train.MODEL_SAVE_PATH)
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