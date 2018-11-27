#python3
# -*- coding: utf-8 -*-
# File  : AlexNet_Inference.py
# Author: Wang Chao
# Date  : 2018/11/22

import tensorflow as tf

#define the structure of AlexNet
def inference(images,n_classes):
    images = tf.reshape(images, [-1, 28, 28, 1])
    ###########卷积层conv1
    #原weight shape = [11,11,3,96], biases shape = [96],
    with tf.variable_scope('conv1_1rn') as scope:
        weights = tf.get_variable(
            'weights',
            shape = [11,11,1,96],
            dtype = tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32)
        )
        biases = tf.get_variable(
            'biases',
            shape = [96],
            dtype = tf.float32,
            initializer=tf.constant_initializer(0.1)
        )
        # 原strides=[1,4,4,1]
        conv = tf.nn.conv2d(images,weights,strides = [1,1,1,1],padding = 'SAME')
        pre_activation = tf.nn.bias_add(conv,biases)
        #使用ReLU激活函数
        conv1 = tf.nn.relu(pre_activation,name = scope.name)
        #conv1的局部响应归一化
        norm1 = tf.nn.lrn(conv1,depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')

    ############池化层pool1：最大池化
    # 原ksize=[1,3,3,1],strides=[1,2,2,1]
    with tf.variable_scope('pooling1') as scope:
        pool1 = tf.nn.max_pool(norm1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pooling1')

    ##############卷积层2 conv2
    with  tf.variable_scope('conv2_lrn') as scope: ####zhe li you xiu gai
        weights = tf.get_variable(
            'weights',
            shape = [5,5,96,256],
            dtype = tf.float32,
            initializer = tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32)
        )
        biases = tf.get_variable(
            'biases',
            shape = [256],
            dtype = tf.float32,
            initializer = tf.constant_initializer(0.1)
        )
        conv = tf.nn.conv2d(pool1, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(pre_activation,name ='conv2')
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')

    ################池化层pool2 最大池化
    # 原ksize=[1,3,3,1],strides=[1,2,2,1]
    with tf.variable_scope('pooling2') as scope:
        pool2 = tf.nn.max_pool(norm2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pooling2')

    ##################### 卷积层conv3
    with tf.variable_scope('conv3') as scope:
        weights = tf.get_variable(
            'weights',
            shape = [3,3,256,384],
            dtype = tf.float32,
            initializer = tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32)
        )
        biases = tf.get_variable(
            'biases',
            shape = [384],
            dtype = tf.float32,
            initializer = tf.constant_initializer(0.1)
        )
        conv = tf.nn.conv2d(pool2,weights,strides=[1,1,1,1], padding='SAME') ###############?????????????
        pre_activation = tf.nn.bias_add(conv,biases)
        conv3 = tf.nn.relu(pre_activation, name='conv3')

    ######################卷积层conv4
    with tf.variable_scope('conv4') as scope:
        weights = tf.get_variable(
            'weights',
            shape = [3,3,384,384],
            dtype = tf.float32,
            initializer = tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32)
        )
        biases = tf.get_variable(
            'biases',
            shape = [384],
            dtype = tf.float32,
            initializer = tf.constant_initializer(0.1)
        )
        conv = tf.nn.conv2d(conv3, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv,biases)
        conv4 = tf.nn.relu(pre_activation,name='conv4')

    #####################卷积层conv5
    with tf.variable_scope('conv5') as scope:
        weights = tf.get_variable(
            'weights',
            shape=[3, 3, 384, 256],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32)
        )
        biases = tf.get_variable(
            'biases',
            shape=[256],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.1)
        )
        conv = tf.nn.conv2d(conv4, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(pre_activation, name='conv5')

    ###########################池化层6
    # 原ksize=[1,3,3,1],strides=[1,2,2,1]
    with tf.variable_scope('pooling6') as scope:
        pool6 = tf.nn.max_pool(
            conv5,
            ksize= [1,2,2,1],
            strides=[1,2,2,1],
            padding='SAME',
            name= 'pooling6'
        )

    ############################全连接层7
    with tf.variable_scope('local7') as scope:
        reshape = tf.reshape(pool6,shape=[-1, 4*4*256])
        # dim = reshape.get_shape()[1].value
        weights = tf.get_variable(
            'weights',
            shape = [4*4*256, 4096],
            dtype = tf.float32,
            initializer = tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32)
        )
        biases = tf.get_variable(
            'biases',
            shape=[4096],
            dtype=tf.float32,
            initializer = tf.constant_initializer(0.1)
        )
        local7 = tf.nn.relu(tf.matmul(reshape,weights)+biases,name=scope.name)
        local7 =tf.nn.dropout(local7,keep_prob=0.5)

    ###########################全连接层8
    with tf.variable_scope('local8') as scope:
        weights = tf.get_variable(
            'weights',
            shape=[4096,4096],
            dtype = tf.float32,
            initializer = tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32)
        )
        biases = tf.get_variable(
            'biases',
            shape=[4096],
            dtype = tf.float32,
            initializer = tf.constant_initializer(0.1)
        )
        local8 = tf.nn.relu(tf.matmul(local7,weights)+biases,name=scope.name)
        local8 = tf.nn.dropout(local8,keep_prob=0.5)

    ##########################softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable(
            'softmax_linear',
            shape= [4096,n_classes],
            dtype = tf.float32,
            initializer = tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32)
        )
        biases = tf.get_variable(
            'biases',
            shape = [n_classes],
            dtype = tf.float32,
            initializer = tf.constant_initializer(0.1)
        )
        softmax_linear = tf.add(tf.matmul(local8,weights),biases,name='sotfmax_linear')

    return softmax_linear




