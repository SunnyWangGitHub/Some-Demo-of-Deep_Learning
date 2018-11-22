#python3
# -*- coding: utf-8 -*-
# File  : LeNet-5_Inference.py
# Author: Wang Chao
# Date  : 2018/11/21

import tensorflow as tf


#variables of cnn
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

#第一层卷积层的尺寸和深度
CONV1_DEEP =32
CONV1_SIZE = 5
#第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5
#全连接的节点个数
FC_SIZE = 512

#train 用来区分训练和测试
def inference(input_tensor,train,regularizer):
    #####第一层卷积
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight",
            [CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
            initializer = tf.truncated_normal_initializer(stddev=0.1)
        )
        conv1_biases = tf.get_variable(
            "bias",
            [CONV1_DEEP],
            initializer = tf.constant_initializer(0.0)
        )

        #使用边长为5，深度为32的filter，移动步长为1，且使用全0填充
        conv1 = tf.nn.conv2d(
            input_tensor,
            conv1_weights,
            strides = [1,1,1,1],
            padding = 'SAME'
        )
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    #######第二层池化,输入是上一层的输出28x28x32,输出为14x14x32
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(
            relu1,
            ksize = [1,2,2,1],
            strides = [1,2,2,1],
            padding = 'SAME'
        )

    ########第三层卷积，输入为14x14x32,输出为14x14x64
    with tf.variable_scope('layer3-conv2'):
        conv2_weight = tf.get_variable(
            "weight",
            [CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
            initializer = tf.truncated_normal_initializer(stddev=0.1)
        )
        conv2_biases = tf.get_variable(
            "bias",
            [CONV2_DEEP],
            initializer = tf.constant_initializer(0.0)
        )
        #filter 边长5，深度64，移动步长为1，全0填充
        conv2 = tf.nn.conv2d(
            pool1,
            conv2_weight,
            strides = [1,1,1,1],
            padding='SAME'
        )
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    #######第四层池化,输入14x14x64,输出7x7x64
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(
            relu2,
            ksize = [1,2,2,1],
            strides = [1,2,2,1],
            padding = 'SAME'
        )

    ############为全连接准备,将7x7x64拉直为一个向量输入全连接
    pool_shape = pool2.get_shape().as_list()
    # print(pool_shape)
    #pool_shape[0]为一个batch中的数据个数
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]

    #通过tf.reshape将第四层输出变成一个Batch的向量
    reshaped = tf.reshape(pool2,[pool_shape[0],nodes])

    ########第五层全连接，输入为一组长度为3136的向量，输出为一组长度为512的向量，引入dropout
    with tf.variable_scope('layer5-fc1'):
        fc1_weight = tf.get_variable(
            "weight",
            [nodes,FC_SIZE],
            initializer = tf.truncated_normal_initializer(stddev=0.1)
        )
        ####只有全连接的权重需要加入正则化？？？
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc1_weight))
        fc1_biases = tf.get_variable(
            "bias",
            [FC_SIZE],
            initializer = tf.constant_initializer(0.1)
        )
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weight)+fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1,0.5)

    #########第六层全连接，输入为一组512的向量，输出为一组10的向量。这一层的输出通过softMax 就可以得到最后的分类结果
    with tf.variable_scope('layer6-fc2'):
        fc2_weight = tf.get_variable(
            "weight",
            [FC_SIZE,NUM_LABELS],
            initializer = tf.truncated_normal_initializer(stddev = 0.1)
        )
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc2_weight))
        fc2_biases = tf.get_variable(
            'bias',
            [NUM_LABELS],
            initializer = tf.constant_initializer(0.1)
        )
        logit = tf.matmul(fc1,fc2_weight)+fc2_biases
    return logit


