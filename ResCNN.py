# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 11:07:40 2018
此模型使用的卷积核大小分别是1,2,3
@author: Administrator
"""
import tensorflow as tf
#tf.reset_default_graph()
class ResCNN(object):
    def __init__(self,inputs,batch_size):
        self.inputs=inputs
        self.batch_size=batch_size
#        第一层卷积所需要的一些参数
        self.CONV1=2
        self.NUM_CHANNELS=1
        self.CONV1_DEEP=8
        
        self.CONV2=3
        self.CONV2_DEEP=8
        
        self.CONV3=2
        self.CONV3_DEEP=8
    def CNN_layer(self):
        with tf.variable_scope('layer_one_1',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [self.CONV1,self.CONV1,self.NUM_CHANNELS,self.CONV1_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[self.CONV1_DEEP],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(self.inputs, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer1=tf.nn.relu(tf.nn.bias_add(conv_one, bias_one))
        with tf.variable_scope('layer_one_2',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [self.CONV2,self.CONV2,self.CONV1_DEEP,self.CONV2_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[self.CONV2_DEEP],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer1, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer2=tf.nn.relu(tf.nn.bias_add(conv_one, bias_one))
        with tf.variable_scope('layer_one_3',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [self.CONV3,self.CONV3,self.CONV2_DEEP,self.CONV3_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[self.CONV3_DEEP],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer2, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer3=tf.nn.bias_add(conv_one, bias_one)
        with tf.variable_scope('layer_add',reuse=tf.AUTO_REUSE):
            weight=tf.get_variable("weight",
                                       [self.CONV1,self.CONV1,self.NUM_CHANNELS,8],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable("bias",[8],
                                  initializer=tf.constant_initializer(0))
            conv=tf.nn.conv2d(self.inputs, weight, strides=[1, 1, 1, 1], 
                                      padding='SAME')
#            return tf.nn.relu(tf.nn.bias_add(conv, bias)+layer3)
            layer_add=tf.nn.bias_add(conv, bias)+layer3
            print("block 1 shape :"+str(layer_add))
#            return layer_add
          

        with tf.variable_scope('layer_one_4',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [2,2,8,16],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[16],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer_add, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer4=tf.nn.relu(tf.nn.bias_add(conv_one, bias_one))
        with tf.variable_scope('layer_one_5',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [3,3,16,16],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[16],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer4, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer5=tf.nn.relu(tf.nn.bias_add(conv_one, bias_one))
        with tf.variable_scope('layer_one_6',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [2,2,16,16],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[16],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer5, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer6=tf.nn.bias_add(conv_one, bias_one)
        with tf.variable_scope('layer_add1',reuse=tf.AUTO_REUSE):
            weight=tf.get_variable("weight",
                                       [1,1,8,16],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable("bias",[16],
                                  initializer=tf.constant_initializer(0))
            conv=tf.nn.conv2d(layer_add, weight, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer_add=tf.nn.relu(tf.nn.bias_add(conv, bias)+layer6)
            print("block 2 shape :"+str(layer_add))
#            return layer_add

#       降维

#        with tf.variable_scope('dimension',reuse=tf.AUTO_REUSE):
#            weight=tf.get_variable("weight",
#                                       [2,2,16,16],
#                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
#            bias=tf.get_variable("bias",[16],
#                                  initializer=tf.constant_initializer(0))
#            layer_add=tf.nn.conv2d(layer_add, weight, strides=[1, 2, 1, 1], 
#                                      padding='SAME')
#            print('dimention is :'+str(layer_add.shape))
            
        with tf.variable_scope('layer_one_7',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [2,2,16,32],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[32],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer_add, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer7=tf.nn.relu(tf.nn.bias_add(conv_one, bias_one))
        with tf.variable_scope('layer_one_8',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [3,3,32,32],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[32],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer7, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer8=tf.nn.relu(tf.nn.bias_add(conv_one, bias_one))
        with tf.variable_scope('layer_one_9',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [2,2,32,32],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[32],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer8, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer9=tf.nn.bias_add(conv_one, bias_one)
        with tf.variable_scope('layer_add2',reuse=tf.AUTO_REUSE):
            weight=tf.get_variable("weight",
                                       [1,1,16,32],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable("bias",[32],
                                  initializer=tf.constant_initializer(0))
            conv=tf.nn.conv2d(layer_add, weight, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer_add=tf.nn.relu(tf.nn.bias_add(conv, bias)+layer9)
            print("block 3 shape :"+str(layer_add))
#            return layer_add


        with tf.variable_scope('layer_one_10',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [2,2,32,32],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[32],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer_add, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer10=tf.nn.relu(tf.nn.bias_add(conv_one, bias_one))
        with tf.variable_scope('layer_one_11',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [3,3,32,32],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[32],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer10, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer11=tf.nn.relu(tf.nn.bias_add(conv_one, bias_one))
        with tf.variable_scope('layer_one_12',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [2,2,32,32],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[32],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer11, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer12=tf.nn.bias_add(conv_one, bias_one)
        with tf.variable_scope('layer_add3',reuse=tf.AUTO_REUSE):
            if layer12.shape[3]!=layer_add.shape[3]:
                print('yes')
                weight=tf.get_variable("weight",
                                           [1,1,32,32],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
                bias=tf.get_variable("bias",[32],
                                      initializer=tf.constant_initializer(0))
                conv=tf.nn.conv2d(layer_add, weight, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer_add=tf.nn.relu(tf.nn.bias_add(conv, bias)+layer12)
            return layer_add

'''           
        with tf.variable_scope('layer_one_13',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [2,2,32,32],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[32],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer_add, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer13=tf.nn.relu(tf.nn.bias_add(conv_one, bias_one))
        with tf.variable_scope('layer_one_14',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [3,3,32,32],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[32],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer13, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer14=tf.nn.relu(tf.nn.bias_add(conv_one, bias_one))
        with tf.variable_scope('layer_one_15',reuse=tf.AUTO_REUSE):
            weight_one=tf.get_variable("weight",
                                       [2,2,32,32],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias_one=tf.get_variable("bias",[32],
                                  initializer=tf.constant_initializer(0))
            conv_one=tf.nn.conv2d(layer14, weight_one, strides=[1, 1, 1, 1], 
                                      padding='SAME')
            layer15=tf.nn.bias_add(conv_one, bias_one)
        with tf.variable_scope('layer_add4',reuse=tf.AUTO_REUSE):
            if layer15.shape[3]!=layer_add.shape[3]:
                weight=tf.get_variable("weight",
                                           [1,1,32,32],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
                bias=tf.get_variable("bias",[32],
                                      initializer=tf.constant_initializer(0))
                conv=tf.nn.conv2d(layer_add, weight, strides=[1, 1, 1, 1], 
                                          padding='SAME')
            layer_add=tf.nn.relu(tf.nn.bias_add(conv, bias)+layer15)
            return layer_add    
'''