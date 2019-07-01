# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:19:40 2019

@author: zhang
"""
import os
import numpy as np
import tensorflow as tf
import time
max_epoch=350
model_save_path = 'your own path'
model_name = 'baby.ckpt'
log_dir = 'your own path'

train_x = np.load("your own path")
train_y = np.load("your own path")
index=np.arange(1108)
np.random.shuffle(index)
train_x = train_x[index]
train_y = train_y[index]
##=============================================================================
   ## definition of fundamental operation ##
##=============================================================================
def weight_variable(shape, name):
    initializer = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initializer, name = name)

def bias_variable(shape, name):
    initializer = tf.constant(0.1, shape=shape)
    return tf.Variable(initializer, name = name)

def conv2d(x, W):
    
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, name):
    
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name = name)

##=============================================================================
   ## input placeholder ##
##=============================================================================
xs = tf.placeholder(tf.float32, [1, 128, 128,3], name = 'input')   # 128x128
ys = tf.placeholder(tf.float32, [1, 2])

#xs_v = tf.placeholder(tf.float32, [None, 128, 128,3])   # 128x128
#ys_v = tf.placeholder(tf.float32, [None, 2])
#xs_input = tf.placeholdertf(tf.float32, [1, 128, 128, 3], name = 'input')
##=============================================================================
   ## network architecture ##
##=============================================================================

## conv1 layer 
def inference(x):
    
    W_conv1 = weight_variable([3, 3, 3, 16], 'W_conv1') # patch 5x5, in size 3, out size 16
    b_conv1 = bias_variable([16], 'b_conv1')
    h_conv1 = tf.nn.relu((conv2d(x, W_conv1) + b_conv1), name = 'conv1') # output size 128x128x16
    h_pool1 = max_pool_2x2(h_conv1, 'pool1')                                         # output size 64x64x16
    
    ## conv2 layer 
    W_conv2 = weight_variable([3, 3, 16, 32], 'W_conv2') # patch 5x5, in size 16, out size 32
    b_conv2 = bias_variable([32], 'b_conv2')
    h_conv2 = tf.nn.relu((conv2d(h_pool1, W_conv2) + b_conv2), name = 'conv2') # output size 64x64x32
    h_pool2 = max_pool_2x2(h_conv2, name = 'pool2')                                         # output size 32x32x32
    
    ## flatten
    h_pool_flat = tf.reshape(h_pool2, [-1, 32 * 32 * 32])
    
    ## fc1 layer
    W_fc1 = weight_variable([32 * 32 * 32, 64], 'W_fc1')
    b_fc1 = bias_variable([64], 'b_fc1')
    h_fc1 = tf.nn.relu((tf.matmul(h_pool_flat, W_fc1) + b_fc1), name = 'fc1')
    h_fc1 = tf.nn.dropout(h_fc1, keep_prob = 0.9, name = 'dropout')
    
    ## fc2 layer
    W_fc2 = weight_variable([64, 2], 'W_fc2')
    b_fc2 = bias_variable([2], 'b_fc2')
    prediction = tf.nn.bias_add(tf.matmul(h_fc1, W_fc2), b_fc2, name = 'output')  
    prediction = tf.nn.softmax(prediction, name = 'softmax_output')
    
    return prediction
##=============================================================================
   ## loss ##
##=============================================================================
prediction = inference(xs)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = ys, logits = prediction)
_loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(_loss)
##=============================================================================
   ## accuracy ##
##=============================================================================
bool_acc = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(ys, 1))
acc = tf.reduce_mean(tf.cast(bool_acc, tf.float32))
##=============================================================================
   ## main part ##
##=============================================================================
saver = tf.train.Saver()

with tf.Session() as sess:
    
    init = tf.global_variables_initializer()
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    sess.run(init)
    
    loss_vec = []
    acc_vec = []
    
    for i in range(max_epoch):
#        for m,n in zip(range(0,1108), range(1,1108)):
        start = time.clock()
        loss, accuracy, _ = sess.run([_loss, acc, train_step],feed_dict={xs: train_x, ys: train_y})
        
        loss_vec.append(loss)
#            acc_vec.append(accuracy)
#            acc = sess.run(acc, feed_dict = {xs_v:train_x, ys_v:train_y})
        time_cost = time.clock() - start
        print ('step %d, loss = %.4f, accuracy = %.4f, it costs %g' % (i + 1, loss, accuracy, time_cost),'s')
            
        if i + 1 == max_epoch:
            accc = sess.run(acc, feed_dict={xs: train_x[0:1], ys:train_y[0:1]})
            print(accc)
            saver.save(sess, os.path.join(model_save_path,model_name), global_step = i)
            