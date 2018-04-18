################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
#from pylab import *
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random


import tensorflow as tf

from caffe_classes import class_names

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]



################################################################################
#Read Image, and change to BGR

folders = [
  "vertical",
  "diagonal", 
  "diagonalLine", 
  "gap2", 
  "gapLine", 
  "gapLineMiddle",
  "gapLineMiddleHalves", 
  "gapLineMiddleHalves2", 
  "gapMiddleHalves", 
  "gapMiddleHalves2", 
  "gapLineMiddleHalvesFlipped", 
  "gapLineMiddleHalvesFlipped2", 
  "gapMiddleHalvesFlipped", 
  "gapMiddleHalvesFlipped2", 
  "gapLineHalves", 
  "gapLineHalves2"]
  
numImgsArr = [
  225,
  85,
  85,
  121,
  121,
  121,
  121,
  121,
  121,
  121,
  121,
  121,
  121,
  121,
  121,
  121]

################################################################################

# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))

#In Python 3.5, change this to:
net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
#net_data = load("bvlc_alexnet.npy").item()

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])



x = tf.placeholder(tf.float32, (None,) + xdim)


#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0])
conv1b = tf.Variable(net_data["conv1"][1])
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)
print(conv1.get_shape())

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
print(lrn1.get_shape())

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
print(maxpool1.get_shape())


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)
print(conv2.get_shape())


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
print(lrn2.get_shape())

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
print(maxpool2.get_shape())

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)
print(conv3.get_shape())

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)
print(conv4.get_shape())

#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5W = tf.Variable(net_data["conv5"][0])
conv5b = tf.Variable(net_data["conv5"][1])
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)
print(conv5.get_shape())

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
print(maxpool5.get_shape())

#fc6
#fc(4096, name='fc6')
fc6W = tf.Variable(net_data["fc6"][0], trainable=False)
fc6b = tf.Variable(net_data["fc6"][1])
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
print(fc6.get_shape())

#fc7
#fc(4096, name='fc7')
fc7W = tf.Variable(net_data["fc7"][0])
fc7b = tf.Variable(net_data["fc7"][1])
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
print(fc7.get_shape())

#fc8
#fc(1000, relu=False, name='fc8')
fc8W = tf.Variable(net_data["fc8"][0])
fc8b = tf.Variable(net_data["fc8"][1])
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
print(fc8.get_shape())


#remove softmax
#prob
#softmax(name='prob'))
# prob = tf.nn.softmax(fc8)

# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)

# t = time.time()
# output = sess.run(prob, feed_dict = {x:imgs})
################################################################################


################################################################################

def sim(x,y):
  x = x.flatten()
  y = y.flatten()
  return np.dot(x,y)/(np.linalg.norm(x) * np.linalg.norm(y))

folderOut = "out"
for i in range(len(folders)):  
  folder = folders[i]
  numImgs = numImgsArr[i]
  print(folder)
  
  imgs = []

  mean = np.array([103.939, 116.779, 123.68]) # BGR

  control = (imread(folder + "/control.png")[:,:,:3]).astype(float32)
  #control = control - mean(control)
  control[:, :, 0], control[:, :, 2] = control[:, :, 2], control[:, :, 0]
  control = control - mean[None,None,:]
  imgs.append(control)


  for i in range(numImgs):
    img = (imread(folder + "/" + str(i) + ".png")[:,:,:3]).astype(float32)
    # img = img - mean(img)
    img[:, :, 0], img[:, :, 2] = img[:, :, 2], img[:, :, 0]
    img = img - mean[None,None,:]
    imgs.append(img)
    
  init = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)

  t = time.time()
  output8 = sess.run(fc8, feed_dict = {x:imgs})
  output7 = sess.run(fc7, feed_dict = {x:imgs})
  output6 = sess.run(fc6, feed_dict = {x:imgs})
  output5 = sess.run(maxpool5, feed_dict = {x:imgs})
  output4 = sess.run(conv4, feed_dict = {x:imgs})
  output3 = sess.run(conv3, feed_dict = {x:imgs})
  output2 = sess.run(maxpool2, feed_dict = {x:imgs})
  output1 = sess.run(maxpool1, feed_dict = {x:imgs})

  #Output:
  outputs = [output1, output2, output3, output4, output5,
    output6, output7, output8]
  titles = np.array(["maxpool1","maxpool2","conv3","conv4","maxpool5","fc6","fc7","fc8"])
  # shape[0] should be consistent
  # for i in range(output8.shape[0]):
      # if i == 0:
        # # this is the control; skip it
        # continue
      
      # # print(str(i) + "\t" + str(sim(control, output[i])) + "\t" + str(np.corrcoef(control, output[i])[0][1]) + "\t" + str(sim(control - np.mean(control), output[i] - np.mean(output[i]))));
      # print(str(i) + "\t" + 
        # str(sim(control[0], output1[i])) + "\t" + 
        # str(sim(control[1], output2[i])) + "\t" + 
        # str(sim(control[2], output3[i])) + "\t" + 
        # str(sim(control[3], output4[i])) + "\t" + 
        # str(sim(control[4], output5[i])) + "\t" + 
        # str(sim(control[5], output6[i])) + "\t" + 
        # str(sim(control[6], output7[i])) + "\t" + 
        # str(sim(control[7], output8[i])) + "\t");
      
  # # print(control)
  # print(str(np.mean(control[0])) + " " +
        # str(np.mean(control[1])) + " " +
        # str(np.mean(control[2])) + " " +
        # str(np.mean(control[3])) + " " +
        # str(np.mean(control[4])) + " " +
        # str(np.mean(control[5])) + " " +
        # str(np.mean(control[6])) + " " +
        # str(np.mean(control[7])))
  
  fileOut = os.path.join(folderOut, folder + '.csv')
  with open(fileOut, 'w+') as out:
    for i in range(len(outputs)):
      out.write(titles[i] + ",")
      for j in range(1,output8.shape[0]):
         out.write(str(sim(outputs[i][0], outputs[i][j])) + ",")
      out.write("\n")
      
      # output = simV(outputs[i][0], outputs[i][1:])
    # out.write("maxpool," + ",".join(output1) + "\n")
    # out.write(",".join)
    # out.write("\n")
    # for i in range(output8.shape[0]):
      # if i == 0:
        # # this is the control; skip it
        # continue
      
      # # print(str(i) + "\t" + str(sim(control, output[i])) + "\t" + str(np.corrcoef(control, output[i])[0][1]) + "\t" + str(sim(control - np.mean(control), output[i] - np.mean(output[i]))));
      # out.write(str(i) + "," + 
        # str(sim(control[0], output1[i])) + "," + 
        # str(sim(control[1], output2[i])) + "," + 
        # str(sim(control[2], output3[i])) + "," + 
        # str(sim(control[3], output4[i])) + "," + 
        # str(sim(control[4], output5[i])) + "," + 
        # str(sim(control[5], output6[i])) + "," + 
        # str(sim(control[6], output7[i])) + "," + 
        # str(sim(control[7], output8[i])) + ",");
      
  print(time.time()-t)
