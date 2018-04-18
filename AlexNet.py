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
import numpy as np
import time
import argparse
import sys

import tensorflow as tf
# from scipy.misc import imread
from scipy.ndimage import imread

from caffe_classes import class_names

# command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
# argument for lowest level of network to be trained
layers = ['fc8','fc7','fc6','all']
layer_default = 'fc8'
parser.add_argument('-l', nargs=1, metavar='layer', choices=layers, default=[layer_default], required=False, 
                    help='Lowest layer to be trained; the passed layer, and all\nfuture layers, will be trained.\nAllowed values are: '+', '.join(layers)+'\nDefault: '+layer_default)
# argument for learning rate
lr_default = 0.01
parser.add_argument('-lr', nargs=1, metavar='learning_rate', type=float,required=False, default=[lr_default], 
                    help='Learning rate for gradiant descent.\nDefault: '+str(lr_default))
# argument for final layer type
outputs = ['sigmoid','softmax']
output_default = 'sigmoid'
parser.add_argument('-fl', nargs=1, metavar='final_layer', choices=outputs, default=[output_default], required=False, 
                    help='Final layer used in the network.\nAllowed values are: '+', '.join(outputs)+'\nDefault: '+output_default)
# argument for whether to restore from a save
load_default = -1
parser.add_argument('-es', nargs=1, metavar='epoch_start', type=int, required=False, default=[load_default], 
                    help='If set to a non-negative value, loads the last\nline-model-final.ckpt and starts at the passed epoch.\nDefault: '+str(load_default))
# save file
save_default = 'a'
parser.add_argument('-sf', nargs=1, metavar='save_folder', type=str, required=False, default=[save_default],
                    help='Indirect path of saves folder; if it does not\nexist, it attempts to create the folder.\nIf "a" is used, will use a folder named based on arguments.\nDefault: '+save_default)
# max epochs
maxe_default = 10
parser.add_argument('-me', nargs=1, metavar='max_epochs', type=int, required=False, default=[maxe_default],
                    help='Max number of epochs that will be run\nDefault: '+str(maxe_default))
# max length of program, in minutes
runtime_default = 60
parser.add_argument('-r', nargs=1, metavar='runtime', type=int, required=False, default=[runtime_default],
                    help='Max time program will run, in minutes\nDefault: '+str(runtime_default))
# lines folder
lines_folder_default = 'single_6_075'
parser.add_argument('-lf', nargs=1, metavar='lines_folder', type=str, required=False, default=[lines_folder_default],
                    help='Folder within ../lines/ folder which contains the train, validation and test folder of images\nDefault: '+lines_folder_default)
                    
args = parser.parse_args()
print(args)

layer = args.l[0]
learning_rate = args.lr[0]
final_layer = args.fl[0]
restore = args.es[0] >= 0
epoch_start = args.es[0] if restore else 0
max_epochs = args.me[0]
max_program_length = args.r[0]
save_folder = args.sf[0]
lines_folder = args.lf[0]
if save_folder == "a":
  save_folder = "./saves/saves_" + "_".join([layer, str(learning_rate), final_layer, lines_folder])  
start_time = time.time()

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]

mean = np.array([103.939, 116.779, 123.68]) # BGR

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
net_data = load(open("../bvlc_alexnet.npy", "rb"), encoding="latin1").item()

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
y = tf.placeholder(tf.int32, (None))

#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
if layer == 'all':
  conv1W = tf.Variable(tf.truncated_normal([11,11,3,96], stddev=0.001), trainable=True)
  conv1b = tf.Variable(tf.truncated_normal([96], stddev=0.001), trainable=True)
else:
  conv1W = tf.Variable(net_data["conv1"][0], trainable=False)
  conv1b = tf.Variable(net_data["conv1"][1], trainable=False)
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
if layer == 'all':
  conv2W = tf.Variable(tf.truncated_normal([5,5,48,256], stddev=0.001), trainable=True)
  conv2b = tf.Variable(tf.truncated_normal([256], stddev=0.001), trainable=True)
else:
  conv2W = tf.Variable(net_data["conv2"][0], trainable=False)
  conv2b = tf.Variable(net_data["conv2"][1], trainable=False)
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
if layer == 'all':
  conv3W = tf.Variable(tf.truncated_normal([3,3,256,384], stddev=0.001), trainable=True)
  conv3b = tf.Variable(tf.truncated_normal([384], stddev=0.001), trainable=True)
else:
  conv3W = tf.Variable(net_data["conv3"][0], trainable=False)
  conv3b = tf.Variable(net_data["conv3"][1], trainable=False)
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
if layer == 'all':
  conv4W = tf.Variable(tf.truncated_normal([3,3,192,384], stddev=0.001), trainable=True)
  conv4b = tf.Variable(tf.truncated_normal([384], stddev=0.001), trainable=True)
else:
  conv4W = tf.Variable(net_data["conv4"][0], trainable=False)
  conv4b = tf.Variable(net_data["conv4"][1], trainable=False)
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)


#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
if layer == 'all':
  conv5W = tf.Variable(tf.truncated_normal([3,3,192,256], stddev=0.001), trainable=True)
  conv5b = tf.Variable(tf.truncated_normal([256], stddev=0.001), trainable=True)
else:
  conv5W = tf.Variable(net_data["conv5"][0], trainable=False)
  conv5b = tf.Variable(net_data["conv5"][1], trainable=False)
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#fc6
#fc(4096, name='fc6')
# only train this layer if layer == fc6
if layer == 'all' or layer == 'fc6':
  fc6W = tf.Variable(tf.truncated_normal([9216,4096], stddev=0.001), trainable=True)
  fc6b = tf.Variable(tf.truncated_normal([4096], stddev=0.001), trainable=True)
else:
  fc6W = tf.Variable(net_data["fc6"][0], trainable=False)
  fc6b = tf.Variable(net_data["fc6"][1], trainable=False)
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

#fc7
#fc(4096, name='fc7')
# train this layer if layer == fc6 or fc7
if layer == 'all' or layer == 'fc6' or layer == 'fc7':
  fc7W = tf.Variable(tf.truncated_normal([4096,4096], stddev=0.001), trainable=True)
  fc7b = tf.Variable(tf.truncated_normal([4096], stddev=0.001), trainable=True)
else:
  fc7W = tf.Variable(net_data["fc7"][0], trainable=False)
  fc7b = tf.Variable(net_data["fc7"][1], trainable=False)
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

#fc8
# always train this layer
output_size = 1 if final_layer == 'sigmoid' else 2
fc8W = tf.Variable(tf.truncated_normal([4096,output_size], stddev=0.001), trainable=True)
fc8b = tf.Variable(tf.truncated_normal([output_size], stddev=0.001), trainable=True)
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
if final_layer == 'sigmoid':
  prob = tf.sigmoid(fc8)
else:
  prob = tf.nn.softmax(fc8)
  
fc8W_sigmoid = tf.Variable(tf.truncated_normal([4096,1], stddev=0.001), trainable=True)
fc8b_sigmoid = tf.Variable(tf.truncated_normal([1], stddev=0.001), trainable=True)
fc8_sigmoid = tf.nn.xw_plus_b(fc7, fc8W_sigmoid, fc8b_sigmoid)
prob_sigmoid = tf.sigmoid(fc8_sigmoid)

# calculate loss based on final layer
if final_layer == 'sigmoid':
  loss = tf.losses.log_loss(labels=y, predictions=prob)
else:
  loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=fc8)

opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
var_list = [fc8W, fc8b]
if layer == 'fc7':
  var_list = [fc8W, fc8b, fc7W, fc7b]
if layer == 'fc6':
  var_list = [fc8W, fc8b, fc7W, fc7b, fc6W, fc6b]
if layer == 'all':
  var_list = [fc8W, fc8b, fc7W, fc7b, fc6W, fc6b, conv5W, conv5b, conv4W, conv4b, conv3W, conv3b, conv2W, conv2b, conv1W, conv1b]
train_op = opt.minimize(loss, var_list=var_list)

# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
final_path = os.path.join(save_folder, 'line-model-final.ckpt')
epoch_path = os.path.join(save_folder, 'line-model.ckpt')
if not os.path.exists(save_folder):
  os.makedirs(save_folder)
if not restore:
  # saving the initial model will overwrite the last save (if it exists), so that it is not restored
  saver.save(sess, final_path)
  
def errorRates(sess, imgs, labels, header):
  print(header)
  
  loss_out = sess.run(loss, feed_dict={x:imgs, y:labels})
  error = np.mean(loss_out)
  print("\tloss_error: {}".format(error))
    
  prob_out = sess.run(prob, feed_dict={x:imgs})
  if final_layer == 'sigmoid':
    raw_error = prob_out.flatten()
    raw_error_mean = np.mean(np.abs(1 - raw_error))
  else:  
    raw_error = prob_out[np.arange(len(prob_out)),np.ndarray.astype(labels, int)].flatten()
    raw_error_mean = np.mean(np.abs(1 - raw_error))
  print("\traw_error: {}".format(raw_error_mean))
  
  if final_layer == 'sigmoid':
    classification = np.ndarray.astype(np.round(prob_out), int).flatten()
  else:  
    classification = np.argmax(prob_out, axis=1)
  class_error = np.mean(classification != labels)
  print("\tclassification_error: {}".format(class_error))
  
  return error

buffer_min = 10
batchSize = 100
numImgs = 10000

# INFO to be printed, so hyperparameters were used for output
info_prefix="INFO\t"
layers_train_str = "fc8"
if layer == 'fc7':
  layers_train_str = "fc7 and fc8"
elif layer == 'fc6':
  layers_train_str = "fc6, fc7 and fc8"
print(info_prefix + "About to start training on {} examples over {} epochs in batches of {}".format(numImgs, max_epochs, batchSize))
print(info_prefix + "Will run for at most {} minutes".format(max_program_length))
print(info_prefix + "Output saved to {} and training started at epoch {}".format(save_folder, epoch_start))
print(info_prefix + "Training with a learning rate of {}".format(learning_rate))
print(info_prefix + "Training layers " + layers_train_str)
print(info_prefix + "With a final {} layer".format(final_layer))

correct = np.zeros(numImgs)
# first half are lines
correct[:int(numImgs/2)] = 1

# TODO - add argparse paramater to control images used
folder_prefix = os.path.join("../lines", lines_folder)
folder = "train"
imgs = []
for i in range(numImgs):
  img = (imread(os.path.join(folder_prefix,folder,str(i) + ".png"))[:,:,:3]).astype(float32)
  # img = img - mean(img)
  img[:, :, 0], img[:, :, 2] = img[:, :, 2], img[:, :, 0]
  img = img - mean[None,None,:]
  imgs.append(img)
imgs = np.array(imgs)

folder = "test"
imgs_val = []
numValImgs = 2000
for i in range(numValImgs):
  img = (imread(os.path.join(folder_prefix, folder, str(i) + ".png"))[:,:,:3]).astype(float32)
  # img = img - mean(img)
  img[:, :, 0], img[:, :, 2] = img[:, :, 2], img[:, :, 0]
  img = img - mean[None,None,:]
  imgs_val.append(img)
imgs_val = np.array(imgs_val)

folder = "validation"
imgs_test = []
numTestImgs = 2000
for i in range(numTestImgs):
  img = (imread(os.path.join(folder_prefix, folder, str(i) + ".png"))[:,:,:3]).astype(float32)
  # img = img - mean(img)
  img[:, :, 0], img[:, :, 2] = img[:, :, 2], img[:, :, 0]
  img = img - mean[None,None,:]
  imgs_test.append(img)
imgs_test = np.array(imgs_test)

correct_val = np.zeros(numTestImgs)
# first half are lines
correct_val[:int(numTestImgs/2)] = 1

fc8W_init = sess.run(fc8W)
fc8b_init = sess.run(fc8b)
fc7W_init = sess.run(fc7W)
fc7b_init = sess.run(fc7b)
fc6W_init = sess.run(fc6W)
fc6b_init = sess.run(fc6b)
conv5W_init = sess.run(conv5W)
conv5b_init = sess.run(conv5b)
conv4W_init = sess.run(conv4W)
conv4b_init = sess.run(conv4b)
conv3W_init = sess.run(conv3W)
conv3b_init = sess.run(conv3b)
conv2W_init = sess.run(conv2W)
conv2b_init = sess.run(conv2b)
conv1W_init = sess.run(conv1W)
conv1b_init = sess.run(conv1b)

def printChangeInVariables(sess, header):
  print(header)
  print("fc8W: {}".format(np.mean(np.abs(fc8W_init - sess.run(fc8W)))))
  print("fc8b: {}".format(np.mean(np.abs(fc8b_init - sess.run(fc8b)))))
  print("fc7W: {}".format(np.mean(np.abs(fc7W_init - sess.run(fc7W)))))
  print("fc7b: {}".format(np.mean(np.abs(fc7b_init - sess.run(fc7b)))))
  print("fc6W: {}".format(np.mean(np.abs(fc6W_init - sess.run(fc6W)))))
  print("fc6b: {}".format(np.mean(np.abs(fc6b_init - sess.run(fc6b)))))
  print("conv5W: {}".format(np.mean(np.abs(conv5W_init - sess.run(conv5W)))))
  print("conv5b: {}".format(np.mean(np.abs(conv5b_init - sess.run(conv5b)))))
  print("conv4W: {}".format(np.mean(np.abs(conv4W_init - sess.run(conv4W)))))
  print("conv4b: {}".format(np.mean(np.abs(conv4b_init - sess.run(conv4b)))))
  print("conv3W: {}".format(np.mean(np.abs(conv3W_init - sess.run(conv3W)))))
  print("conv3b: {}".format(np.mean(np.abs(conv3b_init - sess.run(conv3b)))))
  print("conv2W: {}".format(np.mean(np.abs(conv2W_init - sess.run(conv2W)))))
  print("conv2b: {}".format(np.mean(np.abs(conv2b_init - sess.run(conv2b)))))
  print("conv1W: {}".format(np.mean(np.abs(conv1W_init - sess.run(conv1W)))))
  print("conv1b: {}".format(np.mean(np.abs(conv1b_init - sess.run(conv1b)))))

with tf.Session() as sess:
  sess.run(init)
  
  # update following variables, as necessary
  global_step = 100*epoch_start
  saver.restore(sess, final_path)
  
  # test that restore worked
  printChangeInVariables(sess, "change in variables")
  
  errorRates(sess, imgs_val, correct_val, "Initial validation error")
  errorRates(sess, imgs_test, correct_val, "Initial, untrained, test error:")

  numBatchesInEpoch = int(numImgs / batchSize)
  
  for i in range(epoch_start, max_epochs):
    print("Starting epoch {}".format(i))
    perm = np.ndarray.astype(np.random.permutation(numImgs), int)
    for j, batch in enumerate(np.reshape(perm, (int(numImgs / batchSize), batchSize))):     
      sess.run(train_op, feed_dict={x:imgs[batch,],y:correct[batch,]})
      global_step += 1
      
      if global_step % int(numBatchesInEpoch/2) == 0:
        errorRates(sess, imgs[batch,], correct[batch,], "Train error for global step {}:".format(global_step))
    saver.save(sess, epoch_path, global_step=global_step)
      
    error = errorRates(sess, imgs_val, correct_val, "Validation error after epoch {}:".format(i))
    if error < 0.01:
      print("End epochs because validation error is low enough")
      break
    elif i >= max_epochs - 1:
      print("End epochs because reached max epochs")
      break
    if float(time.time() - start_time)/60 > max_program_length - buffer_min:
      print("End iteration because max time was reached")
      break
    
  print("Finished!")
  saver.save(sess, final_path)

  errorRates(sess, imgs_test, correct_val, "Test error:")

  printChangeInVariables(sess, "final change in variables")
