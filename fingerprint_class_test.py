#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



#============================================================================================================
#A function that parses images and labels
def _parse_function(example_proto):
  features = {"train/image": tf.FixedLenFeature([], tf.string),
              "train/label": tf.FixedLenFeature([], tf.int64)}
  parsed_features = tf.parse_single_example(example_proto, features)
  parsed_features['train/label'] = tf.one_hot(parsed_features['train/label'],2,1,0,-1)
  parsed_features['train/image'] = tf.decode_raw(parsed_features['train/image'], tf.uint8)
  return parsed_features["train/image"], parsed_features["train/label"]
#============================================================================================================



#============================================================================================================
#setting a iterator which batch the data and iterate through each epoch
filenames = ["/home/kube_master/mindt/fingerprint_test/classification_tensorflow/train_fingerprint.tfrecords"]
dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(10)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
#============================================================================================================



#============================================================================================================
#Convolutional Neural network
x = tf.placeholder(tf.float32, shape=[None,512*512])
y_ = tf.placeholder(tf.float32, shape=[None,2])
#abstract function calls which generates the initial weights and bias
def init_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def init_bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
# convolution layer function
#strides are for [batch, height, width, channel]
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = 'SAME')
#max pool of a 4*4 block
def max_pool4x4(x):
    return tf.nn.max_pool(x, ksize=[1,4,4,1], strides = [1, 4, 4, 1], padding='SAME')
# patch of 16*16 on the image gives  outputs
W_conv1 = init_weight_variable([16, 16, 1, 1])
b_conv1 = init_bias_variable([1])
x_image = tf.reshape(x, [-1, 512, 512, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# step 4*4 pooling means we have a 128*128 image now
h_pool1 = max_pool4x4(h_conv1)
#patch of 16*16 image takes in 6 inputs and give out 6 outputs
W_conv2 = init_weight_variable([16, 16, 1, 1])
b_conv2 = init_bias_variable([1])
# same pooling now means that we only have a 32*32 image
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool4x4(h_conv2)
#weights for a flattened matrix of 1024 outputs from 32*32*1 deep features
W_fc1 = init_weight_variable([32*32*1, 1024])
b_fc1 = init_bias_variable([1024])
# relu is the rectified linear function which is a "biologically imitating" activation function
h_pool2_flat = tf.reshape(h_pool2, [-1, 32*32*1])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# drop out of some neurons during training can be seen as a way of regularization
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# second hidden layer where 1024 neurons feeds to 1024 neurons, wow!
W_fc2 = init_weight_variable([1024, 1024])
b_fc2 = init_bias_variable([1024])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
# evaluation layer where a simple linear matrix multiplication is used
# this is due to the softmax(multi-variable logistic regression) is used for the cost and prediction
W_fc3 = init_weight_variable([1024, 2])
b_fc3 = init_bias_variable([2])
y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
#======
#Multi-class Linear regression
#W = tf.Variable(tf.zeros([512*512,2]))
#b = tf.Variable(tf.zeros([2]))
#tf.InteractiveSession().run(tf.global_variables_initializer())
#y = tf.matmul(x,W) +b
#============================================================================================================



#============================================================================================================
#Defining Cost function and Optimisation method
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(0.5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#============================================================================================================



#============================================================================================================
#Training
with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Compute for 100 epochs.
    for _ in range(1):
        #a = 0
        sess.run(iterator.initializer)
        while True:
            try:
                data = sess.run(next_element)
                # First term in data is image and second term is the one hot label
                train_step.run(feed_dict={x: data[0], y_: data[1], keep_prob: 0.6})
                print("done batch")
            except tf.errors.OutOfRangeError:
                if _%1 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: data[0], y_: data[1], keep_prob: 1.0})
                    print("epoch_number:", _, "training_accuracy:", train_accuracy)
                break
#============================================================================================================


#Testing statements
#                if a == 6:
#                #plot image
#                    plt.imshow(data[0][3].reshape(512,512))
#                plt.show()
#                #print label
#                print(data[1][1])
#                a+=1


