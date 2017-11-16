#! /usr/bin/python3

import tensorflow as tf

#sess = tf.InteractiveSession()

import numpy as np
import matplotlib.pyplot as plt

x = tf.placeholder(tf.float32, shape=[None,512*512])

y_ = tf.placeholder(tf.float32, shape=[None,2])

W = tf.Variable(tf.zeros([512*512,2]))

b = tf.Variable(tf.zeros([2]))

tf.InteractiveSession().run(tf.global_variables_initializer())

y = tf.matmul(x,W) +b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

data_path = 'train_fingerprint.tfrecords'  # address to save the hdf5 file
with tf.Session() as sess:
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=100)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['train/image'], tf.uint8)

    # Cast label data into int32
    label = tf.cast(features['train/label'], tf.int32)
    # Reshape image data into the original shape
    image = tf.reshape(image, [512*512])

    # Any preprocessing here ...
    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], batch_size=600, capacity=1000, num_threads=1000, min_after_dequeue=10, allow_smaller_final_batch=True)
    #iterator_img = images.make_initializable_iterator()
    #iterator_lbl = labels.make_initializable_iterator()
    #next_element_lbl = iterator_lbl.get_next()
    #next_element_img = iterator_img.get_next()
    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    #for epoch in range(2):
    #sess.run([iterator_lbl.initialiser, iterator_img.initialiser])
    for batch_index in range(2):
        img, lbl = sess.run([images, labels])
        #print(lbl)
        lbl = tf.one_hot(lbl,2,1,0,-1)
        #print(lbl)
        lbl = lbl.eval()
        #print(lbl)
        img = img.astype(np.uint8)
        #print(img.shape)
        train_step.run(feed_dict={x: img, y_: lbl})
        if batch_index % 1 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: img, y_: lbl})
            print('batch %d, training accuracy %g' % (batch_index, train_accuracy))
        #for j in range(6):
        #    plt.subplot(2, 3, j+1)
        #    image_to_plot = img[j,:,:]
        #    plt.imshow(image_to_plot.reshape(512,512))
        #    print(image_to_plot.shape)
        #    plt.title('filename_not_contain_6' if lbl[j]==0 else 'file_name_contain_6')
        #plt.show()
    # Stop the threads
            coord.request_stop()

    # Wait for threads to stop
            coord.join(threads)
    sess.close()

