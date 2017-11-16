#!/usr/bin/python3

#read from a trf record for tensorflow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
data_path = '/home/kube_master/mindt/fingerprint_test/classification_tensorflow/train_fingerprint.tfrecords'  # address to save the hdf5 file
with tf.Session() as sess:
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
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
    image = tf.reshape(image, [512, 512, 1])
    
    # Any preprocessing here ...
    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)
 # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for batch_index in range(5):
        img, lbl = sess.run([images, labels])
        print(lbl)
        lbl = tf.one_hot(lbl,2,1,0,-1)
        print(lbl)
        lbls = lbl.eval()
        print(lbls)
        img = img.astype(np.uint8)
        for j in range(6):
            plt.subplot(2, 3, j+1)
            image_to_plot = img[j,:,:]
            plt.imshow(image_to_plot.reshape(512,512))
            print(image_to_plot.shape)
            plt.title('filename_not_contain_6' if lbl[j]==0 else 'file_name_contain_6')
        plt.show()
    # Stop the threads
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)
    sess.close()
