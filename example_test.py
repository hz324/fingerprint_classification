#!/usr/bin/python3
import tensorflow as tf
import numpy as np

tfrecords_filename = '/home/kube_master/mindt/fingerprint_test/classification_tensorflow/train_fingerprint.tfrecords'
record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

for string_record in record_iterator:
    # Parse the next example
    example = tf.train.Example()
    example.ParseFromString(string_record)

    # Get the features you stored (change to match your tfrecord writing code)
    #height = int(example.features.feature['height']
    #                             .int64_list
    #                             .value[0])

    width = int(example.features.feature['train/label']
                                .int64_list
                                .value[0])

    img_string = (example.features.feature['train/image']
                                  .bytes_list
                                  .value[0])
    # Convert to a numpy array (change dtype to the datatype you stored)
    img_1d = np.fromstring(img_string, dtype=np.uint8)
    # Print the image shape; does it match your expectations?
    print(img_1d.shape)
