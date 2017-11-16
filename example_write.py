#!/usr/bin/python3

# save files to a trf record
import sys
import tensorflow as tf
from random import shuffle
import glob
from input_image_parser import input_image_parser as input_parser
shuffle_data = True  # shuffle the addresses before saving
fingerprint_train_path = '/home/kube_master/mindt/fingerprint_test/NIST/sd04/png_txt/figs_*/*.png'
fingerprint_label_path = '/home/kube_master/mindt/fingerprint_test/NIST/sd04/png_txt/figs_*/*.txt'
# read addresses and labels from the 'train' folder
addrs = glob.glob(fingerprint_train_path)
#test_statement
labels = [1 if '6' in addr else 0 for addr in addrs]  # random test of 0 and 1 
# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

# Divide the hata into 60% train, 20% cross-validation, and 20% test
train_addrs = addrs[0:int(0.6*len(addrs))]
train_labels = labels[0:int(0.6*len(labels))]
val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = labels[int(0.8*len(labels)):]

#print(len(train_addrs))
#print(len(train_labels))
#print(train_labels)
#print(len(val_addrs))
#print(len(val_labels))
#print(len(test_addrs))
#print(type(test_addrs))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def example_write(filename, purpose, addrs, labels):
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(len(addrs)):
        # Load the image
        img, label = input_parser(addrs[i], labels[i])
        # print how many images are saved every 1000 images
        if not i % 1000:
            print('%s data: {}/{}'.format(purpose, i, len(addrs)))
            print(img.dtype)
            sys.stdout.flush()
        # Create a feature
        feature = {"%s/label" % purpose: _int64_feature(label),
                   "%s/image" % purpose: _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
    
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    
    writer.close()
    sys.stdout.flush()

example_write("train_fingerprint.tfrecords", "train", train_addrs, train_labels)
example_write("val_fingerprint.tfrecords", "cross_val", val_addrs, val_labels)
example_write("test_fingerprint.tfrecords", "test", test_addrs, test_labels)


