#!/usr/bin/python3

import numpy as np

def batch(files, labels, batch_size):
    #Length of the files array and the labels array should hold the same value
    length = len(files)
    files = np.array_split(files, length/batch_size)
    labels = np.array_split(labels, length/batch_size)
    return files,labels

def batch_feed(train_files, train_labels, it_num, conv_func):
    train_files, train_labels = conv_func(train_files[it_num], train_labels[it_num])
    return train_files, train_labels

#test function
def test_conv(train_files, train_labels):
    train_files = train_files*3
    train_labels = train_labels*6
    return train_files, train_labels

#test case
files = np.arange(25)
labels = np.random.randint(2, size=25)
batch_size = 4
#print(files)

train_files, train_labels = batch(files, labels, batch_size)

print(train_files)
print(train_labels)

print(batch_feed(train_files, train_labels, 3, test_conv))
