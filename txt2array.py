#!/usr/bin/python3
import numpy as np
def txt2array(input_file_path, output_file_path):
    text_file = open(input_file_path, "r")
    lines = text_file.readlines()
    data = np.array(list(map(lambda x: x.strip().split(","), lines)))
    #labels = data[:,1].astype(np.int32)
    #print(labels)
    dataset = np.vstack((data[:,0], data[:,1]))
    #print(lines)
    print(dataset)
    np.save(output_file_path, dataset)

output_file_path = '/home/kube_master/mindi/fingerprint_test/classification_tensorflow/np_array_test'
input_file_path = '/home/kube_master/mindi/fingerprint_test/classification_tensorflow/plain_text_test.txt'

txt2array(input_file_path , output_file_path)

