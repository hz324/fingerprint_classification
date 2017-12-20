#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
import zipfile
import glob

all_input_file_paths = '/home/kube_master/mindt/data/raw_data/╧р┐╪╒є═╝╞╫╚▒╧▌┤·┬ы╖╓└р/═╝╞╫/*/'
#Please do NOT keep any .npy or other arrays that are useful to you in the output_file_path before running the following code
#Or you can comment out the last line in the compress_output() function
output_file_path = '/home/kube_master/mindt/data/raw_data/'
data_api_path = "/home/kube_master/mindt/data_api/data_api"

def prec2txt(input_file_path):
    #make sure the file directory ends with a "/"
    call([data_api_path, input_file_path])


def txt2array(input_file_path, output_file_path, label):
    text_file = open(input_file_path, "r")
    lines = text_file.readlines()
    images = np.array(list(map(lambda x: x.strip().split("*"), lines)))[0][1:-1]
    #print(len(images))
    n=0
    name_list = np.arange(len(images))
    for image in images:
        data = np.array(image.split(","))
        image_array = np.reshape(data[2:-1], (int(data[0]),int(data[1]))).astype(np.float)
        print(np.shape(image_array))
        #plt.imshow(image_array)
        #print(image_array)
        np.save(output_file_path + "array_image" + "_label_" + str(label) + "_" + str(n), image_array)
        n+=1
    #output_file.close()
    #plt.show()

def compress_output(output_file_path):
    output_images = glob.glob( output_file_path + "array_image*")
    with zipfile.ZipFile(output_file_path + "array_image_data" + ".zip", "w") as myzip:
        for output in output_images:
            myzip.write(output)
    #comment out the line below if you want to keep both the archived zip file and the loose arrays
    call(["find", output_file_path, "-name", "*.npy","-delete"])

def overall_conv(all_input_file_paths, output_file_path):
    addrs = glob.glob(all_input_file_paths)
    #test_labels
    labels = np.arange(len(addrs)) 
    for i in range(len(addrs)):
        prec2txt(addrs[i])
        #another testing string here in the line below
        txt2array(addrs[i] + "data_api.txt", output_file_path, labels[i])
    compress_output(output_file_path)
#overall_conv(all_input_file_paths, output_file_path)

#testing statements
#prec2txt("/home/kube_master/mindt/data/raw_data/╧р┐╪╒є═╝╞╫╚▒╧▌┤·┬ы╖╓└р/═╝╞╫/32-PG-07056-K-15W─┌░╝/")
#txt2array(input_file_path , output_file_path)
#compress_output(output_file_path)
