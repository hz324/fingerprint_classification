#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
def txt2array(input_file_path, output_file_path):
    text_file = open(input_file_path, "r")
    lines = text_file.readlines()
    images = np.array(list(map(lambda x: x.strip().split("*"), lines)))[0][1:-1]
    #print(images)
    for image in images:
        data = np.array(image.split(","))
        image_array = np.reshape(data[2:-1], (int(data[0]),int(data[1]))).astype(np.float)
        print(np.shape(image_array))
        plt.imshow(image_array)
        #print(image_array)
        #output_file = open(output_file_path, 'a')
    np.savez_compressed(output_file_path, image_array)
    #output_file.close()
    plt.show()

input_file_path = '/home/kube_master/mindt/data/raw_data/╧р┐╪╒є═╝╞╫╚▒╧▌┤·┬ы╖╓└р/═╝╞╫/31A-PM02001-42W─┌░╝/data_api.txt'
output_file_path = '/home/kube_master/mindt/data/raw_data/╧р┐╪╒є═╝╞╫╚▒╧▌┤·┬ы╖╓└р/═╝╞╫/31A-PM02001-42W─┌░╝/data_api'

txt2array(input_file_path , output_file_path)

