#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

array_file_path =  '/home/kube_master/mindt/data/raw_data/╧р┐╪╒є═╝╞╫╚▒╧▌┤·┬ы╖╓└р/═╝╞╫/31A-PM02001-42W─┌░╝/data_api.npz'

images = np.load(array_file_path)

#print(images.files)

plt.imshow(images['arr_0'])

plt.show()


