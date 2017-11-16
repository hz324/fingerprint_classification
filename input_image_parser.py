# This file simply defines a function that takes an input image and return it to tensorflow
import tensorflow as tf
import numpy as np
from scipy import misc

def input_image_parser(img_name, label):
    #initiating variables, remember to do this for a different dataset
    #img_path = '/home/kube_master/mindi/fingerprint_test/NIST/sd04/png_txt/'
    num_classes = 2 
    # convert image label to one-hot
    #sess = tf.Session()
    #one_hot = tf.one_hot(label, num_classes)
    #one_hot = sess.run(one_hot)
    #one_hot = "test_one_hot"
    # read the image
    img_decoded = misc.imread(img_name)
    #img_decoded = tf.pack(img_decoded)

    #img_file = tf.read_file(img_path + img_name)
    #img_decoded = tf.image.decode_image(img_file, channels=1)

    return img_decoded, label #, one_hot


#img_test_name = 'figs_6/f1643_09.png'

#img, one_hot = input_image_parser(img_test_name, 6)

#print(np.shape(img))
