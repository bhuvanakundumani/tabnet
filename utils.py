import os
import numpy as np
import tensorflow as tf

def check_dir(dir_name):
    #Check for the presence of the directory else create one
    if not os.path.isdir(dir_name):
        os.system(f"mkdir {dir_name}")

def glu_layer(value, n_units):
    out_val = value[:, :n_units] * tf.nn.sigmoid(value[:, n_units:])
    return out_val
