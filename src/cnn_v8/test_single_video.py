
import numpy as np
import tensorflow as tf
import pandas as pd
import os, sys
import time, math
from config import params
from data_handler import get_dataset, get_batch, get_clip, split_valid_set


# Register first time tf.contrib namspace for loading graph from meta ckpt
# because ops  in this are lazily registered
dir(tf.contrib)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

model_path = "./model_saver_cnn_lstm/model_1.ckpt"
saver = tf.train.import_meta_graph("{}.meta".format(model_path))


# setup config for session
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.


with tf.Session(config=config) as sess:
    saver.restore(sess, model_path)
    g = sess.graph

    frames 

    _current_state = np.zeros((num_layers_lstm, 2, params['CNN_LSTM_BATCH_SIZE'], n_hidden_lstm))
    