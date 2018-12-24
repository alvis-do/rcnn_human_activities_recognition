
# coding: utf-8

# In[53]:


import tensorflow as tf
import numpy as np
import os
import cv2
import sys
from matplotlib import pyplot as plt

# sys.path.insert(0, 'tf_cnnvis/tf_cnnvis')
from tf_cnnvis import deconv_visualization

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# In[23]:


video_path = '/media/vudhk/Storage/DHBK/HK181/LVTN/UCF4/biking/v_biking_03/v_biking_03_04.mpg'
frame_num = 70
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
_, frame = cap.read()
frame = cv2.resize(frame, (224,224))
label = [1,0,0,0]
# cv2.imshow('ImageWindow',frame)
# cv2.waitKey()


# In[38]:


modelpath = '../../model_saver/model_10.ckpt'
saver = tf.train.import_meta_graph("{}.meta".format(modelpath))

with tf.Session() as sess:
    saver.restore(sess, modelpath)
    g = sess.graph

    max_pool = g.get_tensor_by_name('CNN_MODEL/MaxPool:0')
    X = g.get_tensor_by_name('CNN_MODEL/input:0')
    y = g.get_tensor_by_name("CNN_OP/y_true:0")
    X_in = g.get_tensor_by_name('CNN_MODEL/truediv:0')
    

    # init global vars
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, modelpath)

    _input = [frame]
    _label = [label]
    
    feed_dict = {X: _input, y: _label}
    
    layers = ["CNN_MODEL/MaxPool", "CNN_MODEL/MaxPool_1", "CNN_MODEL/MaxPool_2"]
    
    deconv_visualization(sess_graph_path = sess, 
                        value_feed_dict = feed_dict, 
                        input_tensor=X_in, 
                        layers=layers, 
                        path_logdir=os.path.join("Log","UCF4"), 
                        path_outdir=os.path.join("Output","UCF4"))

