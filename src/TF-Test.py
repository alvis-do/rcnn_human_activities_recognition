import numpy as np 
import tensorflow as tf 

with tf.Session() as sess:
	sigmoid = tf.sigmoid([0, 1000., 12.])
	#tf.Print(sigmoid, [sigmoid])
	print(sigmoid.eval())