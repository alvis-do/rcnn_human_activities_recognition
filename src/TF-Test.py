import numpy as np 
import tensorflow as tf 

with tf.Session() as sess:
	a = tf.ones([4, 3,2])
	b = tf.ones([3,2])
	c = tf.map_fn(lambda x: tf.cast(tf.equal(tf.argmax(x, 1), tf.argmax(b, 1)), tf.float32), a)
	print(c.eval())