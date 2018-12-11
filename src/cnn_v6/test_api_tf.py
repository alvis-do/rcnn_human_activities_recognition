import tensorflow as tf

with tf.Session() as sess:
	x = tf.ones([1, 3, 5])
	y = tf.unstack(x, axis=1)
	print(y)
	print(y.eval())