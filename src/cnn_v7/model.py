import tensorflow as tf
from tensorflow.keras import layers, initializers

def CNN(is_activate_last=True):
	model = tf.keras.Sequential()
	# model.add(layers.InputLayer(input_shape=(224, 224, 3)))
	model.add(layers.Conv2D(96, 7, 2, "valid",  
							activation="relu", 
							kernel_initializer=initializers.RandomNormal(stddev=0.05),
							bias_initializer=initializers.RandomNormal(stddev=0.05)))
	model.add(layers.MaxPool2D(3, 2, "same"))
	model.add(layers.Conv2D(256, 5, 2, "valid",
							activation="relu", 
							kernel_initializer=initializers.RandomNormal(stddev=0.05),
							bias_initializer=initializers.RandomNormal(stddev=0.05)))
	model.add(layers.MaxPool2D(3, 2, "same"))
	model.add(layers.Conv2D(384, 3, 1, "same",
							activation="relu", 
							kernel_initializer=initializers.RandomNormal(stddev=0.05),
							bias_initializer=initializers.RandomNormal(stddev=0.05)))
	model.add(layers.Conv2D(384, 3, 1, "same",
							activation="relu", 
							kernel_initializer=initializers.RandomNormal(stddev=0.05),
							bias_initializer=initializers.RandomNormal(stddev=0.05)))
	model.add(layers.Conv2D(256, 3, 1, "same",
							activation="relu", 
							kernel_initializer=initializers.RandomNormal(stddev=0.05),
							bias_initializer=initializers.RandomNormal(stddev=0.05)))
	model.add(layers.MaxPool2D(3, 2, "same"))
	# model.add(layers.Dropout(0.5))
	model.add(layers.Flatten())
	model.add(layers.Dense(4096, 
							activation='relu', 
							kernel_initializer=initializers.RandomNormal(stddev=0.05),
							bias_initializer=initializers.RandomNormal(stddev=0.05)))
	model.add(layers.Dense(4096, 
							activation='relu', 
							kernel_initializer=initializers.RandomNormal(stddev=0.05),
							bias_initializer=initializers.RandomNormal(stddev=0.05)))
	# model.add(layers.Dropout(0.5))
	model.add(layers.Dense(4, 
							activation='softmax',
							kernel_initializer=initializers.RandomNormal(stddev=0.05),
							bias_initializer=initializers.RandomNormal(stddev=0.05)))

	return model


def LRCN():
	model = tf.keras.Sequential()
	model.add(layers.TimeDistributed(CNN(), input_shape=(16,224,224,3)))
	# model.add(layers.TimeDistributed(layers.Activation('relu'), input_shape=(16,4)))
	model.add(layers.LSTM(512, return_sequences=True, input_shape=(16,4)))
	model.add(layers.LSTM(512))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(4))
	model.add(layers.Activation('softmax'))
	return model