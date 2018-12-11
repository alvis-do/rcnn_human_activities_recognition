
import numpy as np
import tensorflow as tf
import os, sys
from model import CNN, LRCN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics, losses
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from data_handle import get_dataset, get_batch, get_clip, split_valid_set

# Register first time tf.contrib namspace for loading graph from meta ckpt
# because ops  in this are lazily registered
dir(tf.contrib)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
filepath = 'lrcn_model.h5'
resume_train = False



train_set, _ = get_dataset()
np.random.shuffle(train_set)

train_length = len(train_set)
valid_length = round(train_length * 0.2)

valid_set = train_set[train_length - valid_length : train_length]
train_set = train_set[:train_length - valid_length]


# class NBatchLogger(Callback):
# 	def __init__(self,batch_display=100):
# 		'''
# 		display: Number of batches to wait before outputting loss
# 		'''
# 		self.seen = 0
# 		self.batch_display = batch_display

# 	def on_batch_end(self,batch,logs={}):
# 		if batch % self.batch_display == 0:
# 			print('')
# 			# print('\n{}/{} - Batch loss: {} - categorical_accuracy: {}'.format(self.seen, self.params['samples'], logs['loss'], logs['categorical_accuracy']))


def checkpoint():
	return ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

def callbacks():
	return [
		# Interrupt training if `val_loss` stops improving for over 2 epochs
		# tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
		# Write TensorBoard logs to `./logs` directory
		tf.keras.callbacks.TensorBoard(log_dir='./logs'),
		checkpoint()
		# batchLog
	]



def batch_gen(dataset):
	while 1:
		for d in dataset:
			x, y = get_clip([d])
			yield (x, y)



if not resume_train:
	_metrics = [metrics.categorical_accuracy]

	# batchLog = NBatchLogger(batch_display=10)

	model = LRCN()
	model.build((None, 16, 224, 224, 3))
	model.compile(loss=losses.categorical_crossentropy, optimizer=Adam(0.0001), metrics=_metrics)
	# model.summary()
else:
	model = tf.keras.models.load_model(filepath)


history = model.fit_generator(batch_gen(train_set),
							steps_per_epoch=64, 
							epochs=100, 
							validation_data=batch_gen(valid_set), 
							validation_steps=64,
							use_multiprocessing=True, 
							callbacks=callbacks())
