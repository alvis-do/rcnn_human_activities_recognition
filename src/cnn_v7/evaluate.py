import tensorflow as tf
from data_handle import get_dataset, get_batch, get_clip, split_valid_set


# Recreate the exact same model, including weights and optimizer.
model = tf.keras.models.load_model('my_model.h5')

_, test_set = get_dataset()

data, labels = get_clip(test_set)

model.evaluate(data, labels, batch_size=32)