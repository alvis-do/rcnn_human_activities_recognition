import tensorflow as tf

# Recreate the exact same model, including weights and optimizer.
model = tf.keras.models.load_model('my_model.h5')



result = model.predict(data, batch_size=16)
print(result.shape)