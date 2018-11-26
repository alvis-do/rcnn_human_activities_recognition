
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from config import params
from data_handler import get_dataset, get_batch, get_clip

# Register first time tf.contrib namspace for loading graph from meta ckpt
# because ops  in this are lazily registered
dir(tf.contrib)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

cnn_model_num = 5
learning_rate = 0.0001
lambda_loss_amount = 0.0015

model_path = "./graph/cnn_lstm_model.ckpt"
#model_path = "{}/model_{}.ckpt".format(params['CNN_MODEL_SAVER_PATH'], cnn_model_num)
saver = tf.train.import_meta_graph("{}.meta".format(model_path))

# setup config for session
config = tf.ConfigProto(log_device_placement=False)
# config.gpu_options.allow_growth = True
# config.gpu_options.per_proces_gpu_memory_fraction = 1.
with tf.Session(config=config) as sess:
    saver.restore(sess, model_path)
    g = sess.graph
    with g.name_scope("TRAIN_LSTM"):

        X = g.get_tensor_by_name("CNN_MODEL/input:0")
        #cnn_features = g.get_tensor_by_name("CNN_MODEL/out_features:0")

        #lstm_X = g.get_tensor_by_name("LSTM_MODEL/input:0")
        isTraining = g.get_tensor_by_name("LSTM_MODEL/is_training:0")
        y_pred = g.get_tensor_by_name("LSTM_MODEL/output:0")
        Y = tf.placeholder(tf.float32, [None, params['N_CLASSES']], name='y_true')

        #cnn_features_reshape = tf.reshape(cnn_features, [tf.shape(Y)[0], params['N_FRAMES'], params['N_FEATURES']])
        
        # Define loss and optimizer
        l2 = lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        #l2 = 0
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=y_pred)) + l2
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='lstm_adam')
        train_op = optimizer.minimize(loss_op)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    # init session
    sess.run(tf.global_variables_initializer())

    train_set, test_set = get_dataset()

    # Start training
    epoch = 0
    while True:
        for step, clips in get_batch(train_set, params['CNN_LSTM_BATCH_SIZE']):
            if step == 0:
                continue

            frames, labels = get_clip(clips)
            input_ = np.reshape(frames, (-1, params['INPUT_WIDTH'], params['INPUT_HEIGHT'], params['INPUT_CHANNEL']))

            _, loss, acc = sess.run([train_op, loss_op, accuracy], 
                feed_dict={
                    X: input_, 
                    #lstm_X: cnn_features_reshape, 
                    Y: labels, 
                    isTraining: True
                })

            print("Epoch {} - batch {}: Loss {} - Acc {}".format(epoch, step - 1, loss, acc))

        epoch += 1

        if epoch % 10 == 0:
            save_path = saver.save(lstm_sess, "{}/model_{}.ckpt".format(MODEL_SAVER_ALL_PATH, epoch))
            print("Model saved in path: %s" % save_path)

        del lstm_input

