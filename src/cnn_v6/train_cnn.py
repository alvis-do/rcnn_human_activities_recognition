#!/usr/bin/env python
# coding: utf-8

# 0-basketball_shooting; 
# 1-biking; 
# 2-diving; 
# 3-golf_swing

# In[1]:


import numpy as np
import tensorflow as tf
import pandas as pd
import os
from config import params

# Register first time tf.contrib namspace for loading graph from meta ckpt
# because ops  in this are lazily registered
dir(tf.contrib)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


data = np.load(params['DATA_PATH']).item()
data['X_train'] = data['X_train']
data['X_test'] = data['X_test']
train_len = data['X_train'].shape[0]
test_len = data['X_test'].shape[0]

# In[4]:


# Training Parameters
learning_rate = 0.0001
num_steps = 200
display_step = 5


def get_batch(dict_field_name, batch):
    return data[dict_field_name][batch*params['CNN_BATCH_SIZE']:min((batch+1)*params['CNN_BATCH_SIZE'],train_len)]

saver = tf.train.import_meta_graph("./graph/cnn_lstm_model.ckpt.meta")

config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.

with tf.Session(config=config) as sess:
    # Construct model
    saver.restore(sess, "./graph/cnn_lstm_model.ckpt")
    g = sess.graph
    with g.name_scope("TRAIN_CNN"):

        Y = tf.placeholder(tf.float32, [None, params['N_CLASSES']], name='y_train')
        X = g.get_tensor_by_name("CNN_MODEL/input:0")
        logits = g.get_tensor_by_name("CNN_MODEL/out_class:0")


        prediction = tf.nn.softmax(logits)
        #prediction = tf.Print(prediction,[prediction],summarize=20)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)


        # Evaluate model
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    # Run the initializer
    sess.run(init)
        
    epoch = 0
        
    while True:
        #total_batch = train_len // params['CNN_BATCH_SIZE']
        total_batch =  5
        for batch in range(total_batch):
            # lay batch tiep theo
            batch_input = get_batch('X_train', batch) 
            batch_label = get_batch('y_train_class_only', batch)
            #print("Y: ", batch_label)
            _, loss, acc = sess.run([train_op, loss_op, accuracy], feed_dict={X: batch_input, Y: batch_label})

            if batch%display_step == 0:
                print("Epoch " + str(epoch) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))

                    # avg_acc_test = 0.
                    # total_batch_test =  test_len // CNN_BATCH_SIZE
                    # for batch_test in range(total_batch_test):
                    #     batch_input_test = get_batch('X_test', batch_test) 
                    #     batch_label_test = get_batch('y_test_class_only', batch_test) 
                    #     acc_test = sess.run(accuracy, feed_dict={X:batch_input_test, Y:batch_label_test})
                    #     avg_acc_test += acc_test / total_batch_test
                    # print("Accuracy on test set: %.5f \n" % (avg_acc_test))

            
        epoch += 1   
        if epoch % 5 == 0:
            save_path = saver.save(sess, "{}/model_{}.ckpt".format(params['CNN_MODEL_SAVER_PATH'],epoch))
            print("Model saved in path: %s" % save_path)
            
     