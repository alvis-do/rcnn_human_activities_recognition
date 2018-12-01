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
import os, sys
import math
from config import params
from data_handler import get_dataset, get_batch, get_clip, split_valid_set

# Register first time tf.contrib namspace for loading graph from meta ckpt
# because ops  in this are lazily registered
dir(tf.contrib)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Get data set
#train_set, test_set = get_dataset()


 # Get data set
train_set, test_set = get_dataset()



# Training Parameters
learning_rate = 0.0001
num_steps = 200
display_step = 1



# def get_batch(dict_field_name, batch):
#     return data[dict_field_name][batch*params['CNN_BATCH_SIZE']:min((batch+1)*params['CNN_BATCH_SIZE'],train_len)]

train_mode = True if sys.argv[1] == '--train' else False

# model_path = "./model_saver_cnn/model_4.ckpt"
model_path = "./graph/cnn_lstm_model.ckpt"
saver = tf.train.import_meta_graph("{}.meta".format(model_path))


config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.

with tf.Session(config=config) as sess:
    # Construct model
    saver.restore(sess, model_path)
    g = sess.graph
    with g.name_scope("TRAIN_CNN"):

        Y = tf.placeholder(tf.float32, [None, params['N_CLASSES']], name='y_train')
        X = g.get_tensor_by_name("CNN_MODEL/input:0")
        logits = g.get_tensor_by_name("CNN_MODEL/out_class:0")  
        # logits = tf.Print(logits, [logits], summarize=4*32)

        cnn_wc1 = g.get_tensor_by_name("CNN_MODEL/wc1:0");


        prediction = tf.nn.softmax(logits)
        #prediction = tf.Print(prediction,[prediction],summarize=20)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='cnn_adam')
        train_op = optimizer.minimize(loss_op)
       
        # Evaluate model
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        tf.summary.scalar("loss_op", loss_op)
        tf.summary.scalar("accuracy", accuracy)
        merged_summary_op = tf.summary.merge_all()

        #saver.save(sess, "{}/graph_model_train_cnn.ckpt".format(params['CNN_MODEL_SAVER_PATH']))


    tf_writer = tf.summary.FileWriter(params['CNN_MODEL_SAVER_PATH'])

    # Run the initializer
    if train_mode:
        sess.run(tf.global_variables_initializer())

        
    stride = 4
    epoch = 0
    acc_max = 0
    gl_step = 0
    while True:
        if train_mode:
            train_len = len(train_set) * (params['N_FRAMES'] // stride)
            batch_size = params['CNN_BATCH_SIZE'] // stride
            print("Training {} frames ...".format(train_len))
            total_batch = math.ceil(train_len // batch_size)
            loss_sum = []
            acc_sum = []
            for batch, clips in get_batch(train_set, batch_size):
                if batch == 0:
                    continue
                if len(clips) == 0:
                    break
                # lay batch tiep theo
                #batch_input = get_batch('X_train', batch) 
                #batch_label = get_batch('y_train_class_only', batch)
                #print("Y: ", batch_label)
                
                frames, batch_label = get_clip(clips, stride)
                batch_label = np.repeat(batch_label, stride, axis=0)
                batch_input = np.reshape(frames, (-1, params['INPUT_WIDTH'], params['INPUT_HEIGHT'], params['INPUT_CHANNEL']))
                s = np.arange(frames.shape[0])
                np.random.shuffle(s)
                batch_input = batch_input[s]
                batch_label = batch_label[s]

                _, loss, acc, summ, wc1 = sess.run([train_op, loss_op, accuracy, merged_summary_op, cnn_wc1], feed_dict={X: batch_input, Y: batch_label})
                loss_sum.append(loss)
                acc_sum.append(acc)

                print("-- batch x")

                tf_writer.add_summary(summ, gl_step)
                gl_step += 1
                
                # if batch%display_step == 0:
                #     print("Epoch {}, Batch {} Loss = {}, Training Accuracy = {}".format(epoch, batch, loss_sum / batch, acc_sum / batch))
            print("Epoch {}: Loss = {}, Training Accuracy = {}".format(epoch, np.mean(loss_sum), np.mean(acc_sum)))
                    

        #print("Testing {} frames ...".format(len(test_set) * (params['N_FRAMES'] // stride)))
        arr_acc_test = []
        for batch_test, clips in get_batch(test_set, params['CNN_BATCH_SIZE'] // stride):
            if batch_test == 0:
                continue
            if len(clips) == 0:
                break
            frames, batch_label = get_clip(clips, stride)
            batch_label_test = np.repeat(batch_label, stride, axis=0)
            batch_input_test = np.reshape(frames, (-1, params['INPUT_WIDTH'], params['INPUT_HEIGHT'], params['INPUT_CHANNEL']))
            acc_test = sess.run(accuracy, feed_dict={X:batch_input_test, Y:batch_label_test})
            arr_acc_test.append(acc_test)

        avg_acc_test = np.mean(arr_acc_test)
        print("Accuracy on test set: {}".format(avg_acc_test))
        
        if not train_mode:
            break

        epoch += 1   
        #if epoch % 1 == 0:
        if (avg_acc_test > acc_max):
            save_path = saver.save(sess, "{}/model_{}.ckpt".format(params['CNN_MODEL_SAVER_PATH'],epoch))
            print("Model saved in path: %s" % save_path)
            acc_max = avg_acc_test
            
     