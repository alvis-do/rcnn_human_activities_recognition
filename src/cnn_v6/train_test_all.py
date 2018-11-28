
import numpy as np
import tensorflow as tf
import pandas as pd
import os, sys
import time
from config import params
from data_handler import get_dataset, get_batch, get_clip, split_valid_set

# Register first time tf.contrib namspace for loading graph from meta ckpt
# because ops  in this are lazily registered
dir(tf.contrib)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


learning_rate = 0.0001
lambda_loss_amount = 0.0015


train_mode = True if sys.argv[1] == '--train' else False

model_path = "./graph/cnn_lstm_model.ckpt"
# model_path = "./model_saver_cnn_lstm/model_4.ckpt"
saver = tf.train.import_meta_graph("{}.meta".format(model_path))



# Get data set
train_valid_set, test_set = get_dataset()


# setup config for session
config = tf.ConfigProto(log_device_placement=False)
# config.gpu_options.allow_growth = True
# config.gpu_options.per_proces_gpu_memory_fraction = 1.



with tf.Session(config=config) as sess:
    saver.restore(sess, model_path)
    g = sess.graph
    with g.name_scope("TRAIN_LSTM"):

        X = g.get_tensor_by_name("CNN_MODEL/input:0")

        isTraining = g.get_tensor_by_name("LSTM_MODEL/is_training:0")
        Y = tf.placeholder(tf.float32, [None, params['N_CLASSES']], name='y_true')
        y_pred = g.get_tensor_by_name("LSTM_MODEL/output:0")
        y_pred = y_pred[:,params['N_FRAMES'] - 1, :]
        #y_pred = tf.Print(y_pred, [y_pred], summarize=16, message='y_pred =>> ')
        
        # LSTM
        #l2 = lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        l2 = 0
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=y_pred)) + l2
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='lstm_adam')
        train_op = optimizer.minimize(loss_op)
        # Evaluate model
        correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # CNN
        # cnn_Y = tf.placeholder(tf.float32, [None, params['N_CLASSES']], name='cnn_y_true')
        # cnn_y_pred = g.get_tensor_by_name("CNN_MODEL/out_class:0")
        # #cnn_y_pred = tf.Print(cnn_y_pred, [cnn_y_pred], summarize=16)
        # cnn_loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=cnn_y_pred, labels=cnn_Y))
        # cnn_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='cnn_adam')
        # cnn_train_op = cnn_optimizer.minimize(cnn_loss_op)
        # # Evaluate model
        # cnn_correct_pred = tf.equal(tf.argmax(tf.nn.softmax(cnn_y_pred), 1), tf.argmax(cnn_Y, 1))
        # cnn_accuracy = tf.reduce_mean(tf.cast(cnn_correct_pred, tf.float32))

        tf.summary.scalar("loss_op", loss_op)
        tf.summary.scalar("accuracy", accuracy)
        merged_summary_op = tf.summary.merge_all()


    tf_writer = tf.summary.FileWriter(params['CNN_LSTM_MODEL_SAVER_PATH'])

    # init session
    if train_mode:
        sess.run(tf.global_variables_initializer())


    # Start training
    epoch = 0
    while epoch < 50:
        if train_mode:
            # split validation set
            train_set, valid_set = split_valid_set(train_valid_set, epoch)
            
            # Training on train dataset
            print("Training on {} clips...".format(len(train_set)))
            # count = 0
            for step, clips in get_batch(train_set, params['CNN_LSTM_BATCH_SIZE']):
                if step == 0:
                    continue
                if len(clips) == 0:
                    break

                frames, labels = get_clip(clips)

                #print(cnn_labels)
                input_ = np.reshape(frames, (-1, params['INPUT_WIDTH'], params['INPUT_HEIGHT'], params['INPUT_CHANNEL']))
                #print("Y =>> {}".format(labels))
                _, lstm_loss, lstm_acc, summ = sess.run([train_op, loss_op, accuracy, merged_summary_op], feed_dict={X: input_, Y: labels, isTraining: True})

                tf_writer.add_summary(summ)

                print("""---Batch {}: lstm_loss {}
                    lstm_acc {}""".format(step - 1, lstm_loss, lstm_acc))
                # count += 1
                # if (count > 4):
                #     break

        # Testing on  dataset
        tmp_set = valid_set if train_mode else test_set
        print("Test lstm net ...")
        acc_avg = []
        for step, clips in get_batch(tmp_set, params['CNN_LSTM_BATCH_SIZE']):
            if (step == 0):
                continue
            if len(clips) == 0:
                break
        
            valid_inputs, valid_labels = get_clip(clips)
            input_ = np.reshape(valid_inputs, (-1, params['INPUT_WIDTH'], params['INPUT_HEIGHT'], params['INPUT_CHANNEL']))
          
            acc = sess.run(accuracy, feed_dict={X: input_, Y: valid_labels, isTraining: False})
            acc_avg.append(acc)
            #print("Batch {}".format(step))
        print("Epoch {} : Acc on validation {}".format(epoch, np.mean(acc_avg)))

        if not train_mode:
            break

        ####
        epoch += 1
        if epoch % 5 == 0:
            save_path = saver.save(sess, "{}/model_{}.ckpt".format(params['CNN_LSTM_MODEL_SAVER_PATH'], epoch))
            print("Model saved in path: %s" % save_path)


