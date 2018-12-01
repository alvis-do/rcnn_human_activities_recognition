
import numpy as np
import tensorflow as tf
import pandas as pd
import os, sys
import time, math
from config import params
from data_handler import get_dataset, get_batch, get_clip, split_valid_set

# Register first time tf.contrib namspace for loading graph from meta ckpt
# because ops  in this are lazily registered
dir(tf.contrib)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

n_hidden_lstm = 512
num_layers_lstm = 3
learning_rate = 0.0001
#lambda_loss_amount = 0.0015


train_mode = True if sys.argv[1] == '--train' else False

# model_path = "./graph/cnn_lstm_model.ckpt"
model_path = "./model_saver_cnn/model_1.ckpt"
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

        # Placeholder
        isTraining = g.get_tensor_by_name("LSTM_MODEL/is_training:0")
        X = g.get_tensor_by_name("CNN_MODEL/input:0") 
        init_state = g.get_tensor_by_name("LSTM_MODEL/init_state:0")
        Y = tf.placeholder(tf.float32, [None, params['N_CLASSES']], name='y_true') # (?, 4)

        # lstm output
        #labels_series = tf.unstack(Y, axis=1)
        logits_series = g.get_tensor_by_name("LSTM_MODEL/output:0") # (16, ?, 4)
        current_state = g.get_tensor_by_name("LSTM_MODEL/current_state:0") # (2, 2, ?, 512)
        # predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
        predictions_series = tf.map_fn(lambda x: tf.nn.softmax(x), logits_series)

        # lstm loss
        
        losses = tf.map_fn(lambda logits: tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y), logits_series) # (16, ?)
        # losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_series[-1], labels=Y)
        loss_op = tf.reduce_mean(losses)
        loss_op = tf.identity(loss_op, name='lstm_loss_op')

        train_op = tf.train.AdagradOptimizer(learning_rate, name='lstm_Adagrad').minimize(loss_op)
        #loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=y_pred))
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='lstm_adam')
        # train_op = optimizer.minimize(loss_op)

        # Evaluate model
        #predictions_series_stack = tf.stack(predictions_series, axis=1)
        # correct_pred = [tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1)) for prediction in predictions_series_stack]
        correct_pred = tf.map_fn(lambda x: tf.cast(tf.equal(tf.argmax(x, 1), tf.argmax(Y, 1)), tf.float32), predictions_series) # (16, ?)
        accuracy = tf.reduce_mean(correct_pred)
        accuracy = tf.identity(accuracy, name='lstm_accuracy')


        # CNN
        try:
            cnn_Y = g.get_tensor_by_name("TRAIN_CNN/y_train:0")
        except:
            cnn_Y = None

        # Scalar summary
        tf.summary.scalar("loss_op", loss_op)
        tf.summary.scalar("accuracy_train", accuracy)
        merged_summary_op = tf.summary.merge_all()


    tf_writer = tf.summary.FileWriter(params['CNN_LSTM_MODEL_SAVER_PATH'])

    # init session
    if train_mode:
        sess.run(tf.global_variables_initializer())


    # Start training
    epoch = 0
    acc_max = 0
    gl_step = 0
    while True:
        # count = 0
        _current_state = np.zeros((num_layers_lstm, 2, params['CNN_LSTM_BATCH_SIZE'], n_hidden_lstm))
        if train_mode:
            # split validation set
            train_set, valid_set = split_valid_set(train_valid_set, epoch)

            # Training on train dataset
            print("Training on {} clips...".format(len(train_set)))
            total_step = math.ceil(len(train_set) // params['CNN_LSTM_BATCH_SIZE'])
            lstm_loss_avg = []
            lstm_acc_avg = []
            for step, clips in get_batch(train_set, params['CNN_LSTM_BATCH_SIZE']):
                if step == 0:
                    continue
                if len(clips) == 0:
                    break
                #print(step)
                frames, labels = get_clip(clips)
                # print(labels)
                #print(cnn_labels)
                input_ = np.reshape(frames, (-1, params['INPUT_WIDTH'], params['INPUT_HEIGHT'], params['INPUT_CHANNEL']))

                # cnn_out = sess.run(cnn_y_pred, feed_dict={cnn_X: input_, cnn_Y: cnn_y_true})
                # lstm_input = np.reshape(cnn_out, (-1, params['N_FRAMES'], params['N_CLASSES']))
                #print(input_[0,0,0])
                #print("Y =>> {}".format(labels))
                feed_dict = None
                if cnn_Y == None:
                    feed_dict = {X: input_, Y: labels, isTraining: True, init_state: _current_state}
                else:
                    cnn_y_true = np.repeat(labels, params['N_FRAMES'], axis=0)
                    feed_dict = {X: input_, Y: labels, isTraining: True, init_state: _current_state, cnn_Y: cnn_y_true}

                _, lstm_loss, lstm_acc, summ, _current_state, _predictions_series = \
                    sess.run([train_op, loss_op, accuracy, merged_summary_op, current_state, predictions_series], feed_dict=feed_dict)
                
                # print("---Batch {}: loss {} --- acc {}".format(step, lstm_loss, lstm_acc))
                # print(np.array(_predictions_series)) # (16, batch_size, 4)
                # print(np.array(_current_state).shape) # (num_layers, 2, batch_size, 512)

                lstm_loss_avg.append(lstm_loss)
                lstm_acc_avg.append(lstm_acc) 

                tf_writer.add_summary(summ, gl_step)
                gl_step += 1;
                # break
                # count += 1
                # if (count > 4):
                #     break
                # break
            print("""Epoch {}: lstm_loss {}; lstm_acc {}""".format(epoch, np.mean(lstm_loss_avg), np.mean(lstm_acc_avg)))
        #         break
        # epoch += 1
        # continue
        # Testing on  dataset
        tmp_set = valid_set if train_mode else test_set
        #print("Test lstm net ...")
        arr_acc_test = []
        for step, clips in get_batch(tmp_set, params['CNN_LSTM_BATCH_SIZE']):
            if (step == 0):
                continue
            if len(clips) == 0:
                break
        
            valid_inputs, valid_labels = get_clip(clips)
            input_ = np.reshape(valid_inputs, (-1, params['INPUT_WIDTH'], params['INPUT_HEIGHT'], params['INPUT_CHANNEL']))
          
            acc, _predictions_series = sess.run([accuracy, predictions_series], feed_dict={X: input_, Y: valid_labels, isTraining: False, init_state: _current_state})
            arr_acc_test.append(acc)
            # print(valid_labels)
            # print(_predictions_series)
            # break
        avg_acc_test = np.mean(arr_acc_test)
        print("Acc on validation {}".format(avg_acc_test))

        if not train_mode:
            break

        ####
        epoch += 1
        if epoch % 5 == 0:
        # if (avg_acc_test > acc_max):
            save_path = saver.save(sess, "{}/model_{}.ckpt".format(params['CNN_LSTM_MODEL_SAVER_PATH'], epoch))
            print("Model saved in path: %s" % save_path)
            acc_max = avg_acc_test
            

