
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
n_hidden_lstm_1 = 256
num_layers_lstm = 3
num_layers_lstm_1 = 2
learning_rate = 0.0001
#lambda_loss_amount = 0.0015

train_mode = True if sys.argv[1] == '--train' else False

#model_path = "./graph/cnn_lstm_model.ckpt"
model_path = "/home/vudhk/Desktop/model_saver_cnn/model_18.ckpt"
saver = tf.train.import_meta_graph("{}.meta".format(model_path))



# Get data set
train_valid_set, test_set = get_dataset()


# setup config for session
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.


with tf.Session(config=config) as sess:
    saver.restore(sess, model_path)
    g = sess.graph

    with g.name_scope("LSTM_MODEL_1"):

            # Create LSTM cell
            n_hidden = 256
            num_layers = 2


            def lstm_cpu(X, init_state):
                with g.name_scope("lstm_cpu"):
                    X_seq = tf.unstack(X, axis=1)

                    cells = [tf.contrib.rnn.LSTMBlockCell(n_hidden, forget_bias=1.0) for i in range(num_layers)]
                    multi_cell = tf.contrib.rnn.MultiRNNCell(cells)
                    output, state = tf.contrib.rnn.static_rnn(multi_cell, X_seq, initial_state=init_state, dtype=tf.float32)
                    #output = tf.stack(output, axis=1)
                    #print(state)

                    return output, state

            
                
            # Weight + bias
            weights = {
                'hidden': tf.Variable(tf.random_normal([params['N_CLASSES'], n_hidden]), name='lstm_wh'),
                'out': tf.Variable(tf.random_normal([n_hidden, params['N_CLASSES']], mean=1.0), name='lstm_wo')
            }
            biases = {
                'hidden': tf.Variable(tf.random_normal([n_hidden]), name='lstm_bh'),
                'out': tf.Variable(tf.random_normal([params['N_CLASSES']]), name='lstm_bo')
            }


            init_state = tf.placeholder(tf.float32, [num_layers, 2, params['CNN_LSTM_BATCH_SIZE'], n_hidden], name='init_state')
            state_per_layer_list = tf.unstack(init_state, axis=0)
            rnn_tuple_state = tuple(
                [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
                 for idx in range(num_layers)]
            )

            # X = tf.placeholder(tf.float32, [None, params['N_CLASSES']], name='input')
            X = g.get_tensor_by_name("CNN_MODEL/out_class:0")
            # X = tf.Print(X, [X[0]], summarize=4)
            #X_ = tf.Print(X, [X[0]], summarize=4, message='X =>> ')

            # ReLU activation, thanks to Yu Zhao for adding this improvement here:
            X_ = tf.nn.relu(tf.matmul(X, weights['hidden']) + biases['hidden']) # (?, 256)
            X_ = tf.reshape(X_, [-1, params['N_FRAMES'], n_hidden])

            X_ = tf.identity(X_, name='input') # (?, ,16, 4)

            # Creates a recurrent neural network specified by RNNCell cell.
            states_series, current_state = lstm_cpu(X_, rnn_tuple_state)
            
            out = [tf.matmul(state, weights['out']) + biases['out'] for state in states_series]
            out = tf.identity(out, name='output') # (16, ?, 4)

            curr_state = tf.identity(current_state, name='current_state') # (2, 2, ?, 512)


    with g.name_scope("TRAIN_LSTM"):

        # Placeholder
        isTraining = g.get_tensor_by_name("LSTM_MODEL/is_training:0")
        X = g.get_tensor_by_name("CNN_MODEL/input:0") 
        # X_lstm = g.get_tensor_by_name("LSTM_MODEL_1/input:0") 
        init_state = g.get_tensor_by_name("LSTM_MODEL/init_state:0")
        init_state_1 = g.get_tensor_by_name("LSTM_MODEL_1/init_state:0")
        Y = tf.placeholder(tf.float32, [None, params['N_CLASSES']], name='y_true') # (?, 4)

        # lstm output
        logits_series = g.get_tensor_by_name("LSTM_MODEL_1/output:0") # (16, ?, 4)
        current_state = g.get_tensor_by_name("LSTM_MODEL_1/current_state:0") # (2, 2, ?, 512)
        predictions_series = tf.map_fn(lambda x: tf.nn.softmax(x), logits_series)

        # lstm loss
        # many-to-many
        # losses = tf.map_fn(lambda logits: tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y), logits_series) # (16, ?)
        # many-to-one
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_series[-1], labels=Y)
        loss_op = tf.reduce_mean(losses)
        loss_op = tf.identity(loss_op, name='lstm_loss_op')

        train_op = tf.train.AdamOptimizer(learning_rate, name='lstm_Adam').minimize(loss_op)

        # Evaluate model
        # many-to-many
        # correct_pred = tf.map_fn(lambda x: tf.cast(tf.equal(tf.argmax(x, 1), tf.argmax(Y, 1)), tf.float32), predictions_series) # (16, ?)
        # many-to-one
        correct_pred = tf.cast(tf.equal(tf.argmax(predictions_series[-1], 1), tf.argmax(Y, 1)), tf.float32)
        accuracy = tf.reduce_mean(correct_pred)
        accuracy = tf.identity(accuracy, name='lstm_accuracy')


        # CNN
        try:
            cnn_Y = g.get_tensor_by_name("TRAIN_CNN/y_train:0")
        except:
            cnn_Y = None


        # log variable
        cnn_out = g.get_tensor_by_name("CNN_MODEL/out_class:0")
        wh_lstm_1 = g.get_tensor_by_name("LSTM_MODEL_1/lstm_wh:0")
        wo_lstm_1 = g.get_tensor_by_name("LSTM_MODEL_1/lstm_wo:0")
        wc_cnn = g.get_tensor_by_name("CNN_MODEL/wc1:0")
        wo_cnn = g.get_tensor_by_name("CNN_MODEL/w_out:0")


        # Scalar summary
        tf.summary.scalar("loss_op", loss_op)
        tf.summary.scalar("accuracy_train", accuracy)
        merged_summary_op = tf.summary.merge_all()


    tf_writer = tf.summary.FileWriter(params['CNN_LSTM_MODEL_SAVER_PATH'])

    # init session
    if train_mode:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)


    # Start training
    epoch = 0
    acc_max = 0
    gl_step = 0
    # split validation set
    train_set, valid_set = split_valid_set(train_valid_set, epoch)
    while True:
        # count = 0
        if train_mode:

            # Training on train dataset
            print("Training on {} clips...".format(len(train_set)))
            total_step = math.ceil(len(train_set) // params['CNN_LSTM_BATCH_SIZE'])
            lstm_loss_avg = []
            lstm_acc_avg = []
            for step, clips in get_batch(train_set, params['CNN_LSTM_BATCH_SIZE']):
                if step == 0:
                    continue
                #if len(clips) == 0:
                if step >= total_step or len(clips)==0:
                    break

                _current_state = np.zeros((num_layers_lstm, 2, params['CNN_LSTM_BATCH_SIZE'], n_hidden_lstm))
                _current_state_1 = np.zeros((num_layers_lstm_1, 2, params['CNN_LSTM_BATCH_SIZE'], n_hidden_lstm_1))
                
                frames, labels = get_clip(clips)

                input_ = np.reshape(frames, (-1, params['INPUT_WIDTH'], params['INPUT_HEIGHT'], params['INPUT_CHANNEL']))
                feed_dict = {}


                # _cnn_out = np.ones((params['CNN_LSTM_BATCH_SIZE'] * params['N_FRAMES'], params['N_CLASSES']))
                # if cnn_Y == None:
                #     feed_dict = {X: input_, X_lstm: _cnn_out, Y: labels, isTraining: True, init_state: _current_state, init_state_1: _current_state_1}
                # else:
                #     cnn_y_true = np.repeat(labels, params['N_FRAMES'], axis=0)
                #     feed_dict = {X: input_, X_lstm: _cnn_out, Y: labels, isTraining: True, init_state: _current_state, init_state_1: _current_state_1, cnn_Y: cnn_y_true}

                # # predict cnn
                # _cnn_out = sess.run(cnn_out, feed_dict=feed_dict)

                if cnn_Y == None:
                    feed_dict = {X: input_,  Y: labels, isTraining: True, init_state: _current_state, init_state_1: _current_state_1}
                else:
                    cnn_y_true = np.repeat(labels, params['N_FRAMES'], axis=0)
                    feed_dict = {X: input_,  Y: labels, isTraining: True, init_state: _current_state, init_state_1: _current_state_1, cnn_Y: cnn_y_true}


                _, lstm_loss, lstm_acc, summ, _current_state_1, _cnn_out = \
                    sess.run([train_op, loss_op, accuracy, merged_summary_op, current_state, cnn_out], feed_dict=feed_dict)
                

                # Log variable
                # w1, w2, w3, w4 = sess.run([wh_lstm_1, wo_lstm_1, wc_cnn, wo_cnn], feed_dict=feed_dict)
                # print("w1 {} --- w2 {} --- w3 {} --- w4 {}".format(np.sum(w1),np.sum(w2),np.sum(w3),np.sum(w4)))

                cnn_acc = np.mean(np.equal(np.argmax(_cnn_out, 1), np.argmax(np.repeat(labels, params['N_FRAMES'], axis=0), 1)).astype(float))
                print("---Batch {}: loss {} --- acc {} --- cnn_acc {} \n".format(step, lstm_loss, lstm_acc, cnn_acc))
                # print(np.array(_predictions_series)) # (16, batch_size, 4)
                # print(np.array(_current_state).shape) # (num_layers, 2, batch_size, 512)

                lstm_loss_avg.append(lstm_loss)
                lstm_acc_avg.append(lstm_acc) 

                tf_writer.add_summary(summ, gl_step)
                gl_step += 1

            print("""Epoch {}: lstm_loss {}; lstm_acc {}""".format(epoch, np.mean(lstm_loss_avg), np.mean(lstm_acc_avg)))

        # Testing on  dataset
        try:
            tmp_set = valid_set if train_mode else test_set
            #print("Test lstm net ...")
            arr_acc_test = []
            total_step = math.ceil(len(tmp_set) // params['CNN_LSTM_BATCH_SIZE'])
            for step, clips in get_batch(tmp_set, params['CNN_LSTM_BATCH_SIZE']):
                if (step == 0):
                    continue
                #if len(clips) == 0:
                if step >= total_step or len(clips)==0:
                    break
            
                _current_state = np.zeros((num_layers_lstm, 2, params['CNN_LSTM_BATCH_SIZE'], n_hidden_lstm))
                _current_state_1 = np.zeros((num_layers_lstm_1, 2, params['CNN_LSTM_BATCH_SIZE'], n_hidden_lstm_1))

                valid_inputs, valid_labels = get_clip(clips)
                input_ = np.reshape(valid_inputs, (-1, params['INPUT_WIDTH'], params['INPUT_HEIGHT'], params['INPUT_CHANNEL']))
              
                acc, _predictions_series = sess.run([accuracy, predictions_series], 
                    feed_dict={X: input_, Y: valid_labels, isTraining: False, init_state: _current_state, init_state_1: _current_state_1})

                arr_acc_test.append(acc)

            avg_acc_test = np.mean(arr_acc_test)
            print("Acc on validation {}".format(avg_acc_test))
        except:
            print("Test Error: {}".format(sys.exc_info()))

        if not train_mode:
            break

        ####
        epoch += 1
        if epoch % 1 == 0:
        # if (avg_acc_test > acc_max):
            save_path = saver.save(sess, "{}/model_{}.ckpt".format(params['CNN_LSTM_MODEL_SAVER_PATH'], epoch))
            print("Model saved in path: %s" % save_path)
            acc_max = avg_acc_test
            

