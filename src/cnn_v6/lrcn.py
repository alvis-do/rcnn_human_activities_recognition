'''
$ python3 lrcn.py 0 /floyd/input/my_cnn_model/model_10.ckpt
'''


import argparse
import numpy as np
import tensorflow as tf
import os, sys
from config import params
from create_graph import create_cnn_net_graph, create_lstm_net_graph
from data_handler import get_dataset, get_batch, get_clip, split_valid_set


# validate parameters
parser = argparse.ArgumentParser(description='Train and test the lrcn model')
parser.add_argument('mode', metavar='mode', type=int, nargs=1, choices=[0,1],
                    help='0 for train mode, 1 for test mode')
parser.add_argument('cnn_model', metavar='cnn_model', nargs=1,
                    help='The path of trained cnn model.')
parser.add_argument('-m', '--modelpath', nargs=1, 
                    help='The path of trained lrcn model.')
args = parser.parse_args()



# Register first time tf.contrib namspace for loading graph from meta ckpt
# because ops  in this are lazily registered
dir(tf.contrib)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


learning_rate = 0.0001
display_step = 5
epochs = 50
epoch_save = 5
batch_size = 16 # the number of clips



 # Get data set
train_set, test_set = get_dataset()
train_set, valid_set = split_valid_set(train_set)

train_len = len(train_set)
test_len = len(test_set)
valid_len = len(valid_set)



# setup config for session
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.



# create the cnn graph
lrcn_graph = None
saver = None
first_train = True

if args.mode[0] == 0 and args.modelpath == None: # (first train mode)
    saver = tf.train.import_meta_graph("{}.meta".format(args.cnn_model[0]))
else:
    first_train = False
    saver = tf.train.import_meta_graph("{}.meta".format(args.modelpath[0]))



with tf.Session(graph=lrcn_graph, config=config) as sess:
    
    # restore lrcn model if existed
    if first_train:
        saver.restore(sess, args.cnn_model[0])
    else:
        saver.restore(sess, args.modelpath[0])


    # get graph of current session
    g = sess.graph

    # add lstm graph if not existed
    if first_train:
        g = create_lstm_net_graph(g)

    with g.name_scope("LSTM_OP"):

        # Placeholder
        X = g.get_tensor_by_name("LSTM_MODEL/input:0") 
        init_state = g.get_tensor_by_name("LSTM_MODEL/init_state:0")

        # output
        logits = g.get_tensor_by_name("LSTM_MODEL/output:0") # (16, ?, 4)
        current_state = g.get_tensor_by_name("LSTM_MODEL/current_state:0") # (2, 2, ?, 512)
        
        if first_train:
            # y true placeholder
            Y = tf.placeholder(tf.float32, [None, params['N_CLASSES']], name='y_true') # Placeholder
            
            # lstm predict
            prediction = tf.nn.softmax(logits)

            # lstm loss many-to-one
            lstm_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y) 
            loss_op = tf.reduce_mean(lstm_losses) 
            train_op = tf.train.AdamOptimizer(learning_rate, name='lstm_Adam').minimize(loss_op)
            
            # Evaluate model
            y_hat = tf.argmax(prediction, 1)
            correct_pred = tf.cast(tf.equal(y_hat, tf.argmax(Y, 1)), tf.float32)
            accuracy = tf.reduce_mean(correct_pred)

            # Add to collection
            tf.add_to_collection("lstm_predict", prediction)
            tf.add_to_collection("lstm_loss_op", loss_op)
            tf.add_to_collection("lstm_train_op", train_op)
            tf.add_to_collection("lstm_y_hat", y_hat)
            tf.add_to_collection("lstm_acc", accuracy)

        else:
             # Get variables from collection
            Y = g.get_tensor_by_name("LSTM_OP/y_true:0") # Placeholder
            prediction = tf.get_collection("lstm_predict")[0]
            loss_op = tf.get_collection("lstm_loss_op")[0]
            train_op = tf.get_collection("lstm_train_op")[0]
            y_hat = tf.get_collection("lstm_y_hat")[0]
            accuracy = tf.get_collection("lstm_acc")[0]
       

        # Get ops for cnn graph
        cnn_X = g.get_tensor_by_name("CNN_MODEL/input:0") # Placeholder
        cnn_Y = g.get_tensor_by_name("CNN_OP/y_true:0") # Placeholder
        cnn_features = g.get_tensor_by_name("CNN_MODEL/out_features:0")
        cnn_loss_op = tf.get_collection("cnn_loss_op")[0] 
        cnn_acc = tf.get_collection("cnn_acc")[0]


        # Make monitor from tensor board
        # Make monitor from tensor board
        summary_train = tf.summary.merge([tf.summary.scalar("loss_op", loss_op), tf.summary.scalar("accuracy", accuracy)])
        summary_val = tf.summary.merge([tf.summary.scalar("val_loss_op", loss_op), tf.summary.scalar("val_accuracy", accuracy)])
        



    tf_writer = tf.summary.FileWriter(params['CNN_LSTM_MODEL_SAVER_PATH'])

    # init session
    sess.run(tf.global_variables_initializer())

    # override the available global variables in the cnn model if existed
    if first_train:
        saver.restore(sess, args.cnn_model[0])
    else:
        saver.restore(sess, args.modelpath[0])


    def run(epoch=0):
        loss = 0
        acc = 0
        test_loss = 0
        test_acc = 0
        eval_set = test_set

        if args.mode[0] == 0: # train mode

            eval_set = valid_set
            total_batch = (train_len // batch_size) + 1

            tmp_loss = 0
            tmp_acc = 0
            cnn_tmp_loss = 0
            cnn_tmp_acc = 0

            for batch, clips in get_batch(train_set, batch_size):
                if batch == 0:
                    continue
                if batch >= total_batch or len(clips)==0:
                    break

                # init zero state for lstm
                init_state_zeros = g.get_tensor_by_name("LSTM_MODEL/init_state_zeros:0") 
                _current_state = np.repeat(init_state_zeros.eval(), batch_size, axis=2)


                # get frames from clips - the a small equence in a video
                frames, labels = get_clip(clips)


                # Reshape input and output
                _cnn_inputs = np.reshape(frames, (-1, params['INPUT_WIDTH'], params['INPUT_HEIGHT'], params['INPUT_CHANNEL']))
                _cnn_labels = np.repeat(labels, params['N_FRAMES'], axis=0)

            
                # Retrieve/Predict the features from the CNN net
                _lstm_input = np.empty([1, params['N_FRAMES'], params['N_FEATURES']], dtype=np.float32)
                _cnn_feed_dict = {cnn_X: _cnn_inputs, cnn_Y: _cnn_labels, X: _lstm_input, Y: labels, init_state: _current_state}
                _cnn_loss, _cnn_acc, _cnn_features = sess.run([cnn_loss_op, cnn_acc, cnn_features], feed_dict=_cnn_feed_dict)


                # feed placeholder and train LSTM model
                _lstm_input = np.reshape(_cnn_features, (-1, params['N_FRAMES'], params['N_FEATURES']))
                _lstm_feed_dict = {cnn_X: _cnn_inputs, cnn_Y: _cnn_labels, X: _lstm_input, Y: labels, init_state: _current_state}
                _loss, _acc, _current_state, _summ, _ = sess.run([loss_op, accuracy, current_state, summary_train, train_op], feed_dict=_lstm_feed_dict)


                # Sum loss and acc
                tmp_loss += _loss
                tmp_acc += _acc
                cnn_tmp_loss += _cnn_loss
                cnn_tmp_acc += _cnn_acc
                loss += _loss
                acc += _acc


                # summary data to tensor board
                tf_writer.add_summary(_summ, epoch * total_batch + batch)


                # Display if neccessary
                if batch % display_step == 0:
                    print("Batch {}/{}: Loss {} - Acc {} - CNN_loss {} - CNN_acc {}".format(batch, total_batch, tmp_loss/display_step, tmp_acc/display_step, cnn_tmp_loss/display_step, cnn_tmp_acc/display_step))
                    tmp_loss = 0
                    tmp_acc = 0
                    cnn_tmp_loss = 0
                    cnn_tmp_acc = 0
                

            # compute the average of loss and acc
            loss /= batch
            acc /= batch



        # Valid or test model
        eval_len = len(eval_set)
        total_batch = (eval_len // batch_size) + 1
        for batch, clips in get_batch(eval_set, batch_size):
            if batch == 0:
                continue
            if batch >= total_batch or len(clips)==0:
                break


            # init zero state for lstm
            init_state_zeros = g.get_tensor_by_name("LSTM_MODEL/init_state_zeros:0")
            _current_state = np.repeat(init_state_zeros.eval(), batch_size, axis=2)


            # get frames from clips - the a small equence in a video
            frames, labels = get_clip(clips)


            # Reshape input and output
            _cnn_inputs = np.reshape(frames, (-1, params['INPUT_WIDTH'], params['INPUT_HEIGHT'], params['INPUT_CHANNEL']))
            _cnn_labels = np.repeat(labels, params['N_FRAMES'], axis=0)


            # Retrieve/Predict the features from the CNN net
            _lstm_input = np.empty([1, params['N_FRAMES'], params['N_FEATURES']], dtype=np.float32)
            _cnn_feed_dict = {cnn_X: _cnn_inputs, cnn_Y: _cnn_labels, X: _lstm_input, Y: labels, init_state: _current_state}
            _cnn_loss, _cnn_acc, _cnn_features = sess.run([cnn_loss_op, cnn_acc, cnn_features], feed_dict=_cnn_feed_dict)


            # feed placeholder and train LSTM model
            _lstm_input = np.reshape(_cnn_features, (-1, params['N_FRAMES'], params['N_FEATURES']))
            _lstm_feed_dict = {cnn_X: _cnn_inputs, cnn_Y: _cnn_labels, X: _lstm_input, Y: labels, init_state: _current_state}
            _loss, _acc, _summ = sess.run([loss_op, accuracy, summary_val], feed_dict=_lstm_feed_dict)


            # Sum loss and acc
            test_loss += _loss
            test_acc += _acc

            
            if args.mode[0] == 0: # train mode
                # summary data to tensor board
                tf_writer.add_summary(_summ, epoch * total_batch + batch)
            

        # compute the average of loss and acc
        test_loss /= batch
        test_acc /= batch


        # return
        if args.mode[0] == 0: # train mode 
            return loss, acc, test_loss, test_acc
        else:
            return test_loss, test_acc

    # End function



    # Training or testing on dataset
    if args.mode[0] == 0: # train mode
        print("\nTraining on {} samples, Validate on {} samples".format(train_len, valid_len))
        for epoch in range(epochs):
            print("\n---------------------------------------------")
            print("\n--- Epoch {}/{} ---".format(epoch, epochs))

            loss, acc, val_loss, val_acc = run(epoch)

            print("\nThe final result: Loss {} - Acc {} - Val_loss {} - Val_acc {}".format(loss, acc, val_loss, val_acc))

            # Save model
            if (epoch + 1) % epoch_save == 0:
                save_path = saver.save(sess, "{}/model{}_{}.ckpt".format(params['CNN_LSTM_MODEL_SAVER_PATH'], "_resume" if not first_train else "", epoch + 1))
                print("Model saved in path: %s" % save_path)

    else:
        print("\nTesting on {} sample(s)".format(test_len))

        loss, acc = run()

        print("\nThe final result: Loss {} - Acc {}".format(loss, acc))


# End session =======================================================================================
