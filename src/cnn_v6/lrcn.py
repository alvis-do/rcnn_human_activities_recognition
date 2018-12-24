'''
$ python3 lrcn.py 0 /floyd/input/my_cnn_model/model_10.ckpt
'''


import argparse
import numpy as np
import tensorflow as tf
import os
import time
from config import params
from create_graph import create_lstm_net_graph
from data_handler import get_dataset, get_batch, get_clip, split_valid_set, get_video_test, get_frames_from_video


# validate parameters
parser = argparse.ArgumentParser(description='Train and test the lrcn model')
parser.add_argument('mode', metavar='mode', type=int, nargs=1, choices=[0,1],
                    help='0 for train mode, 1 for test mode')
parser.add_argument('-c', '--cnn', metavar='cnn_model', nargs=1,
                    help='The path of trained cnn model.')
parser.add_argument('-l', '--lrcn', metavar='lrcn_model', nargs=1, 
                    help='The path of trained lrcn model.')
args = parser.parse_args()



# Register first time tf.contrib namspace for loading graph from meta ckpt
# because ops  in this are lazily registered
dir(tf.contrib)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


learning_rate = 0.0001
display_step = 5
epochs = 10
epoch_save = 2
batch_size = 16 # the number of clips


# Get data set
train_set, test_set = get_dataset()
test_len = len(test_set)
valid_set = []
valid_len = 0
if params['USE_VALIDATION_DATASET']:
    train_set, valid_set = split_valid_set(train_set)
    valid_len = len(valid_set)
train_len = len(train_set)


# setup config for session
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.



# create the cnn graph
lrcn_graph = None
saver = None
first_train = True

if args.mode[0] == 0 and args.lrcn == None: # (first train mode)
    saver = tf.train.import_meta_graph("{}.meta".format(args.cnn[0]))
else:
    first_train = False
    saver = tf.train.import_meta_graph("{}.meta".format(args.lrcn[0]))



with tf.Session(graph=lrcn_graph, config=config) as sess:
    
    # restore lrcn model if existed
    if first_train:
        saver.restore(sess, args.cnn[0])
    else:
        saver.restore(sess, args.lrcn[0])


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
        cnn_pred = tf.get_collection("cnn_predict")[0]
        cnn_loss_op = tf.get_collection("cnn_loss_op")[0] 
        cnn_acc = tf.get_collection("cnn_acc")[0]


        # Make monitor from tensor board
        summary_train = tf.summary.merge([tf.summary.scalar("loss_op", loss_op), tf.summary.scalar("accuracy", accuracy)])
        summary_val = tf.summary.merge([tf.summary.scalar("val_loss_op", loss_op), tf.summary.scalar("val_accuracy", accuracy)])
        



    tf_writer = tf.summary.FileWriter(params['CNN_LSTM_MODEL_SAVER_PATH'], g)

    # init session
    sess.run(tf.global_variables_initializer())

    # override the available global variables in the cnn model if existed
    if first_train:
        saver.restore(sess, args.cnn[0])
    else:
        saver.restore(sess, args.lrcn[0])

    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)


    def run(epoch=0):
        loss = 0
        acc = 0
        test_loss = 0
        test_acc = 0

        def init_state_lstm(batch_size):
            init_state_zeros = g.get_tensor_by_name("LSTM_MODEL/init_state_zeros:0") 
            return np.repeat(init_state_zeros.eval(), batch_size, axis=2)

        def reshape_data(frames, labels):
            # Reshape input and output
            _cnn_inputs = np.reshape(frames, (-1, params['INPUT_WIDTH'], params['INPUT_HEIGHT'], params['INPUT_CHANNEL']))
            _cnn_labels = np.repeat(labels, params['N_FRAMES'], axis=0)
            return _cnn_inputs, _cnn_labels

        def extract_feature(frames, labels, cur_state, fetches):
            _cnn_inputs, _cnn_labels = reshape_data(frames, labels)
            # Retrieve/Predict the features from the CNN net
            _lstm_input = np.empty([1, params['N_FRAMES'], params['N_FEATURES']], dtype=np.float32)
            _cnn_feed_dict = {cnn_X: _cnn_inputs, cnn_Y: _cnn_labels, X: _lstm_input, Y: labels, init_state: cur_state}
            return sess.run(fetches, feed_dict=_cnn_feed_dict)

        def run_session(frames, labels, features, cur_state, fetches):
            _cnn_inputs, _cnn_labels = reshape_data(frames, labels)
            # feed placeholder and train LSTM model
            _lstm_input = np.reshape(features, (-1, params['N_FRAMES'], params['N_FEATURES']))
            _lstm_feed_dict = {cnn_X: _cnn_inputs, cnn_Y: _cnn_labels, X: _lstm_input, Y: labels, init_state: cur_state}
            return sess.run(fetches, feed_dict=_lstm_feed_dict)




        if args.mode[0] == 0: # train mode

            total_batch = (train_len // batch_size) + 1

            tmp_loss = 0
            tmp_acc = 0
            cnn_tmp_loss = 0
            cnn_tmp_acc = 0

            for batch, clips in get_batch(train_set, batch_size):
                if batch == 0:
                    continue
                if batch > total_batch or len(clips)==0:
                    break

                frames, labels = get_clip(clips)
                _current_state = init_state_lstm(len(clips))
                _cnn_loss, _cnn_acc, _cnn_features = extract_feature(frames, labels, _current_state, [cnn_loss_op, cnn_acc, cnn_features])
                _loss, _acc, _current_state, _summ, _ = run_session(frames, labels, _cnn_features, _current_state, [loss_op, accuracy, current_state, summary_train, train_op])


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


            # Valid model
            if params['USE_VALIDATION_DATASET']:
                total_batch = (valid_len // batch_size) + 1
                for batch, clips in get_batch(valid_set, batch_size):
                    if batch == 0:
                        continue
                    if batch > total_batch or len(clips)==0:
                        break

                    frames, labels = get_clip(clips)
                    _current_state = init_state_lstm(len(clips))
                    _cnn_features = extract_feature(frames, labels, _current_state, cnn_features)
                    _loss, _acc, _summ = run_session(frames, labels, _cnn_features, _current_state, [loss_op, accuracy, summary_val])

                    # Sum loss and acc
                    test_loss += _loss
                    test_acc += _acc

                    tf_writer.add_summary(_summ, epoch * total_batch + batch)
                       

                # compute the average of loss and acc
                test_loss /= batch
                test_acc /= batch

            return loss, acc, test_loss, test_acc

        else: # Test mode
            lbs = []
            preds = []
            video_records = get_video_test()
            y_true = []
            y_score = []
            for path, label in video_records:
                _vote_label = [0]*params['N_CLASSES']
                _t = time.process_time()
                for frames in get_frames_from_video(path):
                    if frames == None:
                        continue
                    if len(frames) == 0:
                        break
                        
                    _current_state = init_state_lstm(1)
                    _cnn_features = extract_feature([frames], [label], _current_state, cnn_features)
                    _pred_label = run_session([frames], [label], [_cnn_features], _current_state, prediction)
                    _idx_label = np.argmax(_pred_label[0])

                    # vote with own weight
                    _vote_label[_idx_label] += _pred_label[0][_idx_label]

                if frames == None or len(frames) == 0:
                    continue

                elapsed_time = time.process_time() - _t
                # sess.run([acc_update_op, prec_update_op, recall_update_op], feed_dict={metric_label: label, metric_pred: _vote_label})

                print("{} --- {} --- {}(s)".format(path.strip(), params['CLASSES'][np.argmax(_vote_label)], elapsed_time))
                lbs.append(np.argmax(label))
                preds.append(np.argmax(_vote_label))

                y_true.append(label)
                y_score.append(_vote_label)

            # compute confuse matrix
            _confuse_matrix = tf.confusion_matrix(lbs, preds).eval()

            # save all y_score
            y_dict = {'y_true': y_true, 'y_score': y_score}
            np.save('y_pred.npy', y_dict)
            
            # compute accuracy, precision, recall
            with tf.Session() as metric_sess:
                metric_label = tf.placeholder(tf.float32, [None])            
                metric_pred = tf.placeholder(tf.float32, [None])
                metric_accuracy, acc_update_op = tf.metrics.accuracy(metric_label, metric_pred)
                metric_precision, prec_update_op = tf.metrics.precision(metric_label, metric_pred)
                metric_recall, recall_update_op = tf.metrics.recall(metric_label, metric_pred)   

                metric_sess.run(tf.local_variables_initializer())
                metric_sess.run([acc_update_op, prec_update_op, recall_update_op], feed_dict={metric_label: lbs, metric_pred: preds})
                _accuracy, _precision, _recall = metric_sess.run([metric_accuracy, metric_precision, metric_recall])

            return _confuse_matrix, _accuracy, _precision, _recall

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
        print("\nTesting on {} video(s)".format(len(get_video_test())))

        _confuse_matrix, _accuracy, _precision, _recall = run()

        print("------ CONFUSE MATRIX ------\n{}".format(_confuse_matrix))
        print("""\nThe final result:
            | accuracy {}
            | precision {}
            | recall {} \n""".format(_accuracy, _precision, _recall))


# End session =======================================================================================
