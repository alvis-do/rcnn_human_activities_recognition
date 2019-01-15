'''
$ python3 cnn.py 0
'''

import argparse
import numpy as np
import tensorflow as tf
import os
from config import params
from create_graph import create_cnn_net_graph
from data_handler import get_dataset, get_batch, get_clip, split_valid_set


# validate parameters
parser = argparse.ArgumentParser(description='Train and test the cnn model')
parser.add_argument('mode', metavar='mode', type=int, nargs=1, choices=[0,1],
                    help='0 for train mode, 1 for test mode')
parser.add_argument('-m', '--modelpath', nargs=1, 
                    help='The path of trained cnn model.')
args = parser.parse_args()

# print(args.modelpath)

# Register first time tf.contrib namspace for loading graph from meta ckpt
# because ops  in this are lazily registered
dir(tf.contrib)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"



# Training Parameters
learning_rate = 0.0001
display_step = 5
epochs = 10
epoch_save = 2
batch_size = 16 # the number of clips
clip_stride = 2 # get a frame for each clip_stride frames



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
cnn_graph = None
saver = None
first_train = True
if args.mode[0] == 0 and args.modelpath == None: # (first train mode)
    cnn_graph = create_cnn_net_graph()
else:
    first_train = False
    saver = tf.train.import_meta_graph("{}.meta".format(args.modelpath[0]))


# open a tensorflow session
with tf.Session(graph=cnn_graph, config=config) as sess:

    # restore cnn model if existed
    if saver:
        saver.restore(sess, args.modelpath[0])

    # get graph of current session
    g = sess.graph

    with g.name_scope("CNN_OP"):
        
        # Get input, output of cnn model from graph
        X = g.get_tensor_by_name("CNN_MODEL/input:0") # Placeholder
        logits = g.get_tensor_by_name("CNN_MODEL/out_class:0") 


        if first_train:
            # Placeholder the truth labels Y
            Y = tf.placeholder(tf.float32, [None, params['N_CLASSES']], name='y_true') # Placeholder

            # get predict by softmax function 
            prediction = tf.nn.softmax(logits) 

            # Define loss and optimizer
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, name='cnn_adam').minimize(loss_op)
            
            # Evaluate model
            correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            
            # Add to collection
            tf.add_to_collection("cnn_predict", prediction)
            tf.add_to_collection("cnn_loss_op", loss_op)
            tf.add_to_collection("cnn_train_op", train_op)
            tf.add_to_collection("cnn_acc", accuracy)

        else:
            # Get variables from collection
            Y = g.get_tensor_by_name("CNN_OP/y_true:0")
            prediction = tf.get_collection("cnn_predict")[0]
            loss_op = tf.get_collection("cnn_loss_op")[0]
            train_op = tf.get_collection("cnn_train_op")[0]
            accuracy = tf.get_collection("cnn_acc")[0]


        # Make monitor from tensor board
        summary_train = tf.summary.merge([tf.summary.scalar("loss_op", loss_op), tf.summary.scalar("accuracy", accuracy)])
        summary_val = tf.summary.merge([tf.summary.scalar("val_loss_op", loss_op), tf.summary.scalar("val_accuracy", accuracy)])
        


    tf_writer = tf.summary.FileWriter(params['CNN_MODEL_SAVER_PATH'], g)

    # init global vars
    sess.run(tf.global_variables_initializer())

    # override the available global variables in the cnn model if existed
    if saver:
        saver.restore(sess, args.modelpath[0])

    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)

    
    def run(epoch = 0):
        loss = 0
        acc = 0
        test_loss = 0
        test_acc = 0
        eval_set = test_set

        def reshape_input(frames, labels):
            # reshape inputs and labels
            _input = np.reshape(frames, (-1, params['INPUT_WIDTH'], params['INPUT_HEIGHT'], params['INPUT_CHANNEL']))
            _label = np.repeat(labels, params['N_FRAMES'] // clip_stride, axis=0)
            return _input, _label

        def run_session(frames, labels, fetches):
            _input, _label = reshape_input(frames, labels)
            # feed placeholder and train model
            _feed_dict={X: _input, Y: _label}
            return sess.run(fetches, feed_dict=_feed_dict)


        if args.mode[0] == 0: # train mode

            eval_set = valid_set
            total_batch = (train_len // batch_size) + 1

            tmp_loss = 0
            tmp_acc = 0

            for batch, clips in get_batch(train_set, batch_size):
                if batch == 0:
                    continue
                if batch > total_batch or len(clips)==0:
                    break

                # get frames from clips - the a small equence in a video
                frames, labels = get_clip(clips, clip_stride)
                _loss, _acc, _summ, _ = run_session(frames, labels, [loss_op, accuracy, summary_train, train_op])


                # Sum loss and acc
                tmp_loss += _loss
                tmp_acc += _acc
                loss += _loss
                acc += _acc

                # summary data to tensor board
                tf_writer.add_summary(_summ, epoch * total_batch + batch)

                # Display if neccessary
                if batch % display_step == 0:
                    print("Batch {}/{}: Loss {} - Acc {}".format(batch, total_batch, tmp_loss/display_step, tmp_acc/display_step))
                    tmp_loss = 0
                    tmp_acc = 0
                

            # compute the average of loss and acc
            loss /= batch
            acc /= batch



        # Valid or test model
        if params['USE_VALIDATION_DATASET'] or args.mode[0] == 1:
            eval_len = len(eval_set)
            total_batch = (eval_len // batch_size) + 1
            for batch, clips in get_batch(eval_set, batch_size):
                if batch == 0:
                    continue
                if batch > total_batch or len(clips)==0:
                    break

                # get frames from clips - the a small equence in a video
                frames, labels = get_clip(clips, clip_stride)
                _loss, _acc, _summ = run_session(frames, labels, [loss_op, accuracy, summary_val])

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
                save_path = saver.save(sess, "{}/model{}_{}.ckpt".format(params['CNN_MODEL_SAVER_PATH'], "_resume" if not first_train else "", epoch + 1))
                print("Model saved in path: %s" % save_path)

    else:
        print("\nTesting on {} sample(s)".format(test_len))

        loss, acc = run()

        print("\nThe final result: Loss {} - Acc {}".format(loss, acc))


# End session =======================================================================================
