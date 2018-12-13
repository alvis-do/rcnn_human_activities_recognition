'''
$ python3 lrcn_single_pred.py /path/to/lrcn_model.ckpt /path/to/video.wav
$ python3 lrcn_single_pred.py /path/to/lrcn_model.ckpt /path/to/video_folder
'''

import argparse, os
import glob
from data_handler import get_clip_from_video



# validate parameters
parser = argparse.ArgumentParser(description='Test the lrcn model over full video')
parser.add_argument('lrcn_model', metavar='lrcn_model', nargs=1,
                    help='The path of trained lrcn model.')
parser.add_argument('video_path', metavar='video_path', nargs=1, 
                    help='The path of video or directory containing videos.')
args = parser.parse_args()



# Register first time tf.contrib namspace for loading graph from meta ckpt
# because ops  in this are lazily registered
dir(tf.contrib)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# setup config for session
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.



lrcn_model_path = args.lrcn_model[0]
# Get all video in directory if specifed from args
video_paths = args.video_path
if not os.path.isFile(video_path[0]):
	pass


saver = tf.train.import_meta_graph("{}.meta".format(lrcn_model_path))


with tf.Session(config=config) as sess:

	saver.restore(sess, lrcn_model_path)

	# get graph of current session
    g = sess.graph


    with g.name_scope("PRED_VIDEO"):

    	# Placeholders
    	cnn_X = g.get_tensor_by_name("CNN_MODEL/input:0")
    	cnn_Y = g.get_tensor_by_name("CNN_OP/y_true:0")
    	lstm_X = g.get_tensor_by_name("LSTM_MODEL/input:0")
    	lstm_Y = g.get_tensor_by_name("LSTM_OP/y_true:0")
    	lstm_init_state = g.get_tensor_by_name("LSTM_MODEL/init_state:0")

    	# values
    	cnn_features = g.get_tensor_by_name("CNN_MODEL/out_features:0")
    	lstm_prediction = tf.get_collection("lstm_predict")[0]
    	lstm_y_hat = tf.get_collection("lstm_y_hat")[0]
    	lstm_accuracy = tf.get_collection("lstm_acc")[0]



    print("Predicting...")
    print("\n----------------------------------------------------------------")
    
    _acc = 0
    _lbs = []
    _preds = []
    for _vpath in video_paths:
                
        # init zero state for lstm
        init_state_zeros = g.get_tensor_by_name("LSTM_MODEL/init_state_zeros:0") 
        _current_state = init_state_zeros.eval()

    	for frames in get_clip_from_video(_vpath):

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
            _loss, _acc, _y_hat, _summ = sess.run([loss_op, accuracy, y_hat, summary_val], feed_dict=_lstm_feed_dict)




    	print("""- {}:
    			| pred-label: {} - probability: {}
    			| may-be: {} - probability: {}
    			| time: {} (s)""".format(_vpath, _lb, _prob, _lb2, _prob2, _time))


    # End loop


    _cmatrix = tf.confusion_matrix(_lbs, _preds).eval()

    print("\n----------------------------------------------------------------")
    print("Accuracy: {}".format(_acc))
    print("Confuse matrix: \n \t {}".format(_cmatrix))


# End session