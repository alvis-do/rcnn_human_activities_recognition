params = {
	'CUDNN_GPU' : 1,
	#'DATA_PATH' : '../../dataset/my_dict.npy',
	'VIDEO_SRC_PATH': '/floyd/input/ucf4',
	'CNN_MODEL_SAVER_PATH' : './model_saver_cnn',
	'CNN_LSTM_MODEL_SAVER_PATH' : './model_saver_cnn_lstm',

	'INPUT_WIDTH': 224,
	'INPUT_HEIGHT': 224,
	'INPUT_CHANNEL': 3,

	'OUTPUT_WIDTH': 6,
	'OUTPUT_HEIGHT': 6,
	'OUTPUT_CHANNEL': 9,

	# 'CNN_BATCH_SIZE' : 64,
	# 'CNN_LSTM_BATCH_SIZE' : 16,

	'N_FRAMES': 16,
	'N_FEATURES': 4096,

	'CLASSES': ['basketball', 'biking', 'diving', 'golf_swing'],
	'N_CLASSES' : 4,
}