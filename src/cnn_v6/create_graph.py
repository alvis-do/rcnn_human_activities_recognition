
import tensorflow as tf
import os
import shutil
from config import params

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gl = tf.Graph()

def create_cnn_net():
    with gl.as_default() as g:
        with g.name_scope("CNN_MODEL"):
            # Create some wrappers for simplicity
            def conv2d(x, W, b, strides, padding_):
                use_cudnn_on_gpu = True if params['CUDNN_GPU'] == 1 else False
                # Conv2D wrapper, with bias and relu activation
                x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding_,use_cudnn_on_gpu=use_cudnn_on_gpu)
                x = tf.nn.bias_add(x, b)
                return tf.nn.relu(x)


            def maxpool2d(x, k, strides=2):
                # MaxPool2D wrapper
                return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1],
                                      padding='SAME')

            # Store layers weight & bias
            weights = {
                # 5x5 conv, 1 input, 32 outputs
                'wc1': tf.Variable(tf.random_uniform([7, 7, 3, 96],minval=-0.05, maxval=0.05)),
                # 5x5 conv, 32 inputs, 64 outputs
                'wc2': tf.Variable(tf.random_uniform([5, 5, 96, 256],minval=-0.05, maxval=0.05)),
                
                
                'wc3': tf.Variable(tf.random_uniform([3, 3, 256, 384],minval=-0.05, maxval=0.05)),
                'wc4': tf.Variable(tf.random_uniform([3, 3, 384, 384],minval=-0.05, maxval=0.05)),
                'wc5': tf.Variable(tf.random_uniform([3, 3, 384, 256],minval=-0.05, maxval=0.05)),
                'wc6': tf.Variable(tf.random_uniform([3, 3, 256, 512],minval=-0.05, maxval=0.05)),
                'wc7': tf.Variable(tf.random_uniform([3, 3, 512, 512],minval=-0.05, maxval=0.05)),
                
                # fully connected, 7*7*64 inputs, 1024 outputs
                'wd1': tf.Variable(tf.random_uniform([7*7*256, 4096],minval=-0.05, maxval=0.05)),
                #'wd1': tf.Variable(tf.random_uniform([46080, 4096])),
                # fully connected, 7*7*64 inputs, 1024 outputs
                'wd2': tf.Variable(tf.random_uniform([4096, 4096],minval=-0.05, maxval=0.05)),
                # 1024 inputs, 4 outputs (class prediction)
                'out': tf.Variable(tf.random_uniform([4096, params['N_CLASSES']],minval=-0.05, maxval=0.05))
            }

            biases = {
                'bc1': tf.Variable(tf.random_normal([96])),
                'bc2': tf.Variable(tf.random_normal([256])),
                'bc3': tf.Variable(tf.random_normal([384])),
                'bc4': tf.Variable(tf.random_normal([384])),
                'bc5': tf.Variable(tf.random_normal([256])),
                'bc6': tf.Variable(tf.random_normal([512])),
                'bc7': tf.Variable(tf.random_normal([512])),
                'bd1': tf.Variable(tf.random_normal([4096])),
                'bd2': tf.Variable(tf.random_normal([4096])),
                'out': tf.Variable(tf.random_normal([params['N_CLASSES']]))
            }

            # Convolution Layer
            X = tf.placeholder(tf.float32, [None, params['INPUT_WIDTH'], params['INPUT_HEIGHT'], params['INPUT_CHANNEL']], name='input')
            X = X / 255

            conv1 = conv2d(X, weights['wc1'], biases['bc1'],2,'VALID')
            conv1 = maxpool2d(conv1, k=3)

            conv2 = conv2d(conv1, weights['wc2'], biases['bc2'],2,'VALID')
            conv2 = maxpool2d(conv2, k=3)
            
            conv3 = conv2d(conv2, weights['wc3'], biases['bc3'],1,'SAME')
            conv4 = conv2d(conv3, weights['wc4'], biases['bc4'],1,'SAME')
            conv5 = conv2d(conv4, weights['wc5'], biases['bc5'],1,'SAME')
            conv5 = maxpool2d(conv5, k=3)
            
            #Pre-train
            # Fully connected layer
            fc1 = tf.reshape(conv5, [-1, weights['wd1'].get_shape().as_list()[0]])
            fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
            fc1 = tf.nn.relu(fc1)
            
            fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
            fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
            fc2 = tf.nn.relu(fc2)
            fc2 = tf.identity(fc2, name='out_features')

            out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
            out = tf.identity(out, name='out_class')

    


def create_lstm_net():
    with gl.as_default() as g:
        with g.name_scope("LSTM_MODEL"):

            # Create LSTM cell
            n_hidden = 1024


            def lstm_cpu(X):
                with g.name_scope("lstm_cpu"):
                    X_seq = tf.unstack(X, axis=1)
                    lstm_cell = tf.contrib.rnn.LSTMBlockCell(n_hidden, forget_bias=1.0)
                    output, state = tf.contrib.rnn.static_rnn(lstm_cell, X_seq, dtype=tf.float32)
                    return tf.stack(output, axis=1), state

            def lstm_gpu(X):
                with g.name_scope("lstm_gpu"):
                    lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, 
                                                        num_units=n_hidden,
                                                        kernel_initializer=tf.initializers.random_uniform(-0.01, 0.01))
                    lstm.build(X.get_shape())
                    return lstm(X)
            

            isTraining = tf.placeholder(tf.bool, name='is_training')
            #X = tf.placeholder(tf.float32, [None, params['N_FRAMES'], params['N_FEATURES']], name='input')
            X = g.get_tensor_by_name("CNN_MODEL/out_features:0")
            X = tf.reshape(X, [-1, params['N_FRAMES'], params['N_FEATURES']])
            
            # Creates a recurrent neural network specified by RNNCell cell.
            lstm_out, lstm_state = lstm_cpu(X) if params['CUDNN_GPU'] == 0 else lstm_gpu(X)
            
            # Dropout layer
            dropout = tf.layers.dropout(inputs=lstm_out, rate=0.5, training=isTraining)
            
            # Fully connected layer
            out = tf.layers.dense(
                    tf.layers.batch_normalization(dropout),
                    params['N_CLASSES'], activation=None, 
                    kernel_initializer=tf.orthogonal_initializer())

            out = tf.identity(out, name='output')


if __name__ == "__main__":
    create_cnn_net()
    create_lstm_net()

    shutil.rmtree("./graph", ignore_errors=True)
    os.mkdir("./graph")

    tf.summary.FileWriter("./graph", gl)

    with tf.Session(graph = gl) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        path = saver.save(sess, "./graph/cnn_lstm_model.ckpt")  

    print("Graph was stored at {} !!!".format(path))
