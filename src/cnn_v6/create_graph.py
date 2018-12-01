
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
                #x = tf.Print(x, [x], summarize=10)
                use_cudnn_on_gpu = True if params['CUDNN_GPU'] == 1 else False
                # Conv2D wrapper, with bias and relu activation
                x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding_,use_cudnn_on_gpu=use_cudnn_on_gpu)
                # x = tf.nn.bias_add(x, b)
                scale = tf.identity(b)
                mean_conv, var_conv = tf.nn.moments(x, axes = [1,2,3], keep_dims = True)
                x = tf.nn.batch_normalization(x, mean_conv, var_conv, b, scale, 1e-100, name = "batchnorm")
                return tf.nn.relu(x)

            # def conv2d(x, W, b, strides, padding_, out_channel = 0):
            #     use_cudnn_on_gpu = True if params['CUDNN_GPU'] == 1 else False
            #     scale  = tf.Variable(tf.random_normal(shape = [1, 1, 1, out_channel]), dtype = tf.float32, name = "scales")
            #     strides = (1, strides, strides, 1)
            #     conv = tf.nn.conv2d(x, W, strides, padding=padding_,use_cudnn_on_gpu=use_cudnn_on_gpu)
            #     mean_conv, var_conv = tf.nn.moments(conv, axes = [1,2,3], keep_dims = True)
            #     batchnorm = tf.nn.batch_normalization(conv, mean_conv, var_conv, b, scale, 1e-100, name = "batchnorm")
            #     nonlin = tf.nn.relu(batchnorm)
            #     return nonlin


            def maxpool2d(x, k, strides=2):
                # MaxPool2D wrapper
                return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1],
                                      padding='SAME')

            # Store layers weight & bias
            # weights = {
            #     # 5x5 conv, 1 input, 32 outputs
            #     'wc1': tf.Variable(tf.random_uniform([7, 7, 3, 96],maxval=0.005)),
            #     # 5x5 conv, 32 inputs, 64 outputs
            #     'wc2': tf.Variable(tf.random_uniform([5, 5, 96, 256],maxval=0.005)),
                
                
            #     'wc3': tf.Variable(tf.random_uniform([3, 3, 256, 384],maxval=0.005)),
            #     'wc4': tf.Variable(tf.random_uniform([3, 3, 384, 384],maxval=0.005)),
            #     'wc5': tf.Variable(tf.random_uniform([3, 3, 384, 256],maxval=0.005)),
            #     'wc6': tf.Variable(tf.random_uniform([3, 3, 256, 512],maxval=0.005)),
            #     'wc7': tf.Variable(tf.random_uniform([3, 3, 512, 512],maxval=0.005)),
                
            #     # fully connected, 7*7*64 inputs, 1024 outputs
            #     'wd1': tf.Variable(tf.random_uniform([7*7*256, 4096],maxval=0.005)),
            #     #'wd1': tf.Variable(tf.random_uniform([46080, 4096])),
            #     # fully connected, 7*7*64 inputs, 1024 outputs
            #     'wd2': tf.Variable(tf.random_uniform([4096, params['N_FEATURES']],maxval=0.005)),
            #     # 1024 inputs, 4 outputs (class prediction)
            #     'out': tf.Variable(tf.random_uniform([params['N_FEATURES'], params['N_CLASSES']],maxval=0.005))
            # }
            weights = {
                # 5x5 conv, 1 input, 32 outputs
                'wc1': tf.Variable(tf.random_normal([7, 7, 3, 96], stddev=0.1), name='wc1'),
                # 5x5 conv, 32 inputs, 64 outputs
                'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256], stddev=0.1), name='wc2'),
                
                
                'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=0.1), name='wc3'),
                'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384], stddev=0.1), name='wc4'),
                'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=0.1), name='wc5'),
                # 'wc6': tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.1), name='wc6'),
                # 'wc7': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.1), name='wc7'),
                
                # fully connected, 7*7*64 inputs, 1024 outputs
                'wd1': tf.Variable(tf.random_normal([7*7*256, 4096], stddev=0.1), name='wd1'),
                #'wd1': tf.Variable(tf.random_normal([46080, 4096])),
                # fully connected, 7*7*64 inputs, 1024 outputs
                'wd2': tf.Variable(tf.random_normal([4096, params['N_FEATURES']], stddev=0.1), name='wd2'),
                # 1024 inputs, 4 outputs (class prediction)
                'out': tf.Variable(tf.random_normal([params['N_FEATURES'], params['N_CLASSES']], stddev=0.1), name='w_out')
            }

            biases = {
                'bc1': tf.Variable(tf.random_normal([96], stddev=0.1), name='bc1'),
                'bc2': tf.Variable(tf.random_normal([256], stddev=0.1), name='bc2'),
                'bc3': tf.Variable(tf.random_normal([384], stddev=0.1), name='bc3'),
                'bc4': tf.Variable(tf.random_normal([384], stddev=0.1), name='bc4'),
                'bc5': tf.Variable(tf.random_normal([256], stddev=0.1), name='bc5'),
                # 'bc6': tf.Variable(tf.random_normal([512], stddev=0.1), name='bc6'),
                # 'bc7': tf.Variable(tf.random_normal([512], stddev=0.1), name='bc7'),
                'bd1': tf.Variable(tf.random_normal([4096], stddev=0.1), name='db2'),
                'bd2': tf.Variable(tf.random_normal([params['N_FEATURES']], stddev=0.1), name='db2'),
                'out': tf.Variable(tf.random_normal([params['N_CLASSES']], stddev=0.1), name='b_out')
            }

            # Convolution Layer
            X = tf.placeholder(tf.float32, [None, params['INPUT_WIDTH'], params['INPUT_HEIGHT'], params['INPUT_CHANNEL']], name='input')
            X = X / 255

            # net 1
            conv1 = conv2d(X, weights['wc1'], biases['bc1'],2,'VALID')
            conv1 = maxpool2d(conv1, k=3)

            conv2 = conv2d(conv1, weights['wc2'], biases['bc2'],2,'VALID')
            conv2 = maxpool2d(conv2, k=3)
            
            conv3 = conv2d(conv2, weights['wc3'], biases['bc3'],1,'SAME')
            conv4 = conv2d(conv3, weights['wc4'], biases['bc4'],1,'SAME')
            conv5 = conv2d(conv4, weights['wc5'], biases['bc5'],1,'SAME')
            conv5 = maxpool2d(conv5, k=3)

            # conv6 = conv2d(conv5, weights['wc6'], biases['bc6'],1,'SAME')
            # conv7 = conv2d(conv6, weights['wc7'], biases['bc7'],1,'SAME')
            # conv7 = maxpool2d(conv7, k=3, strides=1)

            fc1 = tf.reshape(conv5, [-1, weights['wd1'].get_shape().as_list()[0]])
            fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
            fc1 = tf.nn.relu(fc1)
            
            fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
            fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
            fc2 = tf.nn.relu(fc2)
            fc2 = tf.identity(fc2, name='out_features')

            out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
            out = tf.identity(out, name='out_class')

            return out
    


def create_lstm_net():
    with gl.as_default() as g:
        with g.name_scope("LSTM_MODEL"):

            # Create LSTM cell
            n_hidden = 512
            num_layers = 3


            def lstm_cpu(X, init_state):
                with g.name_scope("lstm_cpu"):
                    X_seq = tf.unstack(X, axis=1)

                    lstm_cell = tf.contrib.rnn.LSTMBlockCell(n_hidden, forget_bias=1.0)
                    multi_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers)
                    output, state = tf.contrib.rnn.static_rnn(multi_cell, X_seq, initial_state=init_state, dtype=tf.float32)
                    #output = tf.stack(output, axis=1)
                    #print(state)

                    return output, state

            def lstm_gpu(X, init_state):
                with g.name_scope("lstm_gpu"):
                    lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=num_layers,
                                                        num_units=n_hidden,
                                                        kernel_initializer=tf.initializers.random_uniform(-0.01, 0.01))
                    lstm.build(X.get_shape())
                    return lstm(X, initial_state=init_state)
                
            # Weight + bias
            weights = {
                'out': tf.Variable(tf.random_normal([n_hidden, params['N_CLASSES']]), name='lstm_wo')
            }
            biases = {
                'out': tf.Variable(tf.random_normal([params['N_CLASSES']]), name='lstm_bo')
            }


            isTraining = tf.placeholder(tf.bool, name='is_training')
            # cell_state = tf.placeholder(tf.float32, [params['CNN_LSTM_BATCH_SIZE'], n_hidden], name='cell_state')
            # hidden_state = tf.placeholder(tf.float32, [params['CNN_LSTM_BATCH_SIZE'], n_hidden], name='hidden_state')
            # init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)
            init_state = tf.placeholder(tf.float32, [num_layers, 2, params['CNN_LSTM_BATCH_SIZE'], n_hidden], name='init_state')
            state_per_layer_list = tf.unstack(init_state, axis=0)
            rnn_tuple_state = tuple(
                [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
                 for idx in range(num_layers)]
            )

            #X = tf.placeholder(tf.float32, [None, params['N_FRAMES'], params['N_CLASSES']], name='input')
            X = g.get_tensor_by_name("CNN_MODEL/out_class:0")
            # X = tf.Print(X, [X[0]], summarize=4)
            X_ = tf.reshape(X, [-1, params['N_FRAMES'], params['N_CLASSES']])
            #X_ = tf.Print(X, [X[0]], summarize=4, message='X =>> ')

            X_ = tf.identity(X_, name='input') # (?, ,16, 4)

            # Creates a recurrent neural network specified by RNNCell cell.
            states_series, current_state = lstm_cpu(X_, rnn_tuple_state) if params['CUDNN_GPU'] == 0 else lstm_gpu(X_, rnn_tuple_state)
            
            out = [tf.matmul(state, weights['out']) + biases['out'] for state in states_series]
            out = tf.identity(out, name='output') # (16, ?, 4)

            curr_state = tf.identity(current_state, name='current_state') # (2, 2, ?, 512)


if __name__ == "__main__":
    create_cnn_net()
    create_lstm_net()

    shutil.rmtree("./graph", ignore_errors=True)
    os.mkdir("./graph")

    tf.summary.FileWriter("./graph", gl)

    with tf.Session(graph = gl) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=None)
        path = saver.save(sess, "./graph/cnn_lstm_model.ckpt")  

    print("Graph was stored at {} !!!".format(path))
