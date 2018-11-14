
# coding: utf-8

# In[1]:


#import sys
#get_ipython().system('{sys.executable} -m pip install --user import_ipynb')


# In[2]:
################################
import sys

# Require set params
if len(sys.argv) < 2 or (sys.argv[1] != 'train' and sys.argv[1] != 'test'):
    print('------------------------------------------------')
    print('usage: python3 cnn_lstm_v3.py <command> [<args>]')
    print('')
    print('The command include:')
    print('    train        Training model.')
    print('    test         Testing model.')
    print('')
    print('The args for train command:')
    print('     -co, --class-only')
    print('         Pretraining only classes.')
    print('     -a, --all')
    print('         (default) Training all params include: x, y, w, h, label.')
    print('------------------------------------------------')
    sys.exit(0)
##################################

import os
import numpy as np
import tensorflow as tf
import pandas as pd
import import_ipynb
import shutil
import tensorflow.contrib.slim as slim
from tensorflow.python import debug as tf_debug


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# In[3]:


# 1 if using cudnn on GPU else using CPU
CUDNN_GPU = 0
LAMDA_NOOBJ = 0.5
LAMDA_COORD = 5.0
model_path = 'my_model.csv'
data_path = '../dataset/my_dict.npy'
model_saver = '../model_saver/'
n_classes = 4
n_input = [224,224,3]
n_output = [7,7,9]
batch_size = 32
command = sys.argv[1]
train_mode = sys.argv[2] if len(sys.argv) == 3 else '-a'


# In[4]:


data = np.load(data_path).item()
train_len = data['X_train'].shape[0]
test_len = data['X_test'].shape[0]
#print(pd.read_csv(model_path))


# In[5]:

'''
def cnn_model(input, model_path, is_training=True):
    use_cudnn_on_gpu = True if CUDNN_GPU == 1 else False
    my_model = pd.read_csv(model_path)
    
    ### Define some function for making layers
    def make_input(input_, out_shape, name):
        #result = tf.reshape(input_, [1,out_shape[0],out_shape[1],out_shape[2]], name=name)
        result = input_/255.0
        #result = tf.Print(result, [result[0]], "=> input: ", summarize=224*224*3)
        #tf.summary.histogram(name, result)
        return result

    def make_conv(input_, in_channel, out_channel, filter_, strides, name):
        
        out_channel = int(out_channel)
        strides = int(strides)
        filter_ = list(map(int,filter_.split('x')))
        filter_ = tf.Variable(tf.random_normal([filter_[0],filter_[1],in_channel,out_channel],stddev=0.1), name=name+'_filter')
        # conv
        result = tf.nn.conv2d(input=input_, 
                         filter=filter_,
                         strides=[1, strides, strides, 1],
                         padding='SAME',
                         use_cudnn_on_gpu=use_cudnn_on_gpu,
                         name=name)
        # add bias
        #bias = tf.Variable(tf.random_normal([out_channel], stddev=0.1))
        #result = tf.nn.bias_add(result, bias)
        # batch norm
        result = tf.layers.batch_normalization(result, momentum=0.1, epsilon=1e-5, training=is_training)
        # relu
        result = tf.nn.leaky_relu(result,alpha=0.1)
        
        return result
        
    def make_maxpool(input_, in_channel, filter_, strides, name):
        strides = int(strides)
        filter_ = list(map(int,filter_.split('x')))
        result = tf.nn.max_pool(value=input_,
                             ksize=[1, filter_[0], filter_[1], 1],
                             strides=[1, strides, strides, 1],
                             padding='SAME',
                             name=name)
        return result
        
    def make_flatten(input_, name):
        return tf.contrib.layers.flatten(inputs=input_, scope=name)
        
    def make_fc(input_, out_shape, name):
        result = tf.contrib.layers.fully_connected(inputs=input_,
                                                    activation_fn=tf.nn.relu,
                                                    num_outputs=int(out_shape[0]),
                                                    scope=name)
        #tf.summary.histogram(name, result)
        return result
    
    def make_dropout(input_, name):
        return tf.nn.dropout(input_, 0.75)
    
    def make_reshape(input_, out_shape, name):
        result = tf.reshape(tensor=input_,
                         shape=[tf.shape(input_)[0],out_shape[0],out_shape[1],out_shape[2]],
                         name=name)
        #tf.summary.histogram(name, result)
        return result
    
    def make_softmax(input_, name):
        result = tf.nn.softmax(input_, name=name)
        return result
    
    ### Generate the model base on the model file
    output = input
    layer_match = {
            'Input':      lambda input_, params, _:       make_input(input_, params[5], 'input_image'),
            'Conv':       lambda input_, params, channel: make_conv(input_, channel, params[2], params[3], params[4], 'layer_'+str(params[0])),
            'MaxPooling': lambda input_, params, channel: make_maxpool(input_, channel, params[3], params[4], 'layer_'+str(params[0])),
            'Flatten':    lambda input_, params, _:       make_flatten(input_, 'layer_'+str(params[0])),
            'Fc':         lambda input_, params, _:       make_fc(input_, params[5], 'layer_'+str(params[0])),
            'Reshape':    lambda input_, params, _:       make_reshape(input_, params[5], 'layer_'+str(params[0])),
            'Dropout':    lambda input_, params, _:       make_dropout(input_, 'layer_'+str(params[0])),
            'Softmax':    lambda input_, params, _:       make_softmax(input_, 'layer_'+str(params[0])),
        }
    prev_channel = None
    for layer in my_model.values:
        # preprocessing layer input
        layer[-1] = np.array(list(map(int,layer[-1].split('x'))))
        # map layer
        output = layer_match[layer[1]](output, layer, prev_channel)
        prev_channel = layer[-1][-1]
        
    return output
'''

def cnn_model(input, model_path, is_training):
    with slim.arg_scope([slim.fully_connected, slim.conv2d], 
                            activation_fn=tf.nn.relu,
                            biases_initializer=None,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training}):
    
        with slim.arg_scope([slim.conv2d], 
                            padding='SAME', 
                            activation_fn=tf.nn.leaky_relu):
            
            # Convolution layer 1
            net = slim.conv2d(input, 16, 3, stride=1, scope='conv1')
            net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
            
            # Convolution layer 2
            net = slim.conv2d(net, 32, 3, stride=1, scope='conv2')
            net = slim.max_pool2d(net, 2, stride=2, scope='pool2')

            # Convolution layer 3
            net = slim.conv2d(net, 64, 3, stride=1, scope='conv3')
            net = slim.max_pool2d(net, 2, stride=2, scope='pool3')

            # Convolution layer 4
            net = slim.conv2d(net, 128, 3, stride=1, scope='conv4')
            net = slim.max_pool2d(net, 2, stride=2, scope='pool4')
            
            # Convolution layer 5
            net = slim.conv2d(net, 256, 3, stride=1, scope='conv5')
            net = slim.max_pool2d(net, 2, stride=2, scope='pool5')
            
            # Convolution layer 6
            net = slim.conv2d(net, 512, 3, stride=1, scope='conv6')
            #net = slim.max_pool2d(net, 2, stride=2, scope='pool5')

            #net = slim.flatten(net, scope='flatten1')
            net = slim.conv2d(net, 9, 1, stride=1, scope='conv7')

            
        # Fully connected layer 1
        #net = slim.fully_connected(net, 4096, scope='fc1')

        #net = slim.dropout(net, is_training=is_training, scope='dropout1')  # 0.5 by default
         # Fully connected layer 2
        #net = slim.fully_connected(net, 441,scope='fc2')

        outputs = net#tf.reshape(tensor=net, shape=[batch_size,7,7,9])
        
    return outputs

# In[6]:


def lstm_model(X, weights, biases, isTraining, num_classes):
    # Preprocess data input
    
    # Create LSTM cell
    lstm_cell = None;
    if CUDNN_GPU == 0:
        lstm_cell = tf.contrib.rnn.LSTMBlockCell(n_hidden, 
                                                forget_bias=1.0)
    else:
        lstm_cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, 
                                                   num_units=n_hidden,
                                                   kernel_initializer=tf.initializers.random_uniform(-0.01, 0.01),
                                                   bias_initializer=tf.initializers.constant(0))
    
    # Creates a recurrent neural network specified by RNNCell cell.
    lstm_out, _ = tf.contrib.rnn.static_rnn(cell=lstm_cell, 
                                            inputs=X,
                                            dtype=tf.float32)
    # Dropout layer
    dropout = tf.layers.dropout(inputs=lstm_out, 
                               rate=0.5,
                               training=isTraining)
    
    # Fully connected layer
    # weights_initializer is gaussian distribution
    # bias_initializer is constant by zero
    fc = tf.contrib.layers.fully_connected(inputs=dropout,
                                            num_outputs=num_classes,
                                            activation_fn=None,
                                            weights_initializer=tf.initializers.truncated_normal(stddev=0.01),
                                            bias_initializer=tf.initializers.constant(0))
    
    # Batch Norm + Scale layer
    batch_norm = tf.layers.batch_normalization(inputs=fc,
                                                axis=2,
                                                training=isTraining)
    
    # ReLU activation
    relu = tf.nn.relu_layer(batch_norm)
    
    lstm_last_output = outputs[-1]
    return lstm_last_output


# In[7]:


# In[8]:


# In[9]:



def loss_op_class_only(y_pred, y_true):

    

    return loss;

def loss_op(y_pred, y_true):
    #y_pred = tf.Print(y_pred, [y_pred[0]], "=> y_pred: ", summarize=7*7*9)
    mask_shape = tf.shape(y_true)[:3]
    
    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(7), [7]), (1, 7, 7, 1)))
    cell_y = tf.transpose(cell_x, (0,2,1,3))
    cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [batch_size, 1, 1, 1])

    coord_mask = tf.zeros(mask_shape)
    conf_mask  = tf.zeros(mask_shape)
    class_mask = tf.zeros(mask_shape)

    """
    Adjust prediction
    """
    ### adjust x and y 

    y_pred_xy = y_pred[..., :2]
    #y_pred_xy = tf.Print(y_pred_xy, [y_pred_xy], "===> y_pred_xy: ")
    pred_box_xy = tf.sigmoid(y_pred_xy) + cell_grid

    ### adjust w and h
    pred_box_wh = y_pred[..., 2:4]
    #pred_box_wh = tf.Print(pred_box_wh, [pred_box_wh[0]], summarize=7*7*2)
    pred_box_wh = tf.exp(pred_box_wh)

    ### adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 4])

    ### adjust class probabilities
    pred_box_class = y_pred[..., 5:]

    """
    Adjust ground truth
    """
    ### adjust x and y
    true_box_xy = y_true[..., 0:2] * n_output[0] # relative position to the containing cell

    ### adjust w and h
    true_box_wh = y_true[..., 2:4] * n_output[0] # number of cells accross, horizontally and vertically

    ### adjust confidence
    true_wh_half = true_box_wh / 2.
    true_mins    = true_box_xy - true_wh_half
    true_maxes   = true_box_xy + true_wh_half

    pred_wh_half = pred_box_wh / 2.
    pred_mins    = pred_box_xy - pred_wh_half
    pred_maxes   = pred_box_xy + pred_wh_half       

    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)

    true_box_conf = iou_scores * y_true[..., 4]

    ### adjust class probabilities
    #true_box_class = tf.argmax(y_true[..., 5:], -1)
    true_box_class = y_true[..., 5:]

    """
    Determine the masks
    """
    ### coordinate mask: simply the position of the ground truth boxes (the predictors)
    coord_mask = tf.expand_dims(y_true[..., 4], axis=-1)

    ### confidence mask: penalize predictors + penalize boxes with low IOU
    conf_noobj_mask = 1 - y_true[..., 4]
    
    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    conf_obj_mask = y_true[..., 4]

    ### class mask: simply the position of the ground truth boxes (the predictors)
    class_mask = y_true[..., 4] * tf.ones(tf.shape(y_true[..., 4])) # convert y_true in interger type to float type


    """
    Finalize the loss
    """
    loss_xy = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy) * coord_mask, [1,2,3])
    loss_wh = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh) * coord_mask, [1,2,3])
    loss_conf_obj = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_obj_mask, [1,2]) 
    loss_conf_noobj = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_noobj_mask, [1,2]) 
    loss_class = tf.reduce_sum(tf.reduce_sum(tf.square(true_box_class - tf.nn.softmax(pred_box_class)), -1) * class_mask, [1,2])

    loss = (loss_xy + loss_wh) * LAMDA_COORD + loss_conf_obj + loss_conf_noobj * LAMDA_NOOBJ + loss_class
    #loss = loss_class

    return tf.reduce_mean(loss)


# In[10]:


# reset graph
tf.reset_default_graph()
if command == 'train':
    # remove model_saver folder
    shutil.rmtree(model_saver, ignore_errors=True)
    shutil.rmtree("../checkpoint", ignore_errors=True)


# In[11]:


# the placeholders will be used for the feed_dict param of tf.Session.run()
is_training = tf.placeholder(tf.bool, name='is_training')
training_class_only = tf.placeholder(tf.bool, name='training_class_only')

X = tf.placeholder(tf.float32, [None, n_input[0], n_input[1], n_input[2]], name='X_train')
y = tf.placeholder(tf.float32, [None, n_output[0], n_output[1], n_output[2]], name='y_train')

# model
cnn_model = cnn_model(X, model_path, is_training)

# learning rate
#global_step = tf.Variable(0, trainable=False, name='global_step')
#learning_rate = tf.train.exponential_decay(
#                        0.0001,  # Base learning rate.
#                        global_step,  # Current index into the dataset.
#                        train_len,  # Decay step.
#                        0.95,  # Decay rate.
#                        staircase=True,
#                        name='learning_rate')
learning_rate = 0.0001

# loss 
loss_op = tf.cond(tf.equal(training_class_only, tf.constant(True)), 
                lambda: loss_op_class_only(cnn_model, y),
                lambda: loss_op(cnn_model, y))

# optimizer corrP
optimal = tf.train.AdamOptimizer(learning_rate, name='adam_optimizer')
# increment global_step at each step.
train_op = optimal.minimize(loss_op, name='optimal_min')

# evaluate model
argmax_pred = tf.argmax(cnn_model[...,5:], 1)
argmax_true = tf.argmax(y[...,5:], 1)
correct_prediction = tf.equal(argmax_pred, argmax_true)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[12]:


# tao summary cua cac monitor de quan sat cac bien
tf.summary.scalar('loss_op', loss_op)
tf.summary.scalar('learning_rate', learning_rate)
tf.summary.scalar('accuracy', accuracy)

# gop cac summaries vao mot operation
merged_summary_op = tf.summary.merge_all()

# tao doi tuong log writer va ghi vao Tensorboard
tf_writer = tf.summary.FileWriter('../checkpoint', graph=tf.get_default_graph())

# khoi tao cac variables
init = tf.global_variables_initializer()
# Add ops to save and restore all the variables.
saver = tf.train.Saver()


# In[13]:


def get_batch(dict_field_name, batch):
    return data[dict_field_name][batch*batch_size:min((batch+1)*batch_size,train_len)]


# In[14]:
def train():
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1.
    # training
    with tf.Session(config=config) as sess:
        #sess = tf_debug.TensorBoardDebugWrapperSession(sess, "vudhk-k53sj:6007")
        sess.run(init, feed_dict={is_training:True})
        # Training cycle
        epoch = 0;
        while True:
            total_batch =  train_len // batch_size
            for batch in range(total_batch):
            #for batch in range(3):
                # lay batch tiep theo
                batch_input = get_batch('X_train', batch) 
                batch_label = get_batch('y_train', batch)
                # chay train_op, loss_op, accuracy
                _, cost, acc, summary = sess.run([train_op, loss_op, accuracy, merged_summary_op], 
                        feed_dict={
                            X: batch_input, 
                            y: batch_label, 
                            is_training: True,
                            training_class_only: True if train_mode == '-co' or train_mode == '--class-only' else False
                        })

                # Write logs at every iteration
                tf_writer.add_summary(summary, epoch * total_batch + batch)
                print("---Batch:" + ('%04d,' % (batch)) + ("cost={%.9f}, training accuracy %.5f" % (cost, acc)) + "\n")
                #if batch == 10:
                #print(argmax_pred)
                #break
                
            epoch += 1;
            
            #continue
            # hien thi ket qua sau moi epoch
            print("Epoch:" + ('%04d,' % (epoch)) + ("cost={%.9f}, training accuracy %.5f" % (cost, acc)) + "\n")
            #if epoch % 5 == 0:
                # Luu tru variables vao disk.
                #save_path = saver.save(sess, model_saver + 'nn_model_%04d.ckpt'%(epoch))
                #print("Model saved in path: %s \n" % save_path)

if command == 'train':
    train()

# In[ ]:


# evaluate
def evaluate():
    # TODO: Change the eval_epoch_num variable by a suitable number of epoch.
    eval_epoch_num = 300

    with tf.Session() as sess:
        saver.restore(sess, model_saver + 'nn_model_%04d.ckpt'%(eval_epoch_num))
        avg_acc = 0.
        total_batch =  test_len // batch_size
        for batch in range(total_batch):
            # get next batch
            batch_input = get_batch('X_test', batch) 
            batch_label = get_batch('y_test', batch) 
            acc = sess.run(accuracy, 
                feed_dict={
                    X:batch_input, 
                    y:batch_label, 
                    is_training:False,
                    training_class_only: True if train_mode == '-co' or train_mode == '--class-only' else False
                })

            avg_acc += acc / total_batch
        print("Accuracy on test set: %.5f \n" % (avg_acc))

if command == 'test':
    evaluate()