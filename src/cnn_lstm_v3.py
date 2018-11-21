
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
batch_size = 8

# global parameters
command = sys.argv[1]
train_mode = sys.argv[2] if len(sys.argv) == 3 else '-a'
training_class_only_arg = True if train_mode == '-co' or train_mode == '--class-only' else False

# In[4]:


data = np.load(data_path).item()
train_len = data['X_train'].shape[0]
test_len = data['X_test'].shape[0]


# In[5]:

def cnn_model(input, model_path, is_training, train_class_only):
    with slim.arg_scope([slim.fully_connected, slim.conv2d], 
                            reuse=tf.AUTO_REUSE,
                            activation_fn=tf.nn.relu,
                            #biases_initializer=None,
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
            #net = slim.conv2d(net, 512, 2, stride=1, scope='conv7')
            #net = slim.conv2d(net, 256, 2, stride=1, scope='conv8')
            #net = slim.max_pool2d(net, 2, stride=1, scope='pool6')

            def class_only_branch(input_): 
                net_ = slim.flatten(input_, scope='flatten1')
                # net_ = slim.fully_connected(net_, 4096, scope='fc1')
                # net_ = slim.dropout(net_, is_training=is_training, scope='dropout1')
                net_ = slim.fully_connected(net_, 512, scope='fc2')
                net_ = slim.dropout(net_, is_training=is_training, scope='dropout2')
                net_ = slim.fully_connected(net_, 4, scope='fc3')
                return net_

            def all_branch(input_):
                #net_ = slim.conv2d(net, 256, 1, stride=1, scope='conv7')
                #net_ = slim.conv2d(net, 512, 3, stride=1, scope='conv8')
                net_ = slim.flatten(input_, scope='flatten2')
                net_ = slim.fully_connected(net_, 4096, scope='fc4')
                net_ = slim.dropout(net_, is_training=is_training, scope='dropout3')
                net_ = slim.fully_connected(net_, 441, scope='fc5')
                net_ = tf.reshape(tensor=net_, shape=[tf.shape(net_)[0],7,7,9])
                return net_

            outputs = tf.cond(tf.equal(train_class_only, tf.constant(True)), 
                lambda: class_only_branch(net),
                lambda: all_branch(net))
        
    return outputs

def loss_op_class_only(y_pred, y_true):
    #y_pred = tf.Print(y_pred, [y_pred], summarize=4*batch_size)
    #labels = tf.one_hot(tf.squeeze(tf.argmax(y_true[...,5:], -1)), n_classes)
    #labels = tf.reshape(tensor=labels,shape=[tf.shape(y_true)[0],4])    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true))    
    return loss

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
    class_mask = y_true[..., 4]


    """
    Finalize the loss
    """
    loss_xy = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy) * coord_mask, [1,2,3])
    loss_wh = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh) * coord_mask, [1,2,3])
    loss_conf_obj = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_obj_mask, [1,2]) 
    loss_conf_noobj = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_noobj_mask, [1,2]) 
    loss_class = tf.reduce_sum(tf.reduce_sum(tf.square(true_box_class - tf.nn.softmax(pred_box_class)), -1) * class_mask, [1,2])

    loss = (loss_xy + loss_wh) * LAMDA_COORD + loss_conf_obj + loss_conf_noobj * LAMDA_NOOBJ + loss_class

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
y = tf.cond(tf.equal(training_class_only, tf.constant(True)),
    lambda: tf.placeholder(tf.float32, [None, n_classes], name='y_train_class_only'),
    lambda: tf.placeholder(tf.float32, [None, n_output[0], n_output[1], n_output[2]], name='y_train'))

# model
cnn_model = cnn_model(X, model_path, is_training, training_class_only)

# learning rate
learning_rate = 0.0001

# loss 
loss_op = tf.cond(tf.equal(training_class_only, tf.constant(True)),
        lambda: loss_op_class_only(cnn_model, y),
        lambda: loss_op(cnn_model, y))

#loss_op = loss_op(cnn_model, y, training_class_only)

# optimizer corrP
optimal = tf.train.AdamOptimizer(learning_rate, name='adam_optimizer')
# increment global_step at each step.
train_op = optimal.minimize(loss_op, name='optimal_min')

# evaluate model
argmax_pred, argmax_true = tf.cond(tf.equal(training_class_only, tf.constant(True)),
    lambda: (tf.to_float(tf.argmax(cnn_model, 1)), 
            tf.to_float(tf.argmax(y, 1))),
    lambda: (0.,0.))
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
            total_acc = 0;
            total_cost = 0;
            total_batch =  train_len // batch_size
            #total_batch = 10;
            for batch in range(total_batch):
                # lay batch tiep theo
                batch_input = get_batch('X_train', batch) 
                batch_label = get_batch('y_train_class_only', batch) if training_class_only_arg == True else get_batch('y_train', batch)
                # chay train_op, loss_op, accuracy
                _, cost, acc, summary = sess.run([train_op, loss_op, accuracy, merged_summary_op], 
                        feed_dict={
                            X: batch_input, 
                            y: batch_label, 
                            is_training: True,
                            training_class_only: training_class_only_arg
                        })
                total_acc += acc
                total_cost += cost
                # Write logs at every iteration
                tf_writer.add_summary(summary, epoch * total_batch + batch)
                print("---Batch:" + ('%04d,' % (batch)) + ("cost={%.9f}, training accuracy %.5f" % (cost, acc)) + "\n")
                
            epoch += 1;
            
            # hien thi ket qua sau moi epoch
            print("Epoch:" + ('%04d,' % (epoch)) + ("average cost={%.9f}, average accuracy %.5f" % (total_cost/total_batch, total_acc/total_batch)) + "\n")
            if epoch % 1 == 0:
                # Luu tru variables vao disk.
                save_path = saver.save(sess, model_saver + 'nn_model_%04d.ckpt'%(epoch))
                print("Model saved in path: %s \n" % save_path)

if command == 'train':
    train()

# In[ ]:


# evaluate
def evaluate():
    # TODO: Change the eval_epoch_num variable by a suitable number of epoch.
    eval_epoch_num =  

    with tf.Session() as sess:
        saver.restore(sess, model_saver + 'nn_model_%04d.ckpt'%(eval_epoch_num))
        avg_acc = 0.
        total_batch =  test_len // batch_size
        for batch in range(total_batch):
            # get next batch
            batch_input = get_batch('X_test', batch) 
            batch_label = get_batch('y_test_class_only', batch) if training_class_only_arg == True else get_batch('y_test', batch)
            acc = sess.run(accuracy, 
                feed_dict={
                    X:batch_input, 
                    y:batch_label, 
                    is_training:False,
                    training_class_only: training_class_only_arg 
                })

            avg_acc += acc / total_batch
        print("Accuracy on test set: %.5f \n" % (avg_acc))

if command == 'test':
    evaluate()