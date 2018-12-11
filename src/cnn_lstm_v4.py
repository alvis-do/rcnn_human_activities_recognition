
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
    print('usage: python3 cnn_lstm_v4.py <command> [<args>]')
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


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# In[3]:

data_path = '../dataset/my_dict.npy'
model_saver = '../model_saver/'

# 1 if using cudnn on GPU else using CPU
CUDNN_GPU = 0
LAMDA_NOOBJ = 0.5
LAMDA_COORD = 5.0
N_CLASSES = 4
INPUT_WIDTH = 224
INPUT_HEIGHT = 224
INPUT_DEPTH = 3
OUTPUT_WIDTH = 7
OUTPUT_HEIGHT = 7
OUTPUT_DEPTH = 9
BATCH_SIZE = 1

# global parameters
command = sys.argv[1]
train_mode = sys.argv[2] if len(sys.argv) == 3 else '-a'
training_class_only_arg = True if train_mode == '-co' or train_mode == '--class-only' else False

# In[4]:


data = np.load(data_path).item()
train_len = data['X_train'].shape[0]
test_len = data['X_test'].shape[0]

# reset graph
tf.reset_default_graph()

# Session tf
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.
sess = tf.Session(config=config)
#sess = tf_debug.TensorBoardDebugWrapperSession(sess, "vudhk-k53sj:6007")
g = sess.graph


# In[5]:

def create_cnn_model_graph():
    with g.name_scope("CNN"):
        def conv(n, in_name, out_channels, kernel_size, stride, nonlin="relu", keep_prob=1, batchnorm=1):
            in_tensor = g.get_tensor_by_name(in_name)
            batch_size, height, width, in_channels = in_tensor.get_shape().as_list()
            with g.name_scope("conv_{}".format(n)):
                kernel = tf.Variable(tf.random_uniform(shape = [kernel_size, kernel_size, in_channels, out_channels])/ (kernel_size*kernel_size*out_channels) , dtype = tf.float32, name = "kernel")
                scale  = tf.Variable(tf.random_normal(shape = [1, 1, 1, out_channels]), dtype = tf.float32, name = "scales")
                bias   = tf.Variable(tf.random_normal(shape = [1, 1, 1, out_channels]), dtype = tf.float32, name = "biases")
                '''
                conv
                batchnorm + bias + scale
                drop
                nonlin
                '''
                strides = (1, stride, stride, 1)
                conv = tf.nn.conv2d(in_tensor, kernel, strides, padding="SAME", name = "conv")
                if (batchnorm):
                    mean_conv, var_conv = tf.nn.moments(conv, axes = [1,2,3], keep_dims = True)
                    batchnorm = tf.nn.batch_normalization(conv, mean_conv, var_conv, bias, scale, 1e-100, name = "batchnorm")
                else:
                    batchnorm = tf.add(conv, bias, name = "batchnorm")
                drop = tf.nn.dropout(batchnorm, keep_prob, name = "drop")
                if nonlin == "relu":
                    nonlin = tf.nn.leaky_relu(drop)
                elif nonlin == "sigmoid":
                    nonlin = tf.sigmoid(drop)
                elif nonlin == "linear":
                    nonlin = tf.identity(drop)
                else:
                    raise Exception(" \"{}\" is not a nonlinear function!".format(nonlin))
                conv = tf.identity(nonlin, name = "out")

            return conv

        def maxpool(n, in_name, kernel_size, stride):
            in_tensor = g.get_tensor_by_name(in_name)
            batch_size, height, width, in_channels = in_tensor.get_shape().as_list()
            with g.name_scope("maxpool_{}".format(n)):
                ksize = [1, kernel_size, kernel_size, 1]
                strides = [1, stride, stride, 1]
                '''
                maxpool
                '''
                maxpool = tf.nn.max_pool(in_tensor, ksize, strides, padding="SAME")
                maxpool = tf.identity(maxpool, name = "out")
            return maxpool
 
        def output(in_name):
            in_tensor = g.get_tensor_by_name(in_name)
            out = tf.identity(in_tensor, name='output')
            return out

        # 224 x 224 x 3
        X = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH], name='input')
        
        conv(0, "CNN/input:0", 32, 3, 1) # 224 x 224 x 32
        maxpool(1, "CNN/conv_0/out:0", 2, 2) # 112 x 112 x 32

        conv(2, "CNN/maxpool_1/out:0", 64, 3, 1) # 112 x 112 x 64
        maxpool(3, "CNN/conv_2/out:0", 2, 2) # 56 x 56 x 64

        conv(4, "CNN/maxpool_3/out:0", 128, 3, 1) # 56 x 56 x 128
        maxpool(5, "CNN/conv_4/out:0", 2, 2) # 28 x 28 x 128
        
        conv(6, "CNN/maxpool_5/out:0", 256, 3, 1) # 28 x 28 x 256
        maxpool(7, "CNN/conv_6/out:0", 2, 2) # 14 x 14 x 256

        conv(8, "CNN/maxpool_7/out:0", 512, 3, 1) # 14 x 14 x 512
        maxpool(9, "CNN/conv_8/out:0", 2, 2) # 7 x 7 x 512

        conv(10, "CNN/maxpool_9/out:0", 1024, 3, 1) # 7 x 7 x 1024
        maxpool(11, "CNN/conv_10/out:0", 2, 1) # 7 x 7 x 1024

        conv(12, "CNN/maxpool_11/out:0", 2048, 3, 1) # 7 x 7 x 2048
        conv(13, "CNN/conv_12/out:0", 512, 1, 1) # 7 x 7 x 512
        conv(14, "CNN/conv_13/out:0", 1024, 3, 1, keep_prob=0.5) # 7 x 7 x 1024
        conv(15, "CNN/conv_14/out:0", 256, 1, 1) # 7 x 7 x 256
        conv(16, "CNN/conv_15/out:0", 9, 1, 1, nonlin="linear", batchnorm=0) # 7 x 7 x 9
        
        output('CNN/conv_16/out:0')



def loss_op(y_pred, y_true):
    mask_shape = tf.shape(y_true)[:3]
    
    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(OUTPUT_WIDTH), [OUTPUT_WIDTH]), (1, OUTPUT_WIDTH, OUTPUT_HEIGHT, 1)))
    #cell_x = tf.Print(cell_x, [cell_x[0]], "=> y_pred: ", summarize=7*7)
    cell_y = tf.transpose(cell_x, (0,2,1,3))
    cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [BATCH_SIZE, 1, 1, 1])

    coord_mask = tf.zeros(mask_shape)
    conf_mask  = tf.zeros(mask_shape)
    class_mask = tf.zeros(mask_shape)

    """
    Adjust prediction
    """
    ### adjust x and y 

    y_pred_xy = y_pred[..., :2]
    pred_box_xy = tf.sigmoid(y_pred_xy) + cell_grid
    #pred_box_xy = tf.Print(pred_box_xy, [pred_box_xy[0,3,4]], summarize=2)

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
    true_box_yx = y_true[..., :2] * OUTPUT_WIDTH # relative position to the containing cell
    box_y = true_box_yx[...,0:1]
    box_x = true_box_yx[...,1:2]
    true_box_xy = tf.concat([box_x, box_y], -1)
    #true_box_xy = tf.Print(true_box_xy, [true_box_xy[0,3,4]], summarize=2)

    ### adjust w and h
    true_box_wh = y_true[..., 2:4]
    #true_box_wh = tf.Print(true_box_wh, [true_box_wh[0,3,4]], summarize=2) 
    true_box_wh = true_box_wh * OUTPUT_WIDTH 

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
    loss_xy = LAMDA_COORD*tf.reduce_mean(tf.reduce_sum(tf.square(true_box_xy-pred_box_xy) * coord_mask, [1,2,3]))
    loss_wh = LAMDA_COORD*tf.reduce_mean(tf.reduce_sum(tf.square(true_box_wh-pred_box_wh) * coord_mask, [1,2,3]))
    loss_conf_obj = tf.reduce_mean(tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_obj_mask, [1,2]) )
    loss_conf_noobj = LAMDA_NOOBJ*tf.reduce_mean(tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_noobj_mask, [1,2]))
    loss_class = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.square(true_box_class - tf.nn.softmax(pred_box_class)), -1) * class_mask, [1,2]))

    loss = loss_xy + loss_wh + loss_conf_obj + loss_conf_noobj + loss_class

    return loss_xy, loss_wh, loss_conf_obj, loss_conf_noobj, loss_class, loss


# In[10]:


if command == 'train':
    # remove model_saver folder
    shutil.rmtree(model_saver, ignore_errors=True)
    shutil.rmtree("../checkpoint", ignore_errors=True)


# In[11]:

# model graph
create_cnn_model_graph()

# the placeholders will be used for the feed_dict param of tf.Session.run()
X = g.get_tensor_by_name('CNN/input:0')
h = g.get_tensor_by_name('CNN/output:0')
y = tf.placeholder(tf.float32, [BATCH_SIZE, OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_DEPTH], name='ground_truth')

# learning rate
learning_rate = 0.0001

# loss 
loss_xy, loss_wh, loss_conf_obj, loss_conf_noobj, loss_class, loss = loss_op(h, y)

#loss_op = loss_op(h, y, training_class_only)

# optimizer corr
optimal = tf.train.AdamOptimizer(learning_rate, name='adam_optimizer')
# increment global_step at each step.
train_op = optimal.minimize(loss, name='optimal_min')

# evaluate model
#argmax_pred, argmax_true = (0.,0.)
#correct_prediction = tf.equal(argmax_pred, argmax_true)
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[12]:


# tao summary cua cac monitor de quan sat cac bien
tf.summary.scalar('loss_op', loss)
#tf.summary.scalar('learning_rate', learning_rate)
#tf.summary.scalar('accuracy', accuracy)

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
    return data[dict_field_name][batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,train_len)]

# In[14]:
def train():
    sess.run(init)
    # Training cycle
    epoch = 0;
    while True:
        total_acc = 0;
        total_cost = 0;
        total_batch =  train_len // BATCH_SIZE
        #total_batch = 1;
        for batch in range(total_batch):
            # lay batch tiep theo
            batch_input = get_batch('X_train', batch) 
            batch_label = get_batch('y_train', batch)
            # chay train_op, loss_op, accuracy
            _, lxy, lwh, lobj, lnoobj, lclass, cost, summary = sess.run([train_op, loss_xy, loss_wh, loss_conf_obj, loss_conf_noobj, loss_class, loss, merged_summary_op], 
                        feed_dict={
                            X: batch_input, 
                            y: batch_label
                        })
            #total_acc += acc
            total_cost += cost
            # Write logs at every iteration
            tf_writer.add_summary(summary, epoch * total_batch + batch)
            print("""---Batch: {}, loss {}, 
                loss_xy     = {}
                loss_wh     = {}
                loss_obj    = {}
                loss_noobj  = {}
                loss_class  = {}\n""".format(batch, cost, lxy, lwh, lobj, lnoobj, lclass))
            #print("---Batch:" + ('%04d,' % (batch)) + ("cost={%.9f}" % (cost)) + "\n")
                
        epoch += 1;
        continue;  
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
    eval_epoch_num = 0
    saver.restore(sess, model_saver + 'nn_model_%04d.ckpt'%(eval_epoch_num))
    avg_acc = 0.
    total_batch =  test_len // BATCH_SIZE
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