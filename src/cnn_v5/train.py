#!/usr/bin/env python3
'''
This script will randomize weights and train the tiny yolo from scratch.
'''
from datahandler import get_batch, gridcell_to_boxes, save_boxes_to_file
from mAP import calc_mAP
import tensorflow as tf
import numpy as np
import os
import sys
import shutil
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

saver = tf.train.import_meta_graph("./graph/tiny-yolo.ckpt.meta")
with tf.Session() as sess:
    saver.restore(sess, "./graph/tiny-yolo.ckpt")
    g = sess.graph
    with g.name_scope("TRAINER"):
        X = g.get_tensor_by_name("YOLO/input:0")
        batch_size, height, width, in_channels = X.get_shape().as_list()
        classes = 4
        out_height = height//32
        out_width = width//32
        out_channels = 5+classes
        h1 = g.get_tensor_by_name("YOLO/output1:0")
        Y1 = tf.placeholder(shape = (batch_size, out_height, out_width, out_channels), dtype = tf.float32, name = "groundtruth1")
    
        #loss
        h = tf.reshape(h1, [batch_size * out_height * out_width, out_channels], name='h')
        Y = tf.reshape(Y1, [batch_size * out_height * out_width, out_channels], name='Y')

        Lcoord = 5
        Lnoobj = 0.5
        loss_xy = Lcoord*tf.reduce_mean(Y[:,0]*((h[:,1] - Y[:,1])**2 + (h[:,2] - Y[:,2])**2))
        loss_wh = Lcoord*tf.reduce_mean(Y[:,0]*((h[:,3]**0.5 - Y[:,3]**0.5)**2+(h[:,4]**0.5 - Y[:,4]**0.5)**2))
        loss_obj = (-1)*tf.reduce_mean(tf.tile(Y[:,0:1], (1, classes))*(Y[:,5:]*tf.log(h[:,5:]) + (1-Y[:,5:])*tf.log(1-h[:,5:])))
        loss_noobj = (-1*Lnoobj)*tf.reduce_mean(tf.tile(1-Y[:,0:1], (1, classes))*(Y[:,5:]*tf.log(h[:,5:]) + (1-Y[:,5:])*tf.log(1-h[:,5:])))
        loss_p = (-1)*tf.reduce_mean(tf.tile(Y[:,0:1], (1, classes))*tf.log((tf.tile(h[:,0:1], (1, classes)) * Y[:,5:])) + (1-tf.tile(Y[:,0:1], (1, classes)))*tf.log(1-(tf.tile(h[:,0:1], (1, classes)) * Y[:,5:])))

        loss = loss_xy + loss_wh + loss_obj + loss_noobj + loss_p

        optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3)
        trainer = optimizer.minimize(loss, name = "trainer")



    if os.path.exists("./train_graph"):
            shutil.rmtree("./train_graph")
    os.mkdir("./train_graph")

    train_writer = tf.summary.FileWriter("./train_graph", g)
    saver = tf.train.Saver()
    tf.summary.histogram("loss", loss)
    merge = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())

    input_size = height

    #for batch in shuffle(batch_size, input_size):
    epoch = 0
    while epoch < 1000:
        loss_arr = np.array([])
        mAP_arr = np.array([])
        for batch in get_batch(batch_size, input_size):
            step, Xp, Y1p, from_idx = batch
            if step == 0:
                time.sleep(1)
                continue
            debugger = tf.logical_or(tf.is_nan(loss), tf.is_inf(loss))

            while (1):
                d, l = sess.run([debugger, loss], 
                    feed_dict = {X:Xp, Y1:Y1p})
                if (not d):
                    break
                else:
                    print("Re-random variables!")
                    sess.run(tf.global_variables_initializer())
            summary, _ , lossp, lxy, lwh, lobj, lnoobj, lp, y_pred = sess.run(
                        [merge, trainer, loss, loss_xy, loss_wh, loss_obj, loss_noobj, loss_p, h1],
                        feed_dict = {X: Xp, Y1: Y1p})

            # mAP y_pred
            for i in batch_size:
                file_name = str(from_idx + i)
                y_pred_i = gridcell_to_boxes(y_pred[i])
                save_boxes_to_file(y_pred_i, True, file_name)
                y_true_i = gridcell_to_boxes(Y1p[i])
                save_boxes_to_file(y_true_i, False, file_name)

            b_mAP = calc_mAP()
            shutil.rmtree("./ground-truth", ignore_errors=True)
            shutil.rmtree("./predicted", ignore_errors=True)

            mAP_arr.append(b_mAP)
            loss_arr.append(lossp)

            print("""Step {} : loss {}
                            loss_xy     = {}
                            loss_wh     = {}
                            loss_obj    = {}
                            loss_noobj  = {}
                            loss_p      = {}\n""".format(step - 1, lossp, lxy, lwh, lobj, lnoobj, lp), end="\n")

            train_writer.add_summary(summary, step - 1)

        print("Epoch {}: loss_avg {}; mAP_avg {}\n".format(epoch, np.mean(loss_arr), np.mean(mAP_arr)))
        print("===============================================================")

        epoch += 1

        if (epoch % 50 == 0):
            saver.save(sess, "./train_graph/tiny-yolo-{}.ckpt".format(epoch))

    print("Training done!!!")


