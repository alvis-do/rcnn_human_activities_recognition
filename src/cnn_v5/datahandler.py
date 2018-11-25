import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf


origin_data_path = "./small_dataset_output_as_string.npy"

origin_data = np.load(origin_data_path).item()

X_train = origin_data['X_train']
X_test = origin_data['X_test']
y_train = origin_data['y_train']
y_test = origin_data['y_test']

input_size = 416
out_height = input_size//32
out_width = input_size//32
classes = 4
out_depth = 5 + classes

class_dict = ['shooting', 'biking', 'diving', 'golf']


def resize(image):
	return cv2.resize(image, (input_size, input_size), interpolation = cv2.INTER_CUBIC)

def convert_y(y):
    label = []
    for objtxt in y.split("\n"):
        if objtxt == "": continue
        cls, x0, y0, w0, h0 = objtxt.strip().split(" ")
        cls  = int(cls)
        x0   = float(x0)
        y0   = float(y0)
        w0   = float(w0)
        h0   = float(h0)
        label.append((cls, x0, y0, w0, h0))

    Y = np.random.random((out_height, out_width, out_depth))
    Y[:, :, 0] = 1
    for obj in label:
        cls, x0, y0, w0, h0 = obj
        if x0<0 or x0>=1 or y0<0 or y0>=1: continue
        box = (w0, h0)
        x = int(out_width*x0)
        y = int(out_height*y0)
        Y[y, x, 0] = 1
        Y[y, x, 1] = x0
        Y[y, x, 2] = y0
        Y[y, x, 3] = w0
        Y[y, x, 4] = h0
        Y[y, x, 5:out_depth] = 0
        Y[y, x, 5+cls] = 1
    #X[X<1e-37] = 1e-37
    Y[Y<1e-37] = 1e-37
    
    return Y

def get_batch(batch_size):
    step = 0
    while (1):
        if (step == 0):
            yield step, None, None, None
        else:
            yield step, X, Y, from_
            del X
            del Y
            del from_
            del to_
            
        to_ = min((step + 1) * batch_size, X_train.shape[0])
        from_ = (step * batch_size) if to_ <= X_train.shape[0] else (to_ - batch_size)

        X = X_train[from_ : to_]
        X = np.array(list(map(resize, X)))

        Y = y_train[from_ : to_]
        Y = np.array(list(map(convert_y, Y)))

        step += 1


def gridcell_to_boxes(gridcell):
    # with tf.Session() as sess:
    #     h = tf.reshape(gridcell, [out_width * out_height, out_depth])
    #     scores = h[:,0]
    #     y1 = h[:,2:3] - h[:,4:5] / 2
    #     x1 = h[:,1:2] - h[:,3:4] / 2
    #     y2 = h[:,2:3] + h[:,4:5] / 2
    #     x2 = h[:,1:2] + h[:,3:4] / 2
    #     boxes = tf.concat([y1,x1,y2,x2], axis=1)
    #     box_indices = tf.image.non_max_suppression(boxes, scores, 10).eval()

    #     boxes = []
    #     for i in box_indices:
    #         boxes.append(h[i])
    #     boxes = tf.concat(boxes, 0).eval()

    return boxes

def save_boxes_to_file(boxes, is_pred, file_name): # value for each box [o, x, y, w, h, c]
    if not os.path.exists("./predicted"):
        os.mkdir("./predicted")
    if not os.path.exists("./ground-truth"):
        os.mkdir("./ground-truth")
    with open("./{}/{}.txt".format("predicted" if is_pred else "ground-truth", file_name), "w+") as f:
        for b in boxes:
            if is_pred:
                f.write("{} {} {} {} {} {}\n".format(class_dict[b[5]], b[0], b[1], b[2], b[3], b[4]))
            else:
                f.write("{} {} {} {} {}\n".format(class_dict[b[5]], b[1], b[2], b[3], b[4]))
