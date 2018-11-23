import numpy as np
import pandas as pd
import cv2

origin_data_path = "../../dataset/my_dict.npy"

origin_data = np.load(origin_data_path).item()

X_train = origin_data['X_train']
X_test = origin_data['X_test']
y_train = origin_data['y_train']
y_test = origin_data['y_test']

input_size = None
classes = 4
out_depth = 5 + classes

def resize(image):
	return cv2.resize(image, (input_size, input_size), interpolation = cv2.INTER_CUBIC)

def convert_y(y):
	out_height = input_size//32
	out_width = input_size//32
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

    Y = np.random.random((1, out_height, out_width, out_depth))
    Y[:, :, :, 0] = 1
    for obj in label:
        cls, x0, y0, w0, h0 = obj
        if x0<0 or x0>=1 or y0<0 or y0>=1: continue
        box = (w0, h0)
        x = int(out_width*x0)
        y = int(out_height*y0)
        Y[0, y, x, 0] = 1
        Y[0, y, x, 1] = x0
        Y[0, y, x, 2] = y0
        Y[0, y, x, 3] = w0
        Y[0, y, x, 4] = h0
        Y[0, y, x, 5:out_depth] = 0
        Y[0, y, x, 5+cls] = 1
    X[X<1e-37] = 1e-37
    Y[Y<1e-37] = 1e-37
    
    return Y

def get_batch(batch_size, input_size):
	input_size = input_size
	step = 0
    while (1):
        if (step == 0):
            yield step, None, None
        else:
            yield step, X, Y
            del X
            del Y
            
        idx = step*batch_size:min((step+1)*batch_size,X_train.shape[0])
        step += 1
		X = X_train[idx]
		X = np.array(list(map(resize, X)))
		Y = y_train[idx]
		Y = np.array(list(map(convert_y, Y)))
