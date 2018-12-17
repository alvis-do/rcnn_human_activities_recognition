
import cv2
from config import params


video_src_path = params['VIDEO_SRC_PATH']

cap = cv2.VideoCapture('./person01_boxing_d4_uncomp.avi')
cap.set(cv2.CAP_PROP_POS_FRAMES, 310)
_, frame = cap.read()
# print('frame {} -- {}'.format(310, '/boxing/person01_boxing_d4_uncomp.avi'))
print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print(frame.shape)
print(frame)
print("--- check frame ---")