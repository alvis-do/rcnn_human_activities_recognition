import numpy as np
import os
import cv2
from config import params


video_src_path = params['VIDEO_SRC_PATH']
video_dict_path = 'video_dict.npy'
video_train_path = 'video_train.txt'
video_test_path = 'video_test.txt'


video_step = 5 # Cu "video_step" frame thi lay 1 frame

def get_info(videos):
	v_info = []
	for v in videos:
		cap = cv2.VideoCapture(v)
		l_ = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		rp = v.replace(video_src_path,'')
		c_ = params['CLASSES'].index(rp.split('/', 2)[1])
		f_from = 0
		f_to = 0
		video_step_tmp = video_step
		while f_to < l_:
			if video_step_tmp == 0:
				break
			f_to = min(f_from + (params['N_FRAMES']) * video_step_tmp, l_)
			f_ = list(range(f_from, f_to, video_step_tmp))
			if len(f_) < params['N_FRAMES']:
				if (f_from > 0): 
					f_from = f_to - (params['N_FRAMES'] * video_step_tmp)
					f_ = list(range(f_from, f_to, video_step_tmp))
				else:
					video_step_tmp -= 1
					f_from = 0
					f_to = 0
					continue
			f_from = f_to
			v_info.append([[rp], f_, [l_], [c_]])
	return v_info

def generate_data_dict():
	with open(video_train_path) as f:
		train_v_info = get_info([(video_src_path + x.strip()) for x in f.readlines()])

	with open(video_test_path) as f:
		test_v_info = get_info([(video_src_path + x.strip()) for x in f.readlines()])


	# Shuffle
	np.random.shuffle(train_v_info)

	# Split data set
	#data_len = len(v_info)
	#train_len = int(data_len * 0.8)

	data_dict = {'train_set': train_v_info, 'test_set': test_v_info}
	# v_info = {"video_path": relate_path_, "frames": frames_, "frame_count": length_, "label": class_}
	np.save(video_dict_path, data_dict)


def get_dataset():
	if not os.path.exists(video_dict_path):
		generate_data_dict()
		print("Generate data dictionary completed at {} !!!".format(video_dict_path))

	data = np.load(video_dict_path).item()
	return data['train_set'], data['test_set']

def get_batch(dataset, batch_size):
	step = 0
	while (1):
		if (step == 0):
			yield step, None
		else:
			yield step, clips
			del clips, from_, to_
            
		to_ = min((step + 1) * batch_size, len(dataset))
		from_ = (step * batch_size) if to_ <= len(dataset) else (to_ - batch_size)

		clips = dataset[from_:to_]
		#clips = dataset[to_ - batch_size:to_]
		step += 1


def get_clip(clips, stride=1): 
	frames = []
	labels = []
	for clip in clips:
		# clip is [video_relate_path, frames, video_length, video_label]
		clip_label = [0]*params['N_CLASSES']
		clip_label[clip[3][0]] = 1
		clip_frames = []
		cap = cv2.VideoCapture(video_src_path + clip[0][0])
		#print(clip[1])
		for i, v in enumerate(clip[1]):
			if (i % stride == 0): 
				cap.set(cv2.CAP_PROP_POS_FRAMES, v)
				_, frame = cap.read()
				frame = cv2.resize(frame, (params['INPUT_WIDTH'], params['INPUT_HEIGHT']), interpolation = cv2.INTER_CUBIC)
				clip_frames.append(frame)
		

		frames.append(clip_frames)
		labels.append(clip_label)

	return np.array(frames), np.array(labels)

def split_valid_set(train_set):
	length = len(train_set)
	valid_len = int(length * 0.2)
	pivot = length - valid_len
	
	valid = train_set[pivot:]
	train = train_set[:pivot]
	
	return  train, valid


def get_frames_from_video(video_path, stride=2):
	frame_idx = 0
	cap = cv2.VideoCapture(video_src_path + video_path.strip())
	count_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	# print(video_src_path + video_path.strip())
	while 1:
		if frame_idx == 0:
			yield None
		else:
			yield frames
		frames = []
		i = frame_idx + 2
		j = 0
		_stride = stride
		while 1:
			if j == params['N_FRAMES']: 
				frame_idx = i - 1
				break
			if i%_stride == 0:
				# print("i: {}".format(i))
				if i >= count_frame:
					# print("stride: {}".format(_stride))
					_stride -= 1
					if _stride <= 0:
						break
					frames = []
					i = frame_idx
					j = 0
					continue
				cap.set(cv2.CAP_PROP_POS_FRAMES, i)
				_, frame = cap.read()
				frame = cv2.resize(frame, (params['INPUT_WIDTH'], params['INPUT_HEIGHT']), interpolation = cv2.INTER_CUBIC)
				frames.append(frame)
				j += 1
			i += 1
		# print("lenframe: {}".format(len(frames)))
		if len(frames) < params['N_FRAMES'] and len(frames) > 8:
			frames.extend(np.zeros((params['N_FRAMES'] - len(frames), params['INPUT_WIDTH'], params['INPUT_HEIGHT'], params['INPUT_CHANNEL']), dtype=np.float32))
			break
		elif len(frames) == params['N_FRAMES']:
			pass
		else:
			frames = []
			break


def get_video_test():
	def tmp_func(r_path):
		p = video_src_path + r_path.strip()
		rp = p.replace(video_src_path,'')
		c = params['CLASSES'].index(rp.split('/', 2)[1])
		label = [0]*params['N_CLASSES']
		label[c] = 1
		return r_path, label

	with open(video_test_path) as f:
		records = [tmp_func(x) for x in f.readlines()]

	return records


if __name__ == "__main__":
	generate_data_dict()


	# test function
	
	# for frames in get_frames_from_video("/golf_swing/v_golf_08/v_golf_08_01.mpg"):
	# 	if frames == None:
	# 		print("continue")
	# 		continue
	# 	if len(frames) <= 0:
	# 		break
	# 	print("-----------")
	# print("done")