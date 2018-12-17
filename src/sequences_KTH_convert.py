import os, sys
import numpy as np

infile = 'sequences_KTH.txt'
outfile_train = 'sequences_KTH_train_result.txt'
outfile_test = 'sequences_KTH_test_result.txt'
suffixes = '_uncomp.avi'

labels = ['running', 'boxing', 'walking', 'handwaving']

p_train = range(1, 17)
p_test = range(17, 26)


with open(infile) as f:
	lines = f.readlines()
	files_train = []
	files_test = []
	for line in lines:
		if line == '\n':
			continue
		name = line.split('\t', 1)[0].strip()
		tmp = name.split('_', 2)
		people, label = tmp[0], tmp[1]
		p_i = int(people[6] + people[7])
		if label in labels:
			name += suffixes
			path = '/{}/{}'.format(label, name)
			if p_i in p_train:
				files_train.append(path)
			elif p_i in p_test:
				files_test.append(path)


	np.random.shuffle(files_train)



with open(outfile_train, 'w') as f:
	f.write('\n'.join(files_train))


with open(outfile_test, 'w') as f:
	f.write('\n'.join(files_test))