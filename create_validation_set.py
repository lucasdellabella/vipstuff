import numpy as np
import os
import shutil
import random
from PIL import Image

#create the validation set from the test set b/c test set has more data

# random sample w/out replacement on list of files from the test directory
# 1. list files in directory
files = list(filter(lambda x: x[-4:] == '.txt', os.listdir('.\\test')))
# 2. sample 1k items
valid_set = random.sample(files, 1000)

try:
	os.makedirs('valid')
	for sample in valid_set:
# 	3. move file to a new valid directory
	shutil.move('.\\test\\' + sample, 'valid')
	shutil.move('.\\test\\' + sample.split('.')[0] + '.png', 'valid')
except FileExistsError:
	print('Validation set already exists, exiting...')