# merge and then re seperate training and testing sets

# pull all the images and corresponding txt files into one collection
#   need to differentiate the two, prepend a or b to files from train or test to prevent collision

# randomly select again

import numpy as np
import os
import shutil
import random
from PIL import Image

TEST_PCT = 0.1

# should gives the tuples we want regardless of what the extension on the file is
def file_name_tuple(file_name):
	if file_name[-4:] == '.png' or file_name[-4:] == '.txt':
		file_name = file_name[:-4]
	return (file_name + '.png', file_name + '.txt')

# Move test files into train directory so we can sample collectively
test_files = list(filter(lambda x: x[-4:] == '.txt', os.listdir('.\\test')))
test_files = list(map(file_name_tuple, test_files))
for png_name, txt_name in test_files:
	shutil.move('.\\test\\' + png_name, 'train\\z' + png_name)
	shutil.move('.\\test\\' + txt_name, 'train\\z' + txt_name)

all_txt_names = list(filter(lambda x: x[-4:] == '.txt', os.listdir('.\\train')))
all_sample_names = list(map(file_name_tuple, all_txt_names))

# move 10% over to the other file
test_set = random.sample(all_sample_names, int(len(all_sample_names) * TEST_PCT))
for png_name, txt_name in test_set:
	shutil.move('.\\train\\' + png_name, 'test\\' + png_name)
	shutil.move('.\\train\\' + txt_name, 'test\\' + txt_name)

# Test directory should now have the proper percentage of samples.