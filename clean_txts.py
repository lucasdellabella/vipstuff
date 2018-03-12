# removes txts that have no matching png

import numpy as np
import os
import shutil
import random
from PIL import Image

def clean_txts(dir_name):
	# 1. list files in directory
	files = list(filter(lambda x: x[-4:] == '.png', os.listdir(dir_name)))
	os.makedirs('tmp')
	# move all pngs and corresponding txts to tmp dir
	for f in files:
	# 	3. move file to a new valid directory
		shutil.move(dir_name + f, 'tmp')
		shutil.move(dir_name + f.split('.')[0] + '.txt', 'tmp')
	# delete old dir
	shutil.rmtree(dir_name)
	# move all images that have txts back into new dir
	shutil.move('tmp', dir_name)

clean_txts('.\\train\\')
clean_txts('.\\test\\')
