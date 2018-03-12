import numpy as np
import os
from PIL import Image

# Create X Matrices
for d in ['train', 'test', 'valid']:
  print('entering X ' + d)
  data_arr = []
  for f in sorted(os.listdir(d)):
    if f[-4:] == '.png':
      image = np.array(Image.open(d + '\\' + f))
      data_arr.append(image)
  np.save('matrixized-data\\' + 'X_' + d + '.npy', np.array(data_arr))
  
# Create y Matrices
for d in ['train', 'test', 'valid']:
  print('entering y ' + d)
  label_arr = []
  for f in sorted(os.listdir(d)):
    if f[-4:] == '.txt':
      label_arr.append(np.loadtxt(d + '\\' + f).flatten())
  np.save('matrixized-data\\' + 'y_' + d + '.npy', np.array(label_arr))

