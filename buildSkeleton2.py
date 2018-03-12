import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import os, fnmatch
from random import random

d1,d2 = 500,500
focal = -533
cx,cy = 112,112
base_dir = '.\\train\\'

# load and show augmented testing data
print('load and show augmented testing data')
def makeSkeleton():
    print("came in ")

    pngarr = fnmatch.filter(os.listdir(base_dir),'*.png')

    for n, png in enumerate(pngarr):
        plt.clf()
        depth = np.array(PIL.Image.open(base_dir + png))
        plt.imshow(depth, cmap = plt.get_cmap('gray'), vmin = d1, vmax = d2)

        uvd = np.loadtxt(base_dir + png.split('.')[0] + '.txt')
        pred = np.loadtxt('.\\train_prediction_txts\\' + str(n + 1).zfill(5) + ".prediction.txt")

        # comment out if the scale of our predictions is better
        for i, (x, y) in enumerate(pred):
            del_x = x - pred[2, 0]
            del_y = y - pred[2, 1]
            pred[i] = [pred[i, 0] + del_x, pred[i, 1] + del_y]
        
        for (x, y, color) in zip(uvd[:,0], uvd[:,1], ['b','g','r','c','m']):
            plt.plot(x, y, '.', c=color)

        for (x, y, color) in zip(pred[:,0], pred[:,1], ['b','g','r','c','m']):
            plt.plot(x, y, '.', c=color)

        #plt.plot(uvd[:,0], uvd[:,1], '.')
        #plt.plot(pred[:,0], pred[:,1], '.')
        plt.draw()
        plt.pause(1)


makeSkeleton()
