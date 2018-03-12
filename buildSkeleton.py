import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import os, fnmatch
from random import random

d1,d2 = 500,500
focal = -533
cx,cy = 112,112
fix_grampaLucas_shit_scaler = 4
base_dir = '.\\train\\'

"Returns scale amount necessary to put middle point in the center of the screen"
def scaleToCenter(x, y):
    dist_to_cx = cx - x
    dist_to_cy = cy - y
    return (1.0 * cx / x, 1.0 * cy / y)

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
        x_scaler, y_scaler = scaleToCenter(pred[2,0], pred[2,1])
        x_scaler, y_scaler = 1,1

        # comment out if the scale of our predictions is better
        for i, (x, y) in enumerate(pred):
            #denoiser = random() * 2 ** 2
            del_x = x - pred[2, 0]# + denoiser
            del_y = y - pred[2, 1]# + denoiser
            pred[i] = [pred[i, 0] + del_x, pred[i, 1] + del_y]
        

        plt.plot(uvd[:,0], uvd[:,1], '.')
        plt.plot(pred[:,0]*x_scaler, pred[:,1]*y_scaler, '.')
        plt.draw()
        plt.pause(0.15)


makeSkeleton()
