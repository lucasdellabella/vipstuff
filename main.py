import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

# param
d1,d2 = 500,1000
focal = -533
cx,cy = 320,240

# camera projection
def xyz2uvd(jnt):
    if jnt.ndim == 2:
        u = jnt[:,0] / jnt[:,2] * focal + cx
        v = -jnt[:,1] / jnt[:,2] * focal + cy
        z = -jnt[:,2]
        return np.concatenate((u,v,z)).reshape(3,-1).T
    if jnt.ndim == 1:
        u = jnt[0] / jnt[2] * focal + cx
        v = -jnt[1] / jnt[2] * focal + cy
        z = -jnt[2]
        return np.array([u,v,z])

# load training image
plt.figure(1)

# load and show training data
print 'load and show training data'
for i in range(4751):
    plt.clf()
    # load depth image and show
    print 'train/%05i.png'%(i+1)
    depth = np.array(PIL.Image.open('train/%05i.png'%(i+1)))
    plt.imshow(depth, cmap = plt.get_cmap('gray'), vmin = d1, vmax = d2)
    # load xyz of 5 joints
    xyz = np.loadtxt('train/%05i.txt'%(i+1))
    uvd = xyz2uvd(xyz)
    plt.plot(uvd[:,0], uvd[:,1], '.')
    plt.draw()
    plt.pause(0.01)
    
# load and show augmented testing data
print 'load and show augmented testing data'
for i in range(6630):
    plt.clf()
    # load depth image and show
    print 'test/%05i.png'%(i+1)
    depth = np.array(PIL.Image.open('test/%05i.png'%(i+1)))
    plt.imshow(depth, cmap = plt.get_cmap('gray'), vmin = d1, vmax = d2)
    # load xyz of 5 joints
    xyz = np.loadtxt('test/%05i.txt'%(i+1))
    uvd = xyz2uvd(xyz)
    plt.plot(uvd[:,0], uvd[:,1], '.')
    plt.draw()
    plt.pause(0.01)
    
# load and show testing sequence
print 'load and show testing sequence'
for i in range(1326):
    plt.clf()
    # load depth image and show
    print 'test_seq/%05i.png'%(i+1)
    depth = np.array(PIL.Image.open('test_seq/%05i.png'%(i+1)))
    plt.imshow(depth, cmap = plt.get_cmap('gray'), vmin = d1, vmax = d2)
    # load xyz of 5 joints
    xyz = np.loadtxt('test_seq/%05i.txt'%(i+1))
    uvd = xyz2uvd(xyz)
    plt.plot(uvd[:,0], uvd[:,1], '.')
    plt.draw()
    plt.pause(0.01)
