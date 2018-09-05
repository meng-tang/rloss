from sys import exit
import sys
sys.path.insert(0,'/home/m62tang/rloss/deeplab/python')
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()


solver = caffe.SGDSolver('pascal_scribble/config/deeplab_largeFOV/solverwithdensecrfloss_train.prototxt')
solver.net.copy_from('pascal_scribble/model/deeplab_largeFOV/train_iter_9000.caffemodel')
training_net = solver.net

solver.step(1)

print 'densecrf loss', training_net.blobs['densecrf_loss'].data
print training_net.blobs['fc8_interp_softmax'].data.shape


prob = training_net.blobs['fc8_interp_softmax'].data
scores = training_net.blobs['fc8_interp']


print type(prob)
print prob[0,0,:,:].shape
print prob[0,0,:,:].dtype

# RGB image
imgblob = training_net.blobs['data'].data
imgblob = imgblob[0,:,:,:]
(C, H, W) = imgblob.shape
img = np.zeros((H, W, C), dtype=np.uint8);

img[:,:,2] = imgblob[0,:,:] + 104.008
img[:,:,1] = imgblob[1,:,:] + 116.669
img[:,:,0] = imgblob[2,:,:] + 122.675

scores_diff = scores.diff

print np.amax(scores_diff[0,1,:,:])
print np.amin(scores_diff[0,1,:,:])

prob0 = prob[0,0,:,:]

prob_diff = training_net.blobs['fc8_interp_softmax'].diff

shrink_scores_diff = training_net.blobs['fc8_pascal_scribble'].diff

for i in range(21):
    fig=plt.figure()
    plt.imshow(prob[0,i,:,:], vmin=0, vmax=1);
    plt.colorbar()
    plt.axis('off')
    plt.savefig('visualization/' + str(i) + 'prob' + '.png')
    plt.show(block=False)
    plt.close(fig)

for i in range(21):
    fig = plt.figure()
    plt.imshow(scores_diff[0,i,:,:],cmap="hot");
    plt.colorbar()
    plt.axis('off')
    plt.savefig('visualization/' + str(i) + 'scorediff' + '.png')
    plt.show(block=False)
    plt.close(fig)
    
plt.figure()
plt.imshow(img, cmap='gray')
plt.show(block=True)

