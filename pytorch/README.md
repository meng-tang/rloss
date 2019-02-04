# Pytorch version of regularized segmentation loss

## build python extension module

The implementation of DenseCRF loss depends on fast bilateral filtering, which is provided in C++. Use SWIG to wrap C++ for python and then build the python module of bilateral filtering.
```
cd wrapper/bilateralfilter
swig -python -c++ bilateralfilter.i
python3 setup.py build
```
## denseCRF loss in pytorch

The source code for the denseCRF loss layer is DenseCRFLoss.py. Declare such a loss layer as follows:
```
losslayer=DenseCRFLoss(weight=weight, sigma_rgb=sigma_rgb, sigma_xy=sigma_xy, scale_factor=scale_factor)
```
Here we specify loss weight, Gaussian kernel bandwidth (for RGB and XY), and an optional scale_factor (used to downscale output segmentation so that forward and backward for DenseCRF loss is faster).

The input to the denseCRF loss layer includes image (in the range of [0-255]), segmentation (output of softmax) and a binary tensor specifying region of interest for the regularized loss (e.g. not interested for padded region).
```
losslayer(image,segmentation,region_of_interest)
```
## how to run the code
Using the following script, we specify to train with densecrf loss. The weight of densecrf loss is 1.5e-9. The bandwidths of Gaussian kernels are 15 and 100 for RGB and XY respectively. Optionally, the output segmentation is downscaled by 0.5 (rloss-scale).
```
python3 train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 50 
--batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 2 
--densecrfloss 1.5e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100
```
(set the path of dataset in mypath.py. For example, the path for pascal should have three subdirectories called "JPEGImages", "SegmentationClassAug", and "pascal_2012_scribble" containing RGB images, groundtruth, and scribbles respectively.)
## acknowledgement

The code here is built on <a href="https://github.com/jfzhang95/pytorch-deeplab-xception" atl="#">pytorch-deeplab-xception</a>. We alto utilized the efficient c++ implementation of permutohedral lattice from <a href="https://github.com/torrvision/crfasrnn" alt="#">CRF-as-RNN</a>. <a href="http://fangyuliu.me" alt="#">Fangyu Liu</a> from the University of Waterloo helped tremendously in releasing this pytorch version.
