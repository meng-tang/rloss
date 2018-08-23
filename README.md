## Regularized Losses (rloss) for Weakly-supervised CNN Segmentation

<span align="center"><img src="teaser.png" alt="" width="800"/></span>

To train CNN for semantic segmentation using weak-supervision (e.g. scribbles), we propose regularized loss framework.
The loss have two parts, partial cross-entropy (pCE) loss over scribbles and regularization loss e.g. DenseCRF.

If you use the code here, please cite the following paper.

"On Regularized Losses for Weakly-supervised CNN Segmentation"</br>
[Meng Tang](http://cs.uwaterloo.ca/~m62tang), [Federico Perazzi](https://fperazzi.github.io/), [Abdelaziz Djelouah](https://adjelouah.github.io/), [Ismail Ben Ayed](https://profs.etsmtl.ca/ibenayed/), [Christopher Schroers](https://www.disneyresearch.com/people/christopher-schroers/), [Yuri Boykov](https://cs.uwaterloo.ca/about/people/yboykov)</br>
In European Conference on Computer Vision (ECCV), Munich, Germany, September 2018.

### DenseCRF Loss ###
To include DenseCRF loss for CNN, add the following loss layer. It takes two bottom blobs, first RGB image and the second is soft segmentation distributions. We need to specify bandwidth of Gaussian kernel for XY (bi_xy_std) and RGB (bi_rgb_std).
```
layer {
  bottom: "image"
  bottom: "segmentation"
  propagate_down: false
  propagate_down: true
  top: "densecrf_loss"
  name: "densecrf_loss"
  type: "DenseCRFLoss"
  loss_weight: ${DENSECRF_LOSS_WEIGHT}
  densecrf_loss_param {
    bi_xy_std: 100
    bi_rgb_std: 15
  }
}
```
The implementation of this loss layer is in:
* <a href="deeplab/src/caffe/layers/densecrf_loss_layer.cpp" alt=#>deeplab/src/caffe/layers/densecrf_loss_layer.cpp</a>
* <a href="deeplab/include/caffe/layers/densecrf_loss_layer.hpp" alt=#>deeplab/include/caffe/layers/densecrf_loss_layer.hpp</a>
</br>which depend on fast high dimensional Gaussian filtering in
* <a href="deeplab/include/caffe/util/filterrgbxy.hpp" alt=#>deeplab/include/caffe/util/filterrgbxy.hpp</a>
* <a href="deeplab/src/caffe/util/filterrgbxy.cpp" alt=#>deeplab/src/caffe/util/filterrgbxy.cpp</a>
* <a href="deeplab/include/caffe/util/permutohedral.hpp" alt=#>deeplab/include/caffe/util/permutohedral.hpp</a>
</br>
This implementation is in CPU supporting multi-core parallelization. To enable, build with -fopenmp, see <a href="deeplab/Makefile" alt=#>deeplab/Makefile</a>.
