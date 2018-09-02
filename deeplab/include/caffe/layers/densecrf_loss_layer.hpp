#ifndef CAFFE_DENSECRF_LOSS_LAYER_HPP_
#define CAFFE_DENSECRF_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

#include <cstddef>
#include "caffe/util/filterrgbxy.hpp"

namespace caffe{

template <typename Dtype>

class DenseCRFLossLayer : public LossLayer<Dtype>{
 public:
  virtual ~DenseCRFLossLayer();
  explicit DenseCRFLossLayer(const LayerParameter & param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*> & bottom,
      const vector<Blob<Dtype>*> & top);
  virtual inline const char* type() const { return "DenseCRFLoss";}
  virtual inline int ExactNumBottomBlobs() const {return 2;}
  virtual inline int ExactNumTopBlobs() const {return 1;}
  
 protected:
  Dtype Compute_DenseCRF(const Dtype * image, const Dtype * segmentation, Dtype * AS_data, const Dtype * ROI, Permutohedral & permutohedral); 
  void Gradient_DenseCRF(const Dtype * image, const Dtype * segmentation, Dtype * gradients, const Dtype * AS_data, const Dtype * ROI); 
  virtual void Forward_cpu(const vector<Blob<Dtype>*> & bottom,
      const vector<Blob<Dtype>*> & top);  
  virtual void Backward_cpu(const vector<Blob<Dtype>*> & top,
      const vector<bool> & propagate_down, const vector<Blob<Dtype>*> & bottom);
  
  int N, C, H, W;
  
  float bi_xy_std_;
  float bi_rgb_std_;
  Blob<Dtype> * AS; //A * S
  
  Dtype * cropping_batch;
  vector<Permutohedral> permutohedrals;

  
};

} // namespace caffe

#endif // CAFFE_DENSECRF_LOSS_LAYER_HPP_
