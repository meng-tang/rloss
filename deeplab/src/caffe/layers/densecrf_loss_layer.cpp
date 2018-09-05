// DenseCRF loss layer

#include <vector>
#include <stdio.h>
#include <float.h> 
#include <math.h>

#include "caffe/layers/densecrf_loss_layer.hpp"
#include <omp.h>

namespace caffe{

template <typename Dtype>
DenseCRFLossLayer<Dtype>::~DenseCRFLossLayer() {
  delete AS;
  delete cropping_batch;
  
}

template <typename Dtype>
void DenseCRFLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
  // blob size
  N = bottom[0]->shape(0);
  C = bottom[1]->shape(1);
  H = bottom[0]->shape(2);
  W = bottom[0]->shape(3);
  // Gaussian kernel parameters
  DenseCRFLossParameter densecrf_loss_param = this->layer_param_.densecrf_loss_param();
  bi_xy_std_ = densecrf_loss_param.bi_xy_std();
  bi_rgb_std_ = densecrf_loss_param.bi_rgb_std();
  printf("LayerSetup\n");
  AS = new Blob<Dtype>(N,C,H,W);
  cropping_batch = new Dtype[N*H*W];
  permutohedrals = vector<Permutohedral>(N);
  
  const int maxNumThreads = omp_get_max_threads();
  printf("Maximum number of threads for this machine: %i\n", maxNumThreads);
  printf("Total number of cores in the CPU: %ld\n", sysconf(_SC_NPROCESSORS_ONLN));
  omp_set_num_threads(std::min(maxNumThreads,30));
}
      
template <typename Dtype>
void DenseCRFLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*> & bottom,
                              const vector<Blob<Dtype>*> & top)
{
  top[0]->Reshape(1,1,1,1);
}

template <typename Dtype>
void DenseCRFLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*> & bottom,
    const vector<Blob<Dtype>*> & top)
{
  const Dtype* images = bottom[0]->cpu_data();
  const Dtype* segmentations = bottom[1]->cpu_data();
  
  // cropping
  caffe_set(N*H*W, Dtype(1.0), cropping_batch);
  for(int n=0;n<N;n++){
    const Dtype * image = images + H*W*3*n;
    for(int h=0;h<H;h++){
      for(int w=0;w<W;w++){
        if(((int)(image[0*H*W + h*W + w])==0)&&((int)(image[1*H*W + h*W + w])==0)&&((int)(image[2*H*W + h*W + w])==0))
          cropping_batch[n*H*W + h*W + w] = Dtype(0);
      }
    }
  }
  
  //printf("DenseCRF forward\n");
  //printf("bi std %.2f %.2f\n", bi_xy_std_, bi_rgb_std_);
  
  
  
  // initialize permutohedrals
  #pragma omp parallel for
  //printf("Permutohedral Number of threads: %d\n", omp_get_num_threads());
  for(int n=0;n<N;n++){
    const Dtype * image = images + H*W*3*n;
    initializePermutohedral((float *)image, W, H, bi_rgb_std_, bi_xy_std_, permutohedrals[n]);
    //printf("Permutohedral Thread %d: %d\n", omp_get_thread_num(), n);
  }
  
  Dtype densecrf_loss = Dtype(0);
  #pragma omp parallel for reduction(+: densecrf_loss)
  for(int n=0;n<N;n++){

    const Dtype * image = images + H*W*3*n;
    
    //printf("size of Dtype %d\n", sizeof(new Dtype[1]));
    //exit(-1);
    densecrf_loss = densecrf_loss + Compute_DenseCRF(image, segmentations + H*W*C*n, AS->mutable_cpu_data() + n*C*H*W, cropping_batch + n*H*W, permutohedrals[n]);
    //exit(-1);
    //printf(" Thread %d: %d\n", omp_get_thread_num(), n);
  }
  densecrf_loss = densecrf_loss / N;
  Dtype* top_data = top[0]->mutable_cpu_data();
  top_data[0] = densecrf_loss;
}

template <typename Dtype>
void DenseCRFLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*> & top,
                                   const vector<bool> & propagate_down,
                                   const vector<Blob<Dtype>*> & bottom)
{
    
  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to image inputs.";
  }
  if (propagate_down[1]) {
    //printf("DenseCRF backward\n");
    const Dtype* images = bottom[0]->cpu_data();
    const Dtype* segmentations = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[1]->mutable_cpu_diff();
    
    #pragma omp parallel for
    for(int n=0;n<N;n++){

      const Dtype * image = images + H*W*3*n;
      Gradient_DenseCRF(image, segmentations + H*W*C*n, bottom_diff + H*W*C*n, AS->cpu_data()+n*C*H*W, cropping_batch + n*H*W);
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / N;
    caffe_scal(bottom[1]->count(), loss_weight, bottom_diff);
    
    //printf("loss_weight is %.2f\n", loss_weight);
  }
  
}

template <typename Dtype>
Dtype DenseCRFLossLayer<Dtype>::Compute_DenseCRF(const Dtype * image, const Dtype * segmentation, Dtype * AS_data,  const Dtype * cropping, Permutohedral & permutohedral)
{
  Dtype densecrf_loss = 0;
  // segmentation in cropping, zero outside
  Dtype * temp = new Dtype[H*W];
  for(int c=0;c<C;c++){
    caffe_mul(H*W, segmentation+c*W*H, cropping, temp);
    //printf("sum of segmentation of this channel %.8f\n", caffe_cpu_dot(H*W, temp, allones->cpu_data()));
    permutohedral.compute((float *)AS_data + c*W*H, (float *)temp, 1);
               
    Dtype SAS   = caffe_cpu_dot(H*W, temp, AS_data + c*W*H);
    densecrf_loss -= SAS;
    if(isnan(SAS))
      LOG(FATAL) << this->type()
               << " Layer SAS: "<< SAS <<std::endl;
  }
  delete [] temp;
  return densecrf_loss;
}

template <typename Dtype>
void DenseCRFLossLayer<Dtype>::Gradient_DenseCRF(const Dtype * image, const Dtype * segmentation, Dtype * gradients, const Dtype * AS_data,  const Dtype * cropping){
  
  caffe_set(H*W*C, Dtype(0), gradients);
  // segmentation in cropping, zero outside
  Dtype * temp = new Dtype[H*W];
  for(int c=0;c<C;c++){
    caffe_mul(H*W, segmentation+c*W*H, cropping, temp);
    for(int i=0;i<H * W;i++){
      gradients[H*W*c + i] = - 2 * AS_data[i+c*H*W];
      if(isnan(gradients[H*W*c + i]))
        LOG(FATAL) << this->type()
               << " Layer gradient is nan!"<<std::endl;
    }    
    caffe_mul(H*W, gradients + c*H*W, cropping, gradients + c*H*W);
  }
  delete temp;
  for(int c=0;c<C;c++)
    caffe_mul(H*W, gradients + c*H*W, cropping, gradients + c*H*W);
  //printf("end of gradient_densecrf\n");
}

#ifdef CPU_ONLY
STUB_GPU(DenseCRFLossLayer);
#endif

INSTANTIATE_CLASS(DenseCRFLossLayer);
REGISTER_LAYER_CLASS(DenseCRFLoss);

} // namespace caffe
