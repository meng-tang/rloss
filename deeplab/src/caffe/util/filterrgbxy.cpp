#include "caffe/util/filterrgbxy.hpp"

void initializePermutohedral(const float * image, int img_w, int img_h, float sigmargb, float sigmaxy, Permutohedral & lattice_){
    float * features = new float[img_w * img_h * 5];
	for( int j=0; j<img_h; j++ ){
		for( int i=0; i<img_w; i++ ){
		    int idx = j*img_w + i;
			features[idx*5+0] = float(i) / sigmaxy;
			features[idx*5+1] = float(j) / sigmaxy;
			features[idx*5+2] = float(image[0*img_w*img_h + idx]) / sigmargb;
			features[idx*5+3] = float(image[1*img_w*img_h + idx]) / sigmargb;
			features[idx*5+4] = float(image[2*img_w*img_h + idx]) / sigmargb;
		}
    }
    
    lattice_.init( features, 5, img_w * img_h );
    delete [] features;
}

void filterrgbxy(const float* image, const float* values, int img_w, int img_h, float sigmargb, float sigmaxy, float* out){
	Permutohedral lattice;
	initializePermutohedral(image, img_w, img_h, sigmargb, sigmaxy, lattice);
	lattice.compute(out, values, 1);
}

