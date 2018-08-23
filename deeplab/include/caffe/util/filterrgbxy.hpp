#ifndef _H_FILTERXY_
#define _H_FILTERXY_
#include <math.h>
#include <vector>
#include <cstddef>
#include <stdlib.h>

//#include "caffe/util/densecrf_pairwise.hpp"
//#include "caffe/util/densecrf_util.hpp"
#include "caffe/util/permutohedral.hpp"

void initializePermutohedral(const float * image, int img_w, int img_h, float sigmargb, float sigmaxy, Permutohedral & lattice_);

void filterrgbxy(const float* image, const float* values, int img_w, int img_h, float sigmargb, float sigmaxy, float* out);

#endif
