//
//
//

#ifndef OBJECTLAYER_H
#define OBJECTLAYER_H



#include <ObjProcess.h>


extern "C" void CTdetforward_gpu(const float *hm, const float *reg,const float *wh ,float *output,
                      const int w,const int h,const int classes,const int kernerl_size,const float visthresh  );
#endif //CTDET_TRT_CTDETLAYER_H
