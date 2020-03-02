//
// 
//
#ifndef OBJECTCONFIG_H
#define OBJECTCONFIG_H

namespace Object_Detection{

    constexpr static float visThresh = 0.4;
    constexpr static int inputSize = 512 ;
    constexpr static int channel = 3 ;
    constexpr static int ouputSize = inputSize/4 ;
    constexpr static int kernelSize = 4 ;

    constexpr static int classNum = 3 ;
    constexpr static float mean[]= {0.373711, 0.377281,0.378505};
    constexpr static float std[] = {0.209044, 0.208966, 0.211592};
    constexpr static char *className[]= {(char*)"Mask_head",(char*)"Clear_head",(char*)"Unclear_head"};
}
#endif //CTDET_TRT_CTDETCONFIG_H