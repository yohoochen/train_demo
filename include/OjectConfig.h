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

    constexpr static int classNum = 1 ;
    constexpr static float mean[]= {0.408, 0.447, 0.470};
    constexpr static float std[] = {0.289, 0.274, 0.278};
//    constexpr static char *className[]= {(char*)"train",(char*)"Red_light",(char*)"Person",(char*)"Green_light"};
    constexpr static char *className[]= {(char*)"head"};
}
#endif //CTDET_TRT_CTDETCONFIG_H
