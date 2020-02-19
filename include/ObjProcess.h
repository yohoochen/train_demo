//
// 
//

#ifndef OBJPROCESS_H
#define OBJPROCESS_H

#include <numeric>
#include <map>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cublas_v2.h>
#include <cudnn.h>
#include <assert.h>
#include "NvInfer.h"
#include <opencv2/opencv.hpp>


#ifndef BLOCK
#define BLOCK 512
#endif
#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }
#endif


namespace Object_Detection
{

    class Profiler : public nvinfer1::IProfiler
    {
    public:
        struct Record
        {
            float time{0};
            int count{0};
        };
        void printTime(const int& runTimes)
        {
            //std::cout << "========== " << mName << " profile ==========" << std::endl;
            float totalTime = 0;
            std::string layerNameStr = "TensorRT layer name";
            int maxLayerNameLength = std::max(static_cast<int>(layerNameStr.size()), 70);
            for (const auto& elem : mProfile)
            {
                totalTime += elem.second.time;
                maxLayerNameLength = std::max(maxLayerNameLength, static_cast<int>(elem.first.size()));
            }

            std::cout<< " total runtime = " << totalTime/runTimes << " ms " << std::endl;
        }

        virtual void reportLayerTime(const char* layerName, float ms)
        {
            mProfile[layerName].count++;
            mProfile[layerName].time += ms;
        }
    private:
        std::map<std::string, Record> mProfile;
    };

    class Logger : public nvinfer1::ILogger
    {
    public:
        Logger(Severity severity = Severity::kWARNING)
                : reportableSeverity(severity)
        {
        }

        void log(Severity severity, const char* msg) override
        {
            // suppress messages with severity enum value greater than the reportable
            if (severity > reportableSeverity)
                return;

            switch (severity)
            {
                case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
                case Severity::kERROR: std::cerr << "ERROR: "; break;
                case Severity::kWARNING: std::cerr << "WARNING: "; break;
                case Severity::kINFO: std::cerr << "INFO: "; break;
                default: std::cerr << "UNKNOWN: "; break;
            }
            std::cerr << msg << std::endl;
        }
        Severity reportableSeverity;
    };

    inline int64_t volume(const nvinfer1::Dims& d)
    {
        return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
    }

    inline unsigned int getElementSize(nvinfer1::DataType t)
    {
        switch (t)
        {
            case nvinfer1::DataType::kINT32: return 4;
            case nvinfer1::DataType::kFLOAT: return 4;
            case nvinfer1::DataType::kHALF: return 2;
            case nvinfer1::DataType::kINT8: return 1;
        }
        throw std::runtime_error("Invalid DataType.");
        return 0;
    }

    inline void* safeCudaMalloc(size_t memSize)
    {
        void* deviceMem;
        CUDA_CHECK(cudaMalloc(&deviceMem, memSize));
        if (deviceMem == nullptr)
        {
            std::cerr << "Out of memory" << std::endl;
            exit(1);
        }
        return deviceMem;
    }

    struct Box{
        float x1;
        float y1;
        float x2;
        float y2;
    };
    struct landmarks{
        float x;
        float y;
    };

    struct Detection{
        //x1 y1 x2 y2
        Box bbox;
        //float objectness;
        landmarks marks[5];
        int classId;
        float prob;
    };


    struct Object
    {
        cv::Rect boundingbox;
        int label;
    };


    extern dim3 cudaGridSize(uint n);
    extern std::vector<float> prepareImage(cv::Mat& img);
    extern void postProcess(std::vector<Detection> & result,const cv::Mat& img);
    extern void Result_convert(std::vector<Detection> & result, std::vector<Object> &Obj_pool);
}

#endif //CTDET_TRT_UTILS_H
