//
// 
//

#ifndef OBJECT_DETECT_H
#define OBJECT_DETECT_H

#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include <OjectConfig.h>
#include <ObjProcess.h>
#include <opencv2/opencv.hpp>


namespace Object_Detection
{
    enum class RUN_MODE
    {
        FLOAT32 = 0 ,
        FLOAT16 = 1 ,
        INT8    = 2
    };


//    static std::string CLASSES[4] = {"train", "Red_light","Person","Green_light"};
    //static std::string CLASSES[1] = {"head"};
	static std::string CLASSES[3] = {"Mask_head","Unclear_head", "No mask"};
    static std::string STATUS[3] = {"comfortable", "normal", "crowded"};
	static int no_mask_frame;
    static std::vector<cv::Point> contours;
    static std::vector<cv::Point> contours2;
    static cv::Scalar COLOR[4]={cv::Scalar(18,87,220),cv::Scalar(255,0,252),cv::Scalar(0,0,255),cv::Scalar(0,255,0)};
//    void draw_bb_top(cv::Mat &img_, std::string &name, cv::Point &pt_lt, cv::Point &pt_br, cv::Scalar &color);
    static std::deque<int> memory;

    void draw_bb_top(cv::Mat &img_, std::string &name, cv::Point &pt_lt, cv::Point &pt_br, cv::Scalar &color);
    bool area(cv::Mat &img_, std::string &name, cv::Point &pt_lt, cv::Point &pt_br, cv::Scalar &color);
    bool getResult(std::pair<std::string, int> &result, cv::Mat &img, std::vector<Object> &Obj_pool);
    void getMemory(int i, int &max);
    class Object_detect
    {
    public:
        Object_detect(const std::string& onnxFile,
                 const std::string& calibFile,
                 RUN_MODE mode = RUN_MODE::FLOAT32);

        Object_detect(const std::string& engineFile);

        ~Object_detect(){
            cudaStreamSynchronize(mCudaStream);
            cudaStreamDestroy(mCudaStream);
            for(auto& item : mCudaBuffers)
                cudaFree(item);
            cudaFree(cudaOutputBuffer);
            if(!mRunTime)
                mRunTime->destroy();
            if(!mContext)
                mContext->destroy();
            if(!mEngine)
                mEngine->destroy();
        }

        void saveEngine(const std::string& fileName);

        void doInference(const void* inputData, void* outputData);

        void detect(cv::Mat& img, std::vector<Object> &Obj_pool);


        void printTime()
        {
            mProfiler.printTime(runIters) ;
        }

        inline size_t getInputSize() {
            return mBindBufferSizes[0];
        };

        int64_t outputBufferSize;
        bool forwardFace;
    private:

        void InitEngine();


        nvinfer1::IExecutionContext* mContext;
        nvinfer1::ICudaEngine* mEngine;
        nvinfer1::IRuntime* mRunTime;

        RUN_MODE runMode;

        std::vector<void*> mCudaBuffers;
        std::vector<int64_t> mBindBufferSizes;
        void * cudaOutputBuffer;

        cudaStream_t mCudaStream;

        int runIters;
        Profiler mProfiler;

        std::unique_ptr<float[]> inference_outputData;
        
        void enableDLA(nvinfer1::IBuilder* b, int useDLACore, bool allowGPUFallback );
        int gUseDLACore;

    };

}


#endif //CTDET_TRT_CTDETNET_H
