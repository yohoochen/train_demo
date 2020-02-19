//
// 
//
#include <ObjProcess.h>
#include <OjectConfig.h>
#include <sstream>
#include <vector>

using namespace std;

namespace Object_Detection
{

    dim3 cudaGridSize(uint n)
    {
        uint k = (n - 1) /BLOCK + 1;
        uint x = k ;
        uint y = 1 ;
        if (x > 65535 )
        {
            x = ceil(sqrt(x));
            y = (n - 1 )/(x*BLOCK) + 1;
        }
        dim3 d = {x,y,1} ;
        return d;
    }

    std::vector<float> prepareImage(cv::Mat& img)
    {

        int channel = Object_Detection::channel ;
        int inputSize = Object_Detection::inputSize;
        float scale = min(float(inputSize)/img.cols,float(inputSize)/img.rows);
        auto scaleSize = cv::Size(img.cols * scale,img.rows * scale);

        //int start = cv::getTickCount();
        //std::cout <<"scaleSize:"<< scaleSize << std::endl;
        cv::Mat resized;
        //cv::resize(img, resized,scaleSize,0,0,cv::INTER_LINEAR);
        cv::resize(img, resized,scaleSize,0,0,cv::INTER_NEAREST);

        //int end_tick = cv::getTickCount();
        //std::cout << " resize time:" << 1000.0 * (end_tick - start) / cv::getTickFrequency() << "ms" << std::endl << std::endl;

        cv::Mat cropped = cv::Mat::zeros(inputSize,inputSize,CV_8UC3);
        cv::Rect rect((inputSize- scaleSize.width)/2, (inputSize-scaleSize.height)/2, scaleSize.width,scaleSize.height);
        //std::cout <<"rect:"<< rect.x<<" "<<rect.y<<" "<<rect.width<<" "<<rect.height << std::endl;
        resized.copyTo(cropped(rect));
        cv::Mat img_float;
        cropped.convertTo(img_float, CV_32FC3,1./255.);


        //HWC TO CHW
        vector<cv::Mat> input_channels(channel);
        cv::split(img_float, input_channels);

        // normalize
        vector<float> result(inputSize*inputSize*channel);
        auto data = result.data();
        int channelLength = inputSize * inputSize;
        for (int i = 0; i < channel; ++i) {
            cv::Mat normed_channel = (input_channels[i]-Object_Detection::mean[i])/Object_Detection::std[i];
            memcpy(data,normed_channel.data,channelLength*sizeof(float));
            data += channelLength;
        }

        std::cout <<"result:"<< result.size() << std::endl;
        return result;
    }

    void postProcess(std::vector<Detection> & result,const cv::Mat& img)
    {
        using namespace cv;
        int mark;
        int inputSize = Object_Detection::inputSize;
        float scale = min(float(inputSize)/img.cols,float(inputSize)/img.rows);
        float dx = (inputSize - scale * img.cols) / 2;
        float dy = (inputSize - scale * img.rows) / 2;
        for(auto&item:result)
        {
            float x1 = (item.bbox.x1 - dx) / scale ;
            float y1 = (item.bbox.y1 - dy) / scale ;
            float x2 = (item.bbox.x2 - dx) / scale ;
            float y2 = (item.bbox.y2 - dy) / scale ;
            x1 = (x1 > 0 ) ? x1 : 0 ;
            y1 = (y1 > 0 ) ? y1 : 0 ;
            x2 = (x2 < img.cols  ) ? x2 : img.cols - 1 ;
            y2 = (y2 < img.rows ) ? y2  : img.rows - 1 ;
            item.bbox.x1  = x1 ;
            item.bbox.y1  = y1 ;
            item.bbox.x2  = x2 ;
            item.bbox.y2  = y2 ;


        }
    }


    void Result_convert(std::vector<Detection> & result, std::vector<Object> &Obj_pool)
    {

        Object obj;
        Obj_pool.clear();
        
        for (size_t i = 0; i < result.size(); i++){
            obj.label = result[i].classId;
            obj.boundingbox = cv::Rect(result[i].bbox.x1, result[i].bbox.y1, result[i].bbox.x2-result[i].bbox.x1, result[i].bbox.y2-result[i].bbox.y1);
            Obj_pool.push_back(obj);
        }

    }

}