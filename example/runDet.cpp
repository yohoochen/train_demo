//
//
//

#include <string>
#include <iostream>
#include <Object_detect.h>
#include <ObjProcess.h>
#include <memory>
#include <queue>

using namespace std;
typedef std::pair<string, int> Result;
int main(int argc, const char** argv){

    deque<int> mem(10);
    cv::namedWindow("result",cv::WINDOW_NORMAL);
    std::string model_file = "model/model_head.engine";

    Object_Detection::Object_detect net(model_file);
    //std::unique_ptr<float[]> outputData(new float[net.outputBufferSize]);
    cv::Mat img;
    cv::VideoCapture cap("/home/chen/Videos/vlc.mp4");
    std::vector<Object_Detection::Object> Obj_pool;

    cv::VideoWriter outputVideo;
    cv::Size s = cv::Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
                          (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    outputVideo.open("/home/chen/centernet/TensorRT-CenterNet/head.avi", CV_FOURCC('X','V','I','D'), 25.0,
                     s, true);
    if (!outputVideo.isOpened())
    {
        cout << "Open video error !"<<endl;
        return -1;
    }
    while (cap.read(img))
    {
        int start = cv::getTickCount();

        net.detect(img, Obj_pool);

        Result result;
        Object_Detection::getResult(result, img, Obj_pool);

        std::cout<<"status:  "<<result.first<<"   numbers:  "<<result.second<<std::endl;
        cv::putText(img, result.first, cv::Point(900, 50),cv::FONT_HERSHEY_SIMPLEX,2,cv::Scalar(0,0,255),4,8);
        cv::putText(img, "number: "+to_string(result.second), cv::Point(0, 200),cv::FONT_HERSHEY_SIMPLEX,2,cv::Scalar(0,0,255),4,8);

        int end_tick = cv::getTickCount();
        std::cout << " detect time:" << 1000.0 * (end_tick - start) / cv::getTickFrequency() << "ms" << std::endl << std::endl;

        cv::imshow("result",img);
        outputVideo<< img;

        if((cv::waitKey(1)& 0xff) == 27){
            cv::destroyAllWindows();
            return 0;
        };

    }

    

    return 0;
}