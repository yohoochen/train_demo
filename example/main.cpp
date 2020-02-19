//
// Created by chen on 19-12-26.
//
#include <string>
#include <iostream>
#include <Object_detect.h>
#include <ObjProcess.h>
#include <memory>
#include <queue>
#include <torch/script.h> // One-stop header.
#include "glog/logging.h"
#include "opencv2/videoio.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "human_pose_estimator.h"
#include "thor/vis.h"

using namespace std;
using namespace thor::vis;
using namespace google;
using namespace human_pose_estimation;
using namespace cv;
typedef std::pair<string, int> Result;

bool judge(const vector<float> &neck, const vector<float> &rhip, const vector<float> &lhip){
    float center[2];
    if(neck[0] == -1 || rhip[0] == -1 || lhip[0] == -1){
        return false;
    }
    center[0] = (rhip[0] + lhip[0])/2;
    center[1] = (rhip[1] + lhip[1])/2;
    float k = (center[1]-neck[1])/(neck[0]-center[0]);
    if(k >= 0.75 || k <= -1){
        return false;
    }
    return true;
}

cv::Mat pose(std::vector<HumanPose> poses, cv::Mat frame){
    vector<vector<float>> keypoint;
    for (HumanPose const& pose : poses) {
        std::stringstream rawPose;
        rawPose << std::fixed << std::setprecision(0);
        rawPose << pose.score;
        std::cout<<"pose.keypoints"<<pose.keypoints<<endl;
        keypoint = {{pose.keypoints[1].x , pose.keypoints[1].y}, {pose.keypoints[8].x, pose.keypoints[8].y}, {pose.keypoints[11].x, pose.keypoints[11].y}};
        float x_max = 0.0 , y_max = 0.0 , x_min = 1280.0, y_min = 720.0;

        for(cv::Point2f point : pose.keypoints){
            if(point.x > x_max){x_max = point.x;}
            if(point.x < x_min){x_min = point.x;}
            if(point.y > y_max){y_max = point.y;}
            if(point.y < y_min){y_min = point.y;}
        }
        if(judge(keypoint[0],keypoint[1],keypoint[2])){
            cout<<"falling"<<endl;
            cv::rectangle(frame, {int(x_min), int(y_min)}, {int(x_max), int(y_max)}, cv::Scalar(0, 0, 255), 2);
            cv::putText(frame,"falling",cv::Point(x_min+7,y_min-4), cv::FONT_HERSHEY_SIMPLEX, 0.6,cv::Scalar(255,255,255),1,8);
        }
    }
    return frame;
}

int main(int argc, const char** argv){

    HumanPoseEstimator estimator("model/human_pose_light_model.pt");

    deque<int> mem(10);
    cv::namedWindow("result",cv::WINDOW_NORMAL);

    std::string model_file = "model/model_head.engine";
    Object_Detection::Object_detect net(model_file);
    //std::unique_ptr<float[]> outputData(new float[net.outputBufferSize]);
    cv::Mat frame;
    cv::VideoCapture cap("vlc.mp4");
//    cv::VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

    std::vector<Object_Detection::Object> Obj_pool;

    cv::VideoWriter outputVideo;
    cv::Size s = cv::Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
                          (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    outputVideo.open("head.avi", CV_FOURCC('X','V','I','D'), 25.0,
                     s, true);
    if (!outputVideo.isOpened())
    {
        cout << "Open video error !"<<endl;
        return -1;
    }
    while (cap.read(frame))
    {
        int start = cv::getTickCount();
//        pose estimation code
        std::vector<HumanPose> poses = estimator.estimate(frame);
        frame2 = pose(poses, frame);

//        human detection
        net.detect(frame, Obj_pool);
        Result result;
//        draw img
        renderHumanPose(poses, frame2);
        Object_Detection::getResult(result, frame2, Obj_pool);
        std::cout<<"status:  "<<result.first<<"   numbers:  "<<result.second<<std::endl;
        cv::putText(frame2, result.first, cv::Point(900, 50),cv::FONT_HERSHEY_SIMPLEX,2,cv::Scalar(0,0,255),4,8);
        cv::putText(frame2, "number: "+to_string(result.second), cv::Point(0, 200),cv::FONT_HERSHEY_SIMPLEX,2,cv::Scalar(0,0,255),4,8);
//        output video
        int end_tick = cv::getTickCount();
        std::cout << " detect time:" << 1000.0 * (end_tick - start) / cv::getTickFrequency() << "ms" << std::endl << std::endl;
        cv::imshow("result",frame2);
        outputVideo<< frame2;
//        waitkey
        if((cv::waitKey(1)& 0xff) == 27){
            cv::destroyAllWindows();
            return 0;
        }
    }
    return 0;
}
