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

cv::Mat fall(std::vector<HumanPose> poses, cv::Mat frame){
    vector<cv::Point2f> keypoint;
	int body[3][2];
	RotatedRect rect;
    for (HumanPose const& pose : poses) {
		for(cv::Point2f point : pose.keypoints){
			if(int(point.x) == -1){continue;}
			keypoint.push_back(point);
		}
        body = {{pose.keypoints[1].x , pose.keypoints[1].y}, {pose.keypoints[8].x, pose.keypoints[8].y}, {pose.keypoints[11].x, pose.keypoints[11].y}};

        if(judge(keypoint[0],keypoint[1],keypoint[2])){
            //cout<<"falling"<<endl;

		rect = cv::minAreaRect(keypoint);
		cv::rectangle(frame, rect.boundingRect(), cv::Scalar(0, 0, 255), 2);
        std::cout<<"keypoint"<<keypoint<<endl;
            cv::putText(frame,"falling",cv::Point(x_min+7,y_min-4), cv::FONT_HERSHEY_SIMPLEX, 0.6,cv::Scalar(255,255,255),1,8);
        }

    }
    return frame;
}

int main(int argc, const char** argv){

    HumanPoseEstimator estimator("model/human_pose_light_model.pt");

    //deque<int> mem(10);
    cv::namedWindow("result",cv::WINDOW_NORMAL);

    std::string model_file = "model/model_head.engine";
    Object_Detection::Object_detect net(model_file);
    //std::unique_ptr<float[]> outputData(new float[net.outputBufferSize]);
    cv::Mat frame;
    cv::VideoCapture cap("fall/2.avi");
//    cv::VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

    std::vector<Object_Detection::Object> Obj_pool;

    cv::VideoWriter outputVideo;
    cv::Size s = cv::Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
                          (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    outputVideo.open("g_video/head.avi", CV_FOURCC('X','V','I','D'), 25.0,
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
//        human detection
        net.detect(frame, Obj_pool);
        Result result;
//        draw img
	// draw falling
        frame = fall(poses, frame);
	// draw pose
        renderHumanPose(poses, frame);
	//draw head
        Object_Detection::getResult(result, frame, Obj_pool);

        std::cout<<"status:  "<<result.first<<"   numbers:  "<<result.second<<std::endl;
        cv::putText(frame, result.first, cv::Point(900, 50),cv::FONT_HERSHEY_SIMPLEX,2,cv::Scalar(0,0,255),4,8);
        cv::putText(frame, "number: "+to_string(result.second), cv::Point(0, 200),cv::FONT_HERSHEY_SIMPLEX,2,cv::Scalar(0,0,255),4,8);

//        output video
        int end_tick = cv::getTickCount();
        std::cout << " detect time:" << 1000.0 * (end_tick - start) / cv::getTickFrequency() << "ms" << std::endl << std::endl;
        cv::imshow("result",frame);
        outputVideo<< frame;
//        waitkey
        if(cv::waitKey(1) == 'q'){
            cv::destroyAllWindows();
            return 0;
        }
    }
    return 0;
}
