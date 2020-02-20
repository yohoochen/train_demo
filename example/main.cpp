//
// Created by chen on 19-12-26.
//
#include <string>
#include <math.h>
#include <iostream>
#include <Object_detect.h>
#include <ObjProcess.h>
#include <algorithm>
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

//&neck&rhip&lhip[0,1,2]
bool judge(const float body[][2]) {
    float center[2];
    if(body[0][0] == -1 || body[1][0] == -1 || body[2][0] == -1){
        return false;
    }
    center[0] = (body[1][0] + body[2][0])/2;
    center[1] = (body[1][1] + body[2][1])/2;
    float k = atan2(center[1]-body[0][1],body[0][0]-center[0]);
	cout<<"k  "<<k<<"++"<<M_PI*5/12<<"++"<<M_PI*7/12<<endl;
    if(k >= M_PI*4/12 && k <= M_PI*9/12){
        return false;
    }
    return true;
}

void fall(std::vector<HumanPose> poses, cv::Mat &frame){
	RotatedRect rect;
    for (HumanPose const& pose : poses) {
    	vector<cv::Point2f> keypoint;
		for(cv::Point2f point : pose.keypoints){
			if(int(point.x) == -1){continue;}
			keypoint.push_back(point);
		}
        float body[3][2] = {{pose.keypoints[1].x , pose.keypoints[1].y}, {pose.keypoints[8].x, pose.keypoints[8].y}, {pose.keypoints[11].x, pose.keypoints[11].y}};

        if(judge(body)){
            cout<<"falling"<<endl;

		    rect = cv::minAreaRect(keypoint);
		    cv::rectangle(frame, rect.boundingRect(), cv::Scalar(0, 0, 255), 2);
            //std::cout<<"keypoint"<<keypoint<<endl;
            cv::putText(frame,"falling",cv::Point(rect.center.x,rect.center.y), cv::FONT_HERSHEY_SIMPLEX, 1.5,cv::Scalar(255,255,255),4,8);
        }

    }
}
double getDistance(cv::Point2f pointO,cv::Point2f pointA)
{
    double distance;
    distance = powf((pointO.x - pointA.x),2) + powf((pointO.y - pointA.y),2);
    distance = sqrtf(distance);
	//cout<<"distance  "<<distance<<endl;
    return distance;
}

void waving(vector<vector<HumanPose>> &persons,cv::Mat &frame){
	int size = persons.size();
	RotatedRect rect;
    for (int i = 0; i< size; i++){
		if(persons[i][0].keypoints[4].x == -1 || persons[i][0].keypoints[3].x == -1){ continue; }
		if(persons[i][0].keypoints[4].y >= persons[i][0].keypoints[3].y){ continue; }
		cout<<"11"<<endl;

		// angle
		float max = 0;
		float min = 2*M_PI;
		float angle;
		int n = persons[i].size();
		for(int j = 0; j< n; j++){
			angle = atan2(persons[i][j].keypoints[3].y - persons[i][j].keypoints[4].y, persons[i][j].keypoints[4].x - persons[i][j].keypoints[3].x);
			cout<<persons[i][j].keypoints[3].y - persons[i][j].keypoints[4].y<<"+++"<<persons[i][j].keypoints[4].x - persons[i][j].keypoints[3].x<<"+++"<<angle<<endl;
			if(angle > max){ max = angle; }
			if(angle < min){ min = angle; }
		}
		cout<<"22"<<min<<"++"<<max<<"++"<<M_PI*1/3<<endl;
		

		if((max - min) > M_PI*1/3){
			vector<cv::Point2f> keypoint;
			for(cv::Point2f point : persons[i][n].keypoints){
				cout<<point.x<<"++"<<point.y<<endl;
				if(int(point.x) == -1){continue;}
				keypoint.push_back(point);
			}
            cout<<"waving"<<n<<endl;
		    rect = cv::minAreaRect(keypoint);
			cout<<"22-1"<<endl;
		    cv::rectangle(frame, rect.boundingRect(), cv::Scalar(0, 0, 255), 2);
            //std::cout<<"keypoint"<<keypoint<<endl;
            cv::putText(frame,"waving",cv::Point(rect.center.x,rect.center.y), cv::FONT_HERSHEY_SIMPLEX, 1.5,cv::Scalar(0,0,0),4,8);
			cout<<"22-2"<<endl;
			
		}
		cout<<"33"<<endl;
	}

}

//0nose, 1neck, 2Rsho, 3Relb, 4Rwri, 5Lsho, 6Lelb, 7Lwri, 8Rhip, 9Rkne, 10Rank, 11Lhip, 12Lkne, 13Lank, 14Leye, 15Reye, 16Lear, 17Rear
void wave_hands(std::vector<HumanPose> &poses, cv::Mat &frame, vector<cv::Point2f> &person,   vector<vector<HumanPose>> &all_poses){

    for (HumanPose const& pose : poses) {
		// has neck or not
        if(int(pose.keypoints[1].x) == -1){ continue;}
		// put neck in person vector
		int size = person.size();
		if(size == 0){
			person.push_back(pose.keypoints[1]);
			all_poses.push_back({pose});
			continue;
		}
		cout<<"1"<<endl;

		double min = 100000;
		int index = 0;
        for (int i = 0; i < size; i++){
			double t = getDistance(person[i], pose.keypoints[1]);
			if(t < min){
				index = i;
				min = t;
			}
        }
		cout<<"2"<<endl;

        if( min > 100){
			//cout<<"2.1"<<endl;
            person.push_back(pose.keypoints[1]);
			//cout<<"2.2"<<endl;
            all_poses.push_back({pose});
			//cout<<"2.3"<<endl;
            continue;
        }
		cout<<"3"<<endl;

        //everyone's neck position
        person[index] = pose.keypoints[1];
        //everyone's keypoints
        if(all_poses[index].size()>20){
            all_poses[index].erase(begin(all_poses[index]));
        }
        all_poses[index].push_back(pose);
		cout<<"4"<<endl;

    }
    waving(all_poses, frame);
}

int main(int argc, const char** argv){

    HumanPoseEstimator estimator("model/human_pose_light_model.pt");

    //deque<int> mem(10);
    cv::namedWindow("result",cv::WINDOW_NORMAL);

    std::string model_file = "model/model_head.engine";
    Object_Detection::Object_detect net(model_file);
    //std::unique_ptr<float[]> outputData(new float[net.outputBufferSize]);
    cv::Mat frame;
    cv::VideoCapture cap("help/3.mp4");
//    cv::VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

    std::vector<Object_Detection::Object> Obj_pool;

    cv::VideoWriter outputVideo;
    cv::Size s = cv::Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
                          (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    outputVideo.open("g_video/wave.avi", CV_FOURCC('X','V','I','D'), 25.0,
                     s, true);

	// keep
	vector<vector<HumanPose>> all_poses;
    // eyeryone's neck
	vector<cv::Point2f> person;


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
        fall(poses, frame);
	// draw waving
		wave_hands(poses, frame, person, all_poses);
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
