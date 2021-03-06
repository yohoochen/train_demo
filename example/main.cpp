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

bool area(int type, cv::Point center, RotatedRect rect){
	vector<cv::Point> contours;
	double result;
	if(type == 1){
        Point2f vertices[4];      //定义矩形的4个顶点
        rect.points(vertices);   //计算矩形的4个顶点
        contours.emplace_back(cv::Point(400, 360));
        contours.emplace_back(cv::Point(880, 360));
        contours.emplace_back(cv::Point(1130, 720));
        contours.emplace_back(cv::Point(150, 720));
        for(int i = 0 ; i < 4 ; i++ ){
			cout<<" vertices[i]"<< vertices[i]<<endl;
            result = pointPolygonTest(contours, vertices[i], false);
            if(result != 1){
                cout<<"zzzz"<<endl;
                return false;
            }
        }
		cout<<"qqqq"<<endl;
		return true;
	}
	if(type == 2){
        contours.emplace_back(cv::Point(400, 230));
        contours.emplace_back(cv::Point(880, 230));
        contours.emplace_back(cv::Point(980, 600));
        contours.emplace_back(cv::Point(300, 600));
        result = pointPolygonTest(contours, center, false);
		if(result == 1){
			return true;
		}
		return false;
	}
}
//&neck&rhip&lhip[0,1,2]
bool judge(const float body[][2], vector<cv::Point2f> keypoint) {
    float center[2];
    if(body[0][0] == -1 || body[1][0] == -1 || body[2][0] == -1){
        return false;
    }
    center[0] = (body[1][0] + body[2][0])/2;
    center[1] = (body[1][1] + body[2][1])/2;
    float k = atan2(center[1]-body[0][1],body[0][0]-center[0]);
	//cout<<"k  "<<k<<"++"<<M_PI*5/12<<"++"<<M_PI*7/12<<endl;
	cv::Point point = cv::Point(center[0],center[1]);
    if(k >= M_PI*1/6 && k <= M_PI*5/6){
        return false;
    }
	RotatedRect rect = cv::minAreaRect(keypoint);
	if(area(1, point, rect)){
    	return true;
	}
	return false;
}

void fall(std::vector<HumanPose> poses, cv::Mat &frame){
	RotatedRect rect;
	cout<<"00"<<endl;
    for (HumanPose const& pose : poses) {
    	vector<cv::Point2f> keypoint;
		for(cv::Point2f point : pose.keypoints){
			if(int(point.x) == -1){continue;}
			keypoint.push_back(point);
		}
        float body[3][2] = {{pose.keypoints[1].x , pose.keypoints[1].y}, {pose.keypoints[8].x, pose.keypoints[8].y}, {pose.keypoints[11].x, pose.keypoints[11].y}};
		cout<<"11"<<endl;
		cout<<"keypoint.size()"<<keypoint.size()<<endl;
		cout<<"22"<<endl;
        if(judge(body, keypoint)){
			cout<<"11"<<endl;
            //cout<<"falling"<<endl;
        	rect = cv::minAreaRect(keypoint);
			cout<<"rect.size.width"<<rect.size.width<<endl;
			if(rect.size.width > 250){
		    	cv::rectangle(frame, rect.boundingRect(), cv::Scalar(0, 0, 255), 2);
            //std::cout<<"keypoint"<<keypoint<<endl;
            	cv::putText(frame,"falling",cv::Point(rect.center.x,rect.center.y), cv::FONT_HERSHEY_SIMPLEX, 1.5,cv::Scalar(255,255,255),4,8);
			}
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
	//cout<<"00  "<<size<<endl;
	RotatedRect rect;
    for (int i = 0; i< size; i++){

		//cout<<"11"<<endl;

		// angle
		float max = 0;
		float min = 2*M_PI;
		float angle;
		int n = persons[i].size();
		//cout<<"n  "<<n<<endl;
		if(n == 0){
			//persons.erase(persons.begin());
			continue;	
		}
		for(int j = 0; j< n; j++){
			angle = atan2(persons[i][j].keypoints[3].y - persons[i][j].keypoints[4].y, persons[i][j].keypoints[4].x - persons[i][j].keypoints[3].x);
			//cout<<persons[i][j].keypoints[3].y<<"&&&"<<persons[i][j].keypoints[4].y<<"&&&"<<persons[i][j].keypoints[4].x<<"&&&"<<persons[i][j].keypoints[3].x<<"&&&"<<angle<<endl;
			if(angle > max){ max = angle; }
			if(angle < min){ min = angle; }
		}
		cout<<"22   "<<min<<"((("<<max<<"((("<<M_PI/3<<"((("<<M_PI/2<<endl;
		if(abs(persons[i][n-1].keypoints[8].x-persons[i][n-1].keypoints[11].x)<50){
			cout<<"nonononononono   "<<abs(persons[i][n-1].keypoints[8].x-persons[i][n-1].keypoints[11].x)<<endl;
			continue;
		}

		if(((max - min) > M_PI/2) && (max > M_PI/2)){
			vector<cv::Point2f> keypoint;
			int face_points = 0;
			int face = 0;
			for(cv::Point2f point : persons[i][n-1].keypoints){
				//cout<<point.x<<"++"<<point.y<<endl;
				if(face_points == 0||face_points == 14||face_points == 15||face_points == 16||face_points == 17){
					if(int(point.x) != -1){face++;}
				}
				face_points++;
				if(int(point.x) == -1){continue;}
				keypoint.push_back(point);
			}
			if(face <=4){
				cout<<"++++++face+++++++"<<face<<endl;
				continue;
			}
			cout<<"22-1"<<endl;
			int limit = keypoint.size();
			if(limit <=7){continue;}
            //cout<<"waving"<<n<<endl;
		    rect = cv::minAreaRect(keypoint);
			cout<<"22-2"<<endl;
			if(area(2, rect.center, rect)){
		    	cv::rectangle(frame, rect.boundingRect(), cv::Scalar(0, 0, 255), 2);
            	std::cout<<"keypoint"<<keypoint<<endl;
				cv::putText(frame,"waving",cv::Point(rect.center.x,rect.center.y), cv::FONT_HERSHEY_SIMPLEX, 1.5,cv::Scalar(255,255,255),4,8);
			//cout<<"22-2"<<endl;
			}

		}
		//cout<<"33"<<endl;
	}

}

//0nose, 1neck, 2Rsho, 3Relb, 4Rwri, 5Lsho, 6Lelb, 7Lwri, 8Rhip, 9Rkne, 10Rank, 11Lhip, 12Lkne, 13Lank, 14Leye, 15Reye, 16Lear, 17Rear
void wave_hands(std::vector<HumanPose> &poses, cv::Mat &frame, vector<cv::Point2f> &person,   vector<vector<HumanPose>> &all_poses, int maximum){
	
	//cout<<"here"<<endl;
    for (HumanPose const& pose : poses) {
		//cout<<"0.1"<<endl;
		// has neck or not
        if(int(pose.keypoints[1].x) == -1){ continue;}
		//cout<<"0.2"<<endl;
		// put neck in person vector
		int size = person.size();
		//cout<<"0.3"<<"size  "<<size<<endl;
		if(size == 0){
			person.push_back(pose.keypoints[1]);
			all_poses.push_back({pose});
			continue;
		}
		//cout<<"1"<<endl;
		double min = 100000;
		int index = 0;
        for (int i = 0; i < size; i++){
			double t = getDistance(person[i], pose.keypoints[1]);
			if(t < min){
				index = i;
				min = t;
			}
        }
		//cout<<"2   "<<min<<endl;

        if( min > 100){
			//cout<<"2.1"<<endl;
            person.push_back(pose.keypoints[1]);
			//cout<<"2.2"<<endl;
            all_poses.push_back({pose});
			//cout<<"2.3"<<endl;
            continue;
        }
		//cout<<"3"<<endl;

        //everyone's neck position
        person[index] = pose.keypoints[1];
        //everyone's keypoints
        if(all_poses[index].size()>20){
            all_poses[index].erase(begin(all_poses[index]));
        }
        all_poses[index].push_back(pose);
		//cout<<"4"<<endl;

    }
    int o = all_poses.size();
	int k = poses.size();

	if(maximum > k){
		maximum = k;
	}

    vector<cv::Point2f> tmp_person = {};
    vector<vector<HumanPose>> tmp_all_poses = {};
	//cout<<"poses  "<<k<<endl;
	//cout<<"all_poses  "<<all_poses.size()<<"  person  "<<person.size()<<endl;
    for (int m = 0; m < o; m++){
        if(m >= (o-maximum)){
            tmp_all_poses.push_back(all_poses[m]);
            tmp_person.push_back(person[m]);
        }
    }
    all_poses.clear();
    person.clear();
    all_poses = tmp_all_poses;
    person = tmp_person;
//	if((o-k)==1){
//		all_poses.erase(all_poses.begin());
//		person.erase(person.begin());
//	}
//	if((o-k)>1){
//		all_poses.erase(all_poses.begin(), all_poses.end()-k+1);
//		//person.erase(person.begin(), person.begin()+(o-k)+1);
//		person.erase(person.begin(), person.end()-k+1);
//	}
    for (int i = 0; i< (int)all_poses.size(); i++) {
		//cout<<"here2  "<<i<<endl;
        for (vector<HumanPose>::iterator  it = all_poses[i].begin(); it != all_poses[i].end();){
			//cout<<"here3"<<endl;
			//cout<<(*it).keypoints<<endl;
            if((*it).keypoints[4].x == -1 || (*it).keypoints[3].x == -1 || (*it).keypoints[8].x == -1 || (*it).keypoints[11].x == -1){
				//cout<<"here4"<<endl;
                it = all_poses[i].erase(it);
				//cout<<"here4.1"<<endl;
				continue;
            }
            if((*it).keypoints[4].y >= (*it).keypoints[3].y){
				//cout<<"here5"<<endl;
                it = all_poses[i].erase(all_poses[i].begin(), it+1);
				continue;
            }
			it++;
        }
    }

    waving(all_poses, frame);
}

int main(int argc, const char** argv){

    HumanPoseEstimator estimator("model/human_pose_light_model.pt");

    //deque<int> mem(10);
    cv::namedWindow("result",cv::WINDOW_NORMAL);

    std::string model_file = "model/head_face_model.trt";
    Object_Detection::Object_detect net(model_file);
    //std::unique_ptr<float[]> outputData(new float[net.outputBufferSize]);
    cv::Mat frame;
    //cv::VideoCapture cap("/home/nvidia/videos/video1/Camera_16/Data_20200107_005634_L.avi");
    cv::VideoCapture cap("/home/nvidia/Pictures/六号线视频/Data_20200309_140057_L.avi");
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

    std::vector<Object_Detection::Object> Obj_pool;

    cv::VideoWriter outputVideo;
    cv::Size s = cv::Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
                          (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    outputVideo.open("g_video/bad.avi", CV_FOURCC('X','V','I','D'), 25.0,
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
		cout<<"0"<<endl;
        fall(poses, frame);
		cout<<"1"<<endl;
	// draw waving
		wave_hands(poses, frame, person, all_poses, Obj_pool.size());
	// draw pose
        renderHumanPose(poses, frame);
	//draw head
        if(Object_Detection::getResult(result, frame, Obj_pool)){
			cv::putText(frame, "Find_NoMask", cv::Point(900, 250),cv::FONT_HERSHEY_SIMPLEX,2,cv::Scalar(0,0,255),4,8);
		}

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
