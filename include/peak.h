// ************************************
// Copyrights by Jin Fagang
// 6/11/19-11-14
// peak
// jinfagang19@gmail.com
// ************************************

//
// Created by jintain on 6/11/19.
//

#ifndef CUSTOM_OPS_PEAK_H
#define CUSTOM_OPS_PEAK_H

#pragma once

#include <vector>
#include <opencv2/core/core.hpp>
#include "thor/structures.h"


using namespace thor;


/**
 *
 *
 * A tiny library solving Heatmaps with peaks
 * finding the keypoints from heatmaps
 */


namespace human_pose_estimation {
struct Peak {
  Peak(const int id = -1,
	   const cv::Point2f& pos = cv::Point2f(),
	   const float score = 0.0f);

  int id;
  cv::Point2f pos;
  float score;
};

struct HumanPoseByPeaksIndices {
  explicit HumanPoseByPeaksIndices(const int keypointsNumber);

  std::vector<int> peaksIndices;
  int nJoints;
  float score;
};

struct TwoJointsConnection {
  TwoJointsConnection(const int firstJointIdx,
					  const int secondJointIdx,
					  const float score);

  int firstJointIdx;
  int secondJointIdx;
  float score;
};

void findPeaks(const std::vector<cv::Mat>& heatMaps,
			   const float minPeaksDistance,
			   std::vector<std::vector<Peak> >& allPeaks,
			   int heatMapId);

std::vector<HumanPose> groupPeaksToPoses(
	const std::vector<std::vector<Peak> >& allPeaks,
	const std::vector<cv::Mat>& pafs,
	const size_t keypointsNumber,
	const float midPointsScoreThreshold,
	const float foundMidPointsRatioThreshold,
	const int minJointsNumber,
	const float minSubsetScore);
}  // namespace human_pose_estimation




#endif //CUSTOM_OPS_PEAK_H
