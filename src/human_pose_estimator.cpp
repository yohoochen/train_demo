// ************************************
// Copyrights by Jin Fagang
// 6/12/19-12-18
// human_pose_estimator
// jinfagang19@gmail.com
// ************************************

#include <torch/script.h>
#include "human_pose_estimator.h"
#include "glog/logging.h"

using namespace google;

namespace human_pose_estimation {

    HumanPoseEstimator::HumanPoseEstimator(const std::string &modelPath)
            : minJointsNumber(3),
              stride(8),
              pad(cv::Vec4i::all(0)),
              meanPixel(cv::Vec3f::all(128)),
              minPeaksDistance(3.0f),
              midPointsScoreThreshold(0.05f),
              foundMidPointsRatioThreshold(0.8f),
              minSubsetScore(0.2f),
              inputLayerSize(456, 256),
              upsampleRatio(4),
              modelPath(modelPath) {

        if (!torch::cuda::is_available()) {
            LOG(INFO) << "CUDA not detected, using CPU instead.";
            device_type = torch::kCPU;
        } else {
            LOG(INFO) << "GPU detected, CUDA enable.";
        }
//        _module_not_ptr = torch::jit::load(modelPath);
//        module = std::make_shared<torch::jit::script::Module>(_module_not_ptr);
        module = torch::jit::load(modelPath);

        module->to(device_type);
        assert(module != nullptr);
        LOG(INFO) << "model loaded.";
    }

    HumanPoseEstimator::~HumanPoseEstimator() {
    }

//estimate
    std::vector<HumanPose> HumanPoseEstimator::estimate(const cv::Mat &image) {
        CV_Assert(image.type() == CV_8UC3);
        cv::Size imageSize = image.size();
        if (this->inputWidthIsChanged(imageSize)) {
            // in torch we do not need do this, since module fit it's input
            // shape according to input image size
            // but we need calculate pad in that function, so we call it once
            LOG(WARNING) << "input width changed.. we currently dont know what to do.";
        }

        cv::Mat in_img;
        preprocess(image, in_img);

        at::Tensor tensor_image = torch::from_blob(in_img.data, {1, in_img.rows, in_img.cols, 3}, at::kByte);
        tensor_image = tensor_image.permute({0, 3, 1, 2}).to(at::kFloat).to(device_type);

        std::vector<torch::jit::IValue> input;
        input.emplace_back(tensor_image);

        double tic = cv::getTickCount();
        auto outputs = module->forward(input).toTensor();
        LOG(INFO) << "forward time: " << ((double) cv::getTickCount() - tic) / cv::getTickFrequency() << "s";

        torch::Tensor stage2_heatmaps = outputs.slice(/*dim=*/1, /*start=*/0, /*end=*/19).detach().squeeze().to(
                torch::kCPU);
        torch::Tensor stage2_pafs = outputs.slice(/*dim=*/1, /*start=*/19, /*end=*/57).detach().squeeze().to(
                torch::kCPU);

        std::vector<HumanPose> poses = postprocess(
                stage2_heatmaps,
                stage2_pafs,
                imageSize);

        return poses;
    }

    void HumanPoseEstimator::preprocess(const cv::Mat &image, cv::Mat &input_image) {
        cv::Mat resizedImage;
        double scale = inputLayerSize.height / static_cast<double>(image.rows);
        cv::resize(image, resizedImage, cv::Size(), scale, scale, cv::INTER_CUBIC);
        cv::Mat paddedImage;
        cv::copyMakeBorder(resizedImage, paddedImage, pad(0), pad(2), pad(1), pad(3),
                           cv::BORDER_CONSTANT, meanPixel);
        paddedImage.copyTo(input_image);
    }

    std::vector<HumanPose> HumanPoseEstimator::postprocess(const torch::Tensor &heatMapsTensor,
                                                           const torch::Tensor &pafsTensor,
                                                           const cv::Size &imageSize) {
        // heatmaps: [19, 32, 43]
        // pafs: 	 [38, 32, 43]
        std::vector<cv::Mat> heatMaps(heatMapsTensor.size(0));

        for (size_t i = 0; i < heatMaps.size(); i++) {
            torch::Tensor one_heat_map = heatMapsTensor[i];
            cv::Mat one_mat(heatMapsTensor.size(1), heatMapsTensor.size(2), CV_32FC1);
            std::memcpy(one_mat.data, one_heat_map.data<float>(), sizeof(float) * one_heat_map.numel());
            heatMaps[i] = one_mat;
        }
        resizeFeatureMaps(heatMaps);

        std::vector<cv::Mat> pafs(pafsTensor.size(0));
        for (size_t i = 0; i < pafs.size(); i++) {
            torch::Tensor one_paf = pafsTensor[i];
            cv::Mat one_mat(pafsTensor.size(1), pafsTensor.size(2), CV_32FC1);
            std::memcpy(one_mat.data, one_paf.data<float>(), sizeof(float) * one_paf.numel());
            pafs[i] = one_mat;
        }

        resizeFeatureMaps(pafs);

        std::vector<HumanPose> poses = extractPoses(heatMaps, pafs);
        correctCoordinates(poses, heatMaps[0].size(), imageSize);
        return poses;
    }


    std::vector<HumanPose> HumanPoseEstimator::extractPoses(const std::vector<cv::Mat> &heatMaps,
                                                            const std::vector<cv::Mat> &pafs) {
        std::vector<std::vector<Peak> > peaksFromHeatMap(heatMaps.size());
        FindPeaksBody findPeaksBody(heatMaps, minPeaksDistance, peaksFromHeatMap);
        cv::parallel_for_(cv::Range(0, static_cast<int>(heatMaps.size())),
                          findPeaksBody);
        int peaksBefore = 0;
        for (size_t heatmapId = 1; heatmapId < heatMaps.size(); heatmapId++) {
            peaksBefore += static_cast<int>(peaksFromHeatMap[heatmapId - 1].size());
            for (auto &peak : peaksFromHeatMap[heatmapId]) {
                peak.id += peaksBefore;
            }
        }
        std::vector<HumanPose> poses = groupPeaksToPoses(
                peaksFromHeatMap, pafs, keypointsNumber, midPointsScoreThreshold,
                foundMidPointsRatioThreshold, minJointsNumber, minSubsetScore);
        return poses;
    }

    void HumanPoseEstimator::resizeFeatureMaps(std::vector<cv::Mat> &featureMaps) {
        for (auto &featureMap : featureMaps) {
            cv::resize(featureMap, featureMap, cv::Size(),
                       upsampleRatio, upsampleRatio, cv::INTER_CUBIC);
        }
    }

    void HumanPoseEstimator::correctCoordinates(std::vector<HumanPose> &poses,
                                                const cv::Size &featureMapsSize,
                                                const cv::Size &imageSize) {
        CV_Assert(stride % upsampleRatio == 0);

        cv::Size fullFeatureMapSize = featureMapsSize * stride / upsampleRatio;

        float scaleX = imageSize.width /
                       static_cast<float>(fullFeatureMapSize.width - pad(1) - pad(3));
        float scaleY = imageSize.height /
                       static_cast<float>(fullFeatureMapSize.height - pad(0) - pad(2));
        for (auto &pose : poses) {
            for (auto &keypoint : pose.keypoints) {
                if (keypoint != cv::Point2f(-1, -1)) {
                    keypoint.x *= stride / upsampleRatio;
                    keypoint.x -= pad(1);
                    keypoint.x *= scaleX;

                    keypoint.y *= stride / upsampleRatio;
                    keypoint.y -= pad(0);
                    keypoint.y *= scaleY;
                }
            }
        }
    }

    bool HumanPoseEstimator::inputWidthIsChanged(const cv::Size &imageSize) {
        double scale = static_cast<double>(inputLayerSize.height) / static_cast<double>(imageSize.height);
        cv::Size scaledSize(static_cast<int>(cvRound(imageSize.width * scale)),
                            static_cast<int>(cvRound(imageSize.height * scale)));
        cv::Size scaledImageSize(std::max(scaledSize.width, inputLayerSize.height),
                                 inputLayerSize.height);
        int minHeight = std::min(scaledImageSize.height, scaledSize.height);
        scaledImageSize.width = static_cast<int>(std::ceil(
                scaledImageSize.width / static_cast<float>(stride))) * stride;
        pad(0) = static_cast<int>(std::floor((scaledImageSize.height - minHeight) / 2.0));
        pad(1) = static_cast<int>(std::floor((scaledImageSize.width - scaledSize.width) / 2.0));
        pad(2) = scaledImageSize.height - minHeight - pad(0);
        pad(3) = scaledImageSize.width - scaledSize.width - pad(1);
        if (scaledSize.width == (inputLayerSize.width - pad(1) - pad(3))) {
            return false;
        }

        inputLayerSize.width = scaledImageSize.width;
        return true;
    }

} // namespace


