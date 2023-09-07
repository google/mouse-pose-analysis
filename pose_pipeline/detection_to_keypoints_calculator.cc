// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_split.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mouse_pose_analysis/pose_pipeline/detection_to_keypoints_calculator.pb.h"
#include "tensorflow/lite/interpreter.h"

namespace mediapipe {
// Calculator to parse the tensors from a detection model.
// The keypoints from the image are assumed to be in normalized coordinates
// between 0 and 1. If an image is provided in the input stream, the output
// keypoints will be transformed in pixels.
//
// Example use:
//
// node {
//  calculator: "DetectionToKeypointsCalculator"
//  input_stream: "TENSORS:detection_tensors"
//  input_stream: "IMAGE:rgb_frames"
//  output_stream: "KEYPOINTS: keypoints"
// }

namespace {
constexpr int num_keypoints_ = 20;

constexpr char kTensorsTag[] = "TENSORS";
constexpr char kImageTag[] = "IMAGE";
constexpr char kKeypointsTag[] = "KEYPOINTS";
constexpr char kKeypointScoresTag[] = "KEYPOINT_SCORES";

using Point2d = std::pair<float, float>;
using KeypointsSet = std::vector<Point2d>;

}  // namespace

class DetectionToKeypointsCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) final;
  absl::Status Process(CalculatorContext* cc) final;

 private:
  void ConvertToBestMouseKeypoints(
      const std::vector<KeypointsSet>& keypoints,
      const std::vector<std::vector<float>>& keypoint_scores,
      const std::vector<float>& detection_scores,
      absl::flat_hash_map<std::string, std::vector<Point2d>>* best_keypoints,
      absl::flat_hash_map<std::string, std::vector<float>>*
          best_keypoint_scores);

  std::vector<std::string> keypoint_names_;
  int num_keypoints_;
};
REGISTER_CALCULATOR(DetectionToKeypointsCalculator);

absl::Status DetectionToKeypointsCalculator::GetContract(
    CalculatorContract* cc) {
  cc->Inputs().Tag(kTensorsTag).Set<std::vector<TfLiteTensor>>();

  if (cc->Inputs().HasTag(kImageTag))
    cc->Inputs().Tag(kImageTag).Set<ImageFrame>();
  cc->Outputs()
      .Tag(kKeypointsTag)
      .Set<absl::flat_hash_map<std::string, std::vector<Point2d>>>();
  if (cc->Outputs().HasTag(kKeypointScoresTag))
    cc->Outputs()
        .Tag(kKeypointScoresTag)
        .Set<absl::flat_hash_map<std::string, std::vector<float>>>();
  return absl::OkStatus();
}

void DetectionToKeypointsCalculator::ConvertToBestMouseKeypoints(
    const std::vector<KeypointsSet>& keypoints,
    const std::vector<std::vector<float>>& keypoint_scores,
    const std::vector<float>& detection_scores,
    absl::flat_hash_map<std::string, std::vector<Point2d>>* best_keypoints,
    absl::flat_hash_map<std::string, std::vector<float>>*
        best_keypoint_scores) {
  CHECK(keypoints.size() == keypoint_scores.size() &&
        keypoints.size() == detection_scores.size())
      << "Number of detections mis-matches.";
  int best_detection =
      std::max_element(detection_scores.begin(), detection_scores.end()) -
      detection_scores.begin();
  for (int i = 0; i < keypoint_names_.size(); ++i) {
    auto name = keypoint_names_[i];
    (*best_keypoints)[name].push_back(keypoints[best_detection][i]);
    (*best_keypoint_scores)[name].push_back(keypoint_scores[best_detection][i]);
  }
}

absl::Status DetectionToKeypointsCalculator::Open(CalculatorContext* cc) {
  const auto& options =
      cc->Options<mediapipe::DetectionToKeypointsCalculatorOptions>();
  keypoint_names_ = absl::StrSplit(options.keypoint_names(), ',');
  num_keypoints_ = keypoint_names_.size();
  return absl::OkStatus();
}

absl::Status DetectionToKeypointsCalculator::Process(CalculatorContext* cc) {
  const auto& input_tensors =
      cc->Inputs().Tag(kTensorsTag).Get<std::vector<TfLiteTensor>>();
  RET_CHECK(!input_tensors.empty());

  // keypoints is of the shape num_detections * num_keypoints * 2
  // keypoint_scores num_detections * num_keypoints
  // detection_scores num_detections
  std::vector<KeypointsSet> keypoints;
  std::vector<std::vector<float>> keypoint_scores;
  std::vector<float> detection_scores;

  // Parse the tensors.
  for (const auto& t : input_tensors) {
    if (t.dims->size < 0) {
      ABSL_LOG(WARNING) << "Unexpected tensor " << t.name
                        << "with dims: " << (t.dims)->size;
      break;
    }

    // Check the first dimension is the batch dimension.
    RET_CHECK(t.dims->data[0] == 1);

    const float* tensor_data = t.data.f;

    const auto& tensor_detection_map =
        cc->Options<mediapipe::DetectionToKeypointsCalculatorOptions>()
            .tensor_detection_map();
    std::string detection_cat = tensor_detection_map.at(t.name);
    if (detection_cat == "keypoints") {
      int num_detections = t.dims->data[1];
      int offset = 0;
      for (int d = 0; d < num_detections; ++d) {
        KeypointsSet kp_one_set;
        for (int k = 0; k < num_keypoints_; ++k) {
          float y = tensor_data[offset++];
          float x = tensor_data[offset++];
          kp_one_set.push_back({x, y});
        }
        keypoints.push_back(kp_one_set);
      }
    } else if (detection_cat == "keypoint_scores") {
      int num_detections = t.dims->data[1];
      int num_values = 1;
      for (int td = 0; td < t.dims->size; ++td) num_values *= t.dims->data[td];
      std::vector<float> all_scores(tensor_data, tensor_data + num_values);

      std::vector<float>::const_iterator offset = all_scores.begin();
      for (int d = 0; d < num_detections; ++d) {
        std::vector<float> scores_one_set(offset, offset + num_keypoints_);
        keypoint_scores.push_back(scores_one_set);
        offset += num_keypoints_;
      }
    } else if (detection_cat == "scores") {
      int num_detections = t.dims->data[1];
      std::copy(tensor_data, tensor_data + num_detections,
                std::back_inserter(detection_scores));
    }
  }

  absl::flat_hash_map<std::string, std::vector<Point2d>> input_keypoints;
  absl::flat_hash_map<std::string, std::vector<float>> input_keypoint_scores;
  ConvertToBestMouseKeypoints(keypoints, keypoint_scores, detection_scores,
                              &input_keypoints, &input_keypoint_scores);

  auto output_keypoints = std::make_unique<
      absl::flat_hash_map<std::string, std::vector<Point2d>>>();
  auto output_keypoint_scores =
      std::make_unique<absl::flat_hash_map<std::string, std::vector<float>>>();

  // Transform to pixels if image's given and set missing keypoint information.
  int height = 0, width = 0;
  if (cc->Inputs().HasTag(kImageTag)) {
    const auto& image_frame = cc->Inputs().Tag(kImageTag).Get<ImageFrame>();
    height = image_frame.Height();
    width = image_frame.Width();
  }

  for (auto& keypoint_name : keypoint_names_) {
    auto keypoint_iter = input_keypoints.find(keypoint_name);
    if (keypoint_iter != input_keypoints.end()) {
      if (height > 0) {
        for (auto& kp : keypoint_iter->second) {
          kp.first *= width;
          kp.second *= height;
        }
      }
      (*output_keypoints)[keypoint_name] = keypoint_iter->second;
      (*output_keypoint_scores)[keypoint_name] =
          input_keypoint_scores[keypoint_name];
    } else {
      LOG(INFO) << "Missing keypoint:" << keypoint_name;
      (*output_keypoints)[keypoint_name].push_back(Point2d(-1., -1.));
      (*output_keypoint_scores)[keypoint_name].push_back(0.);
    }
  }
  cc->Outputs()
      .Tag(kKeypointsTag)
      .Add(output_keypoints.release(), cc->InputTimestamp());
  if (cc->Outputs().HasTag(kKeypointScoresTag)) {
    cc->Outputs()
        .Tag(kKeypointScoresTag)
        .Add(output_keypoint_scores.release(), cc->InputTimestamp());
  }
  return absl::OkStatus();
}
}  // namespace mediapipe
