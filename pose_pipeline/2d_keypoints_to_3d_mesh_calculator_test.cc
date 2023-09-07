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

#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/strings/str_replace.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mouse_pose_analysis/pose_3d/gtest_matchers.h"
#include "mouse_pose_analysis/pose_3d/gtest_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "third_party/eigen3/Eigen/Core"

namespace mediapipe {
namespace {

constexpr char kKeypointLocationsSetTag[] = "KEYPOINT_LOCATIONS_SET";
constexpr char kRiggedMeshTag[] = "RIGGED_MESH";
constexpr char kKeypointXyzsTag[] = "KEYPOINT_XYZS";
constexpr char kXyzTag[] = "XYZ";
constexpr char kOptimizedKeypointLocationsTag[] =
    "OPTIMIZED_KEYPOINT_LOCATIONS";
constexpr char kRotationTranslationAlphasTag[] = "ROT_TRANS_ALPHA";
constexpr char kKeypointsVisualizationTag[] = "KEYPOINTS_VISUALIZATION";
constexpr char kKeypointProbsTag[] = "KEYPOINT_PROBS";
constexpr char kKeypointLocationsTag[] = "KEYPOINT_LOCATIONS";
constexpr char kInputParametersFileTag[] = "INPUT_PARAMETERS_FILE";
constexpr char kMeshOutputDirTag[] = "MESH_OUTPUT_DIR";
constexpr char kImageFrameTag[] = "IMAGE_FRAME";

static const auto kKeypointLocations =
    absl::flat_hash_map<std::string_view, std::pair<float, float>>(
        {{"NOSE", {366, 585}},           {"LEFT_EAR", {280, 569}},
         {"RIGHT_EAR", {286, 638}},      {"LEFT_SHOULDER", {239, 570}},
         {"RIGHT_SHOULDER", {246, 624}}, {"LEFT_FORE_PAW", {293, 544}},
         {"RIGHT_FORE_PAW", {252, 644}}, {"LEFT_HIP", {104, 563}},
         {"RIGHT_HIP", {63, 614}},       {"LEFT_HIND_PAW", {158, 523}},
         {"RIGHT_HIND_PAW", {34, 623}},  {"ROOT_TAIL", {17, 516}},
         {"SPINE_MID", {29, 288}},       {"SPINE_HIP", {40, 292}},
         {"LEFT_KNEE", {40, 277}},       {"RIGHT_KNEE", {30, 289}},
         {"LEFT_ELBOW", {36, 279}},      {"RIGHT_ELBOW", {38, 285}},
         {"SPINE_SHOULDER", {37, 282}},  {"NECK", {36, 277}}});

static const auto kKeypointProbs = absl::flat_hash_map<std::string_view, float>(
    {{"NOSE", 0.4},           {"LEFT_EAR", 0.4},
     {"RIGHT_EAR", 0.4},      {"LEFT_SHOULDER", 0.4},
     {"RIGHT_SHOULDER", 0.4}, {"LEFT_FORE_PAW", 0.4},
     {"RIGHT_FORE_PAW", 0.4}, {"LEFT_HIP", 0.4},
     {"RIGHT_HIP", 0.4},      {"LEFT_HIND_PAW", 0.4},
     {"RIGHT_HIND_PAW", 0.4}, {"ROOT_TAIL", 0.4},
     {"SPINE_MID", 0.4},      {"SPINE_HIP", 0.4},
     {"LEFT_KNEE", 0.4},      {"RIGHT_KNEE", 0.4},
     {"LEFT_ELBOW", 0.4},     {"RIGHT_ELBOW", 0.4},
     {"SPINE_SHOULDER", 0.4}, {"NECK", 0.4}});

std::string GetConfigFile() {
  constexpr char kConfigs[] = R"(
camera_parameters {
  intrinsics: [411.15, 0, 443.0, 0, 410.0, 332.24, 0, 0, 1]
}
skeleton_config_file: "$0/bone_config_world_standardized.csv"
#method: JOINT_ANGLE_POSE_PRIOR_AND_RIGID_FIT
method: JOINT_ANGLE
pose_prior_weight: 10000
shape_prior_weight: 10
mesh_file: "$0/simplified_skin_yf_zu.obj"
vertex_weight_file: "$0/vertex_weights_simplified_skin.csv"
mask_file: "$0/synthetic_mask_0611.png"
pose_pca_file: "$0/pca_bone_config_world.csv"
gmm_file: "$0/both_gmm_mixture_5_fixed_r0.100000.pbtxt"
shape_basis_file: "$0/pca_bone_shape_config.csv"

optimization_constraints {
  translation {
    lower_bound: -10
    upper_bound: 10
  }
  translation {
    lower_bound: -10
    upper_bound: 10
  }
  translation {
    lower_bound: 3
    upper_bound: 5
  }

  rotation {
    lower_bound: -0.4
    upper_bound: 0.4
  }
  rotation {
    lower_bound: -0.8
    upper_bound: 0.4
  }
  rotation {
    lower_bound: -2.4
    upper_bound: 2.4
  }
}

initial_rigid_body_values: [0.,0.,0.,0.,0.,4.]

optimization_options {
  max_num_iterations : 2
}
)";

  std::string config_filepath = file::JoinPath(mouse_pose::GetTestRootDir(),
                                               "pose_pipeline/testdata/");

  std::string config_string = absl::Substitute(kConfigs, config_filepath);
  std::string config_file_path = "/tmp/config.txt";
  std::ofstream config_file(config_file_path);
  config_file << config_string;

  return config_file_path;
}

TEST(KeypointsTo3DPoseShapeCalculatorTest, EmitsAllOutputs) {
  CalculatorRunner runner(R"(
    calculator: "KeypointsTo3DPoseShapeCalculator"
    input_stream: "KEYPOINT_LOCATIONS:keypoint_locations"
    input_stream: "KEYPOINT_PROBS:keypoint_probs"
    input_stream: "IMAGE_FRAME:image_frame"
    input_side_packet: "INPUT_PARAMETERS_FILE:input_params_file"
    input_side_packet: "MESH_OUTPUT_DIR:mesh_output_dir"
    output_stream: "OPTIMIZED_KEYPOINT_LOCATIONS:optimized_keypoint_locations"
    output_stream: "KEYPOINTS_VISUALIZATION:keypoints_visualization"
    output_stream: "RIGGED_MESH:rigged_mesh"
    output_stream: "XYZ:xyz"
    output_stream: "KEYPOINT_XYZS:keyed_xyz"
    output_stream: "ROT_TRANS_ALPHA:rot_trans_alpha"
    )");
  std::string config_filename = GetConfigFile();
  int width = 864;
  int height = 648;
  auto image =
      std::make_unique<ImageFrame>(ImageFormat::SRGB, width, height, 1);
  runner.MutableInputs()
      ->Tag(kImageFrameTag)
      .packets.push_back(Adopt(image.release()).At(Timestamp(0)));
  runner.MutableSidePackets()->Tag(kMeshOutputDirTag) =
      MakePacket<std::string>("/tmp").At(Timestamp(0));
  runner.MutableSidePackets()->Tag(kInputParametersFileTag) =
      MakePacket<std::string>(config_filename).At(Timestamp(0));
  auto keypoint_locations = std::make_unique<
      absl::flat_hash_map<std::string, std::pair<float, float>>>();
  auto keypoint_probs =
      std::make_unique<absl::flat_hash_map<std::string, std::vector<float>>>();
  for (const auto [name, location] : kKeypointLocations) {
    (*keypoint_locations)[name] = location;
  }
  for (const auto [name, prob] : kKeypointProbs) {
    (*keypoint_probs)[name].push_back(prob);
  }
  runner.MutableInputs()
      ->Tag(kKeypointLocationsTag)
      .packets.push_back(Adopt(keypoint_locations.release()).At(Timestamp(0)));
  runner.MutableInputs()
      ->Tag(kKeypointProbsTag)
      .packets.push_back(Adopt(keypoint_probs.release()).At(Timestamp(0)));
  MP_EXPECT_OK(runner.Run());
  EXPECT_EQ(runner.Outputs().Tag(kKeypointsVisualizationTag).packets.size(), 1);
  EXPECT_EQ(runner.Outputs().Tag(kOptimizedKeypointLocationsTag).packets.size(),
            1);
  EXPECT_EQ(runner.Outputs().Tag(kXyzTag).packets.size(), 1);
  EXPECT_EQ(runner.Outputs().Tag(kKeypointXyzsTag).packets.size(), 1);
  EXPECT_EQ(runner.Outputs().Tag(kRiggedMeshTag).packets.size(), 1);
  EXPECT_EQ(runner.Outputs().Tag(kRotationTranslationAlphasTag).packets.size(),
            1);
  MP_EXPECT_OK(file::Exists("/tmp/0of00.obj"));

  auto xyz_map =
      runner.Outputs()
          .Tag(kKeypointXyzsTag)
          .packets[0]
          .Get<absl::flat_hash_map<std::string, std::vector<float>>>();

  // Test that one keypoint is reconstructed.
  EXPECT_THAT(xyz_map, testing::Contains(testing::Key("LEFT_HIP")));
  // Test that it's a 3-vector.
  EXPECT_EQ(3, xyz_map["LEFT_HIP"].size());

  auto xyzs =
      runner.Outputs().Tag(kXyzTag).packets[0].Get<std::vector<float>>();
  EXPECT_EQ(54, xyzs.size());

  auto rot_trans_alphas = runner.Outputs()
                              .Tag(kRotationTranslationAlphasTag)
                              .packets[0]
                              .Get<std::vector<float>>();
  EXPECT_EQ(60, rot_trans_alphas.size());
}

TEST(KeypointsTo3DPoseShapeCalculatorTest, EmitsAllOutputsWithKeypointSet) {
  CalculatorRunner runner(R"(
    calculator: "KeypointsTo3DPoseShapeCalculator"
    input_stream: "KEYPOINT_LOCATIONS_SET:keypoint_locations"
    input_stream: "KEYPOINT_PROBS:keypoint_probs"
    input_stream: "IMAGE_FRAME:image_frame"
    input_side_packet: "INPUT_PARAMETERS_FILE:input_params_file"
    input_side_packet: "MESH_OUTPUT_DIR:mesh_output_dir"
    output_stream: "OPTIMIZED_KEYPOINT_LOCATIONS:optimized_keypoint_locations"
    output_stream: "KEYPOINTS_VISUALIZATION:keypoints_visualization"
    output_stream: "RIGGED_MESH:rigged_mesh"
    output_stream: "XYZ:xyz"
    output_stream: "KEYPOINT_XYZS:keyed_xyz"
    output_stream: "ROT_TRANS_ALPHA:rot_trans_alpha"
    )");
  std::string config_filename = GetConfigFile();
  int width = 864;
  int height = 648;
  auto image =
      std::make_unique<ImageFrame>(ImageFormat::SRGB, width, height, 1);
  runner.MutableInputs()
      ->Tag(kImageFrameTag)
      .packets.push_back(Adopt(image.release()).At(Timestamp(0)));
  runner.MutableSidePackets()->Tag(kMeshOutputDirTag) =
      MakePacket<std::string>("/tmp").At(Timestamp(0));
  runner.MutableSidePackets()->Tag(kInputParametersFileTag) =
      MakePacket<std::string>(config_filename).At(Timestamp(0));
  auto keypoint_locations = std::make_unique<
      absl::flat_hash_map<std::string, std::vector<std::pair<float, float>>>>();
  auto keypoint_probs =
      std::make_unique<absl::flat_hash_map<std::string, std::vector<float>>>();
  // Two sets of keypoints.
  for (const auto [name, location] : kKeypointLocations) {
    (*keypoint_locations)[name].push_back(location);
    (*keypoint_locations)[name].push_back(location);
  }
  for (const auto [name, prob] : kKeypointProbs) {
    (*keypoint_probs)[name].push_back(prob);
    (*keypoint_probs)[name].push_back(prob);
  }
  runner.MutableInputs()
      ->Tag(kKeypointLocationsSetTag)
      .packets.push_back(Adopt(keypoint_locations.release()).At(Timestamp(0)));
  runner.MutableInputs()
      ->Tag(kKeypointProbsTag)
      .packets.push_back(Adopt(keypoint_probs.release()).At(Timestamp(0)));
  MP_EXPECT_OK(runner.Run());
  EXPECT_EQ(runner.Outputs().Tag(kKeypointsVisualizationTag).packets.size(), 1);
  EXPECT_EQ(runner.Outputs().Tag(kOptimizedKeypointLocationsTag).packets.size(),
            1);
  EXPECT_EQ(runner.Outputs().Tag(kXyzTag).packets.size(), 1);
  EXPECT_EQ(runner.Outputs().Tag(kKeypointXyzsTag).packets.size(), 1);
  EXPECT_EQ(runner.Outputs().Tag(kRiggedMeshTag).packets.size(), 1);
  EXPECT_EQ(runner.Outputs().Tag(kRotationTranslationAlphasTag).packets.size(),
            1);
  MP_EXPECT_OK(file::Exists("/tmp/0of01.obj"));

  auto xyz_map =
      runner.Outputs()
          .Tag(kKeypointXyzsTag)
          .packets[0]
          .Get<absl::flat_hash_map<std::string, std::vector<float>>>();

  // Test that one keypoint is reconstructed.
  EXPECT_THAT(xyz_map, testing::Contains(testing::Key("LEFT_HIP")));
  // Test that there are two 3-vectors.
  EXPECT_EQ(6, xyz_map["LEFT_HIP"].size());

  auto rot_trans_alphas = runner.Outputs()
                              .Tag(kRotationTranslationAlphasTag)
                              .packets[0]
                              .Get<std::vector<float>>();
  EXPECT_EQ(2 * 60, rot_trans_alphas.size());
}

TEST(KeypointsTo3DPoseShapeCalculatorTest, TestKeypointLocationsSet) {
  CalculatorRunner runner(R"(
    calculator: "KeypointsTo3DPoseShapeCalculator"
    input_stream: "KEYPOINT_LOCATIONS_SET:keypoint_locations"
    input_stream: "KEYPOINT_PROBS:keypoint_probs"
    input_side_packet: "INPUT_PARAMETERS_FILE:input_params_file"
    input_side_packet: "MESH_OUTPUT_DIR:mesh_output_dir"
    output_stream: "OPTIMIZED_KEYPOINT_LOCATIONS:optimized_keypoint_locations"
    output_stream: "KEYPOINT_XYZS:keyed_xyz"
    )");
  std::string config_filename = GetConfigFile();
  runner.MutableSidePackets()->Tag(kMeshOutputDirTag) =
      MakePacket<std::string>("/tmp").At(Timestamp(0));
  runner.MutableSidePackets()->Tag(kInputParametersFileTag) =
      MakePacket<std::string>(config_filename).At(Timestamp(0));
  auto keypoint_locations = std::make_unique<
      absl::flat_hash_map<std::string, std::vector<std::pair<float, float>>>>();
  auto keypoint_probs =
      std::make_unique<absl::flat_hash_map<std::string, std::vector<float>>>();
  for (const auto [name, location] : kKeypointLocations) {
    (*keypoint_locations)[name].push_back(location);
  }
  for (const auto [name, prob] : kKeypointProbs) {
    (*keypoint_probs)[name].push_back(prob);
  }
  runner.MutableInputs()
      ->Tag(kKeypointLocationsSetTag)
      .packets.push_back(Adopt(keypoint_locations.release()).At(Timestamp(0)));
  runner.MutableInputs()
      ->Tag(kKeypointProbsTag)
      .packets.push_back(Adopt(keypoint_probs.release()).At(Timestamp(0)));
  MP_EXPECT_OK(runner.Run());
  EXPECT_EQ(runner.Outputs().Tag(kOptimizedKeypointLocationsTag).packets.size(),
            1);
  EXPECT_EQ(runner.Outputs().Tag(kKeypointXyzsTag).packets.size(), 1);

  auto xyz_map =
      runner.Outputs()
          .Tag(kKeypointXyzsTag)
          .packets[0]
          .Get<absl::flat_hash_map<std::string, std::vector<float>>>();

  // Test that one keypoint is reconstructed.
  EXPECT_THAT(xyz_map, testing::Contains(testing::Key("NOSE")));
  // Test that it's a 3-vector.
  EXPECT_EQ(3, xyz_map["NOSE"].size());
}
}  // namespace
}  // namespace mediapipe
