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

// 3D Pose optimization demo using kinematic chain and pose optimizer.
//
// The main purpose of this demo is to test and demonstrate the effectiveness
// of our 3D keypoint optimization framework.
//
// This demo constructs a kinematic chain from a formatted skeleton
// configuration file. Then based on the 2D projected keypoints positions and
// the camera matrices, the optimizer will optimize for the transformations.
// This demo outputs: a rigid aligned 3D mesh "/tmp/test_mesh.obj";
// an image shows the target 2D locations as green dots and
// the optimized projected 2D joint locations as red dots in
// "/tmp/test_projected_mesh.jpg".
//
// Use the run_pose_reconstruction_main.sh to change the flags.
// In addition to a textform protobuf that sets the inputs that are required,
// flags that needed to be changes:
// target_2d_points_filename (the targe 2D keypoints to optimize to).
// input_image_filename (the input image that we want to reconstruct pose to).
//
/* A sample protobuf with default values for a synthesized test case is
camera_parameters {
  intrinsics: [1080, 0, 432.0, 0, 1080, 324.0, 0, 0, 1]
  }
  skeleton_config_file:
  "pose_3d/synthetic_test_data/bone_config_world.csv"
  method: JOINT_ANGLE
  pose_prior_weight: 10000
  shape_prior_weight: 10
  mesh_file:
  "pose_3d/synthetic_test_data/simplified_skin_yf_zu.obj"
  vertex_weight_file:
  "pose_3d/synthetic_test_data/vertex_weights_simplified_skin.csv"
  mask_file:
  "pose_3d/synthetic_test_data/synthetic_mask_0611.png"
  pose_pca_file:
  "pose_3d/synthetic_test_data/pca_bone_config_world.csv"
  gmm_file:
  "pose_3d/synthetic_test_data/gmm_mixture_5.pbtxt"
  shape_basis_file:
  "ct-scans/skeleton-basis/pca_bone_shape_config.csv"
*/
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "glog/logging.h"
#include "mouse_pose_analysis/pose_3d/kinematic_chain.h"
#include "mouse_pose_analysis/pose_3d/pose_optimizer.h"
#include "mouse_pose_analysis/pose_3d/pose_optimizer_utils.h"
#include "mouse_pose_analysis/pose_3d/pose_reconstruction.h"
#include "mouse_pose_analysis/pose_3d/pose_reconstruction.pb.h"
#include "mouse_pose_analysis/pose_3d/pose_reconstruction_utils.h"
#include "mouse_pose_analysis/pose_3d/rigged_mesh.h"
#include "mouse_pose_analysis/pose_3d/rigged_mesh_utils.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "third_party/eigen3/Eigen/Core"

ABSL_FLAG(std::string, config_filename,
          "pose_3d/synthetic_test_data/config.pbtxt",
          "A text proto file of input parameters.");

// Each row in the csv: keypoint_name, keypoint_id, x, y
ABSL_FLAG(
    std::string, target_2d_points_filename,
    "pose_3d/synthetic_test_data/labeled_2dkp.csv",
    "The csv file which contains the position of the 2D projected points.");

ABSL_FLAG(
    std::string, input_image_filename,
    "pose_3d/synthetic_test_data/synthetic_image_0611.jpg",
    "The rendered synthetic image will be used for 2D keypoints visualization");

ABSL_FLAG(std::string, output_dir, "/tmp/",
          "The output directory for the skeleton csvs.");

ABSL_FLAG(std::string, output_joint_file, "joints.txt",
          "The optimized key points.");

namespace {

using mouse_pose::optical_mouse::InputParameters;

// Joints 0,1,2,4,7,10,13,16 are not used.
const int used_joints_mask[] = {0, 0, 0, 1, 0, 1, 1, 0, 1,
                                1, 0, 1, 1, 0, 1, 1, 0};

void ShowKeyPoints(const std::vector<Eigen::Vector2d> &key_points,
                   const Eigen::Vector3i &color, const int radius,
                   cv::Mat *image) {
  for (int i = 0; i < key_points.size(); ++i) {
    if (used_joints_mask[i]) {
      auto keypoint = key_points[i];
      cv::circle(*image, cv::Point2i(keypoint[0], keypoint[1]), radius,
                 cv::Scalar(color[0], color[1], color[2]), -1);
    }
  }
}

void VisualizeMeshAndKeyPoints(
    const mouse_pose::PoseOptimizer &pose_optimizer,
    const mouse_pose::RiggedMesh &rigged_mesh,
    const std::vector<Eigen::Vector2d> optimized_2d_keypoints) {
  cv::Mat input_image =
      mouse_pose::ReadImage(absl::GetFlag(FLAGS_input_image_filename));

  // Visualize the 2D keypoints on the input_image.
  ShowKeyPoints(pose_optimizer.GetTarget2DPoints(), Eigen::Vector3i(0, 255, 0),
                3, &input_image);
  ShowKeyPoints(optimized_2d_keypoints, Eigen::Vector3i(0, 0, 255), 2,
                &input_image);

  std::string output_mesh_overlay_filename =
      absl::GetFlag(FLAGS_output_dir) + "test_projected_mesh.jpg";
  cv::Mat mesh_image =
      ProjectMeshToImage(rigged_mesh, pose_optimizer.GetProjectionMat(),
                         input_image.cols, input_image.rows);

  cv::cvtColor(mesh_image, mesh_image, cv::COLOR_GRAY2BGR);
  cv::addWeighted(input_image, 0.9, mesh_image, 0.1, 0, mesh_image);
  cv::imwrite(output_mesh_overlay_filename, mesh_image);
}

void WriteOptimizationResults(
    const std::string filename, const std::vector<double> &rotation,
    const std::vector<double> &translation, const std::vector<float> &joint_xyz,
    const std::vector<Eigen::Vector2d> &optimized_2d_keypoints) {
  std::ofstream output_file(filename);
  // The order has to be in agreement in a skeleton file.
  std::string joint_names[] = {
      "SPINE_SHOULDER", "SPINE_MID",     "SPINE_HIP",     "LEFT_HIP",
      "LEFT_KNEE",      "LEFT_HIND_PAW", "RIGHT_HIP",     "RIGHT_KNEE",
      "RIGHT_HIND_PAW", "ROOT_TAIL",     "NECK",          "NOSE",
      "LEFT_SHOULDER",  "LEFT_ELBOW",    "LEFT_FORE_PAW", "RIGHT_SHOULDER",
      "RIGHT_ELBOW",    "RIGHT_FORE_PAW"};

  std::string buf;
  for (int i = 0; i < optimized_2d_keypoints.size(); ++i) {
    auto &p2 = optimized_2d_keypoints[i];
    float x = joint_xyz[3 * i];
    float y = joint_xyz[3 * i + 1];
    float z = joint_xyz[3 * i + 2];
    buf = absl::StrFormat("%s_2d,%f,%f,0\n", joint_names[i], p2[0], p2[1]);
    output_file << buf;
    buf = absl::StrFormat("%s_3d,%f,%f,%f\n", joint_names[i], x, y, z);
    output_file << buf;
  }

  buf = absl::StrFormat("Rotation,%f,%f,%f\n", rotation[0], rotation[1],
                        rotation[2]);
  output_file << buf;
  buf = absl::StrFormat("Translation,%f,%f,%f\n", translation[0],
                        translation[1], translation[2]);
  output_file << buf;
}

}  // namespace

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);

  mouse_pose::PoseOptimizer pose_optimizer;

  InputParameters input_parameters;
  std::ifstream proto_file(absl::GetFlag(FLAGS_config_filename));
  std::stringstream buf;
  buf << proto_file.rdbuf();
  ::google::protobuf::TextFormat::ParseFromString(buf.str(), &input_parameters);

  QCHECK_OK(mouse_pose::LoadTargetPointsFromFile(
      absl::GetFlag(FLAGS_target_2d_points_filename),
      &(pose_optimizer.GetTarget2DPoints())));

  std::vector<double> trans_params(3, 0);
  std::vector<double> rotate_params(3, 0);
  std::vector<double> alphas(18 * 3, 0);
  std::vector<double> shape_basis_weights(mouse_pose::kNumShapeBasisComponents,
                                          0.);

  std::vector<double> weights(mouse_pose::kNumJointsToOptimize, 1.0);
  const auto &target_2d_points = pose_optimizer.GetTarget2DPoints();
  for (int i = 0; i < target_2d_points.size(); ++i) {
    auto &p2 = target_2d_points[i];
    // Skip unused joints, which have (0,0) in the input CSV file.
    if (p2[0] <= 0 && p2[1] <= 0) {
      weights[i] = 0.;
    }
  }

  SetUpOptimizer(input_parameters, &pose_optimizer);
  CHECK_OK(ReconstructPose(input_parameters, &pose_optimizer, weights,
                           &trans_params, &rotate_params, &alphas,
                           &shape_basis_weights));

  std::vector<Eigen::Vector2d> optimized_2d_keypoints;
  std::vector<Eigen::Matrix4d> chain_transformations;
  std::vector<float> joint_xyz;
  CHECK_OK(UpdateChainAndProjectKeypoints(
      alphas, Eigen::Vector3d(rotate_params.data()),
      Eigen::Vector3d(trans_params.data()), pose_optimizer,
      &optimized_2d_keypoints, &chain_transformations, &joint_xyz));

  // Read in the obj file and do the transformation and output the result obj.
  auto rigged_mesh = mouse_pose::CreateRiggedMeshFromFiles(
      input_parameters.mesh_file(), input_parameters.vertex_weight_file());

  std::string output_obj_filename =
      absl::GetFlag(FLAGS_output_dir) + "test_mesh.obj";
  CHECK_OK(TransformAndSaveMesh(output_obj_filename, chain_transformations,
                                trans_params, rotate_params,
                                rigged_mesh.get()));

  VisualizeMeshAndKeyPoints(pose_optimizer, *rigged_mesh,
                            optimized_2d_keypoints);

  std::string output_filename =
      absl::GetFlag(FLAGS_output_dir) + absl::GetFlag(FLAGS_output_joint_file);
  WriteOptimizationResults(output_filename, rotate_params, trans_params,
                           joint_xyz, optimized_2d_keypoints);
  return 0;
}
