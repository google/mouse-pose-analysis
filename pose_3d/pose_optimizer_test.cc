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

#include "mouse_pose_analysis/pose_3d/pose_optimizer.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "google/protobuf/text_format.h"
#include "googlemock/include/gmock/gmock.h"
#include "gtest/gtest.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "mouse_pose_analysis/pose_3d/cost_functions.h"
#include "mouse_pose_analysis/pose_3d/gtest_matchers.h"
#include "mouse_pose_analysis/pose_3d/gtest_util.h"
#include "mouse_pose_analysis/pose_3d/pose_optimizer_utils.h"
#include "mouse_pose_analysis/pose_3d/pose_reconstruction.pb.h"
#include "mouse_pose_analysis/pose_3d/rigged_mesh_utils.h"
#include "third_party/eigen3/Eigen/Core"

namespace mouse_pose {
namespace {

using ::testing::Test;

class PoseOptimizerTest : public Test {
 protected:
  PoseOptimizerTest() {
    test_files_dir_ = GetTestRootDir() + "pose_3d/testdata/";
  }
  std::string test_files_dir_;
  PoseOptimizer pose_optimizer_;
};

TEST_F(PoseOptimizerTest, LoadAndConstructKinematicChain) {
  EXPECT_OK(LoadAndConstructKinematicChain(
      absl::StrCat(test_files_dir_, "test_skeleton.csv"),
      &(pose_optimizer_.GetKinematicChain())));
  EXPECT_EQ(pose_optimizer_.GetKinematicChain().GetBones().size(), 5);
  EXPECT_EQ(pose_optimizer_.GetKinematicChain().GetJoints().size(), 5);
}

TEST_F(PoseOptimizerTest, LoadTargetPointsFromFile) {
  EXPECT_OK(LoadTargetPointsFromFile(test_files_dir_ + "test_skeleton_2d.csv",
                                     &(pose_optimizer_.GetTarget2DPoints())));
}

TEST_F(PoseOptimizerTest, LoadProjectionMatFromFile) {
  EXPECT_OK(LoadProjectionMatFromFile(test_files_dir_ + "test_camera_P3x4.txt",
                                      &(pose_optimizer_.GetProjectionMat())));
  Eigen::Matrix<double, 3, 4> expected_projection_mat =
      Eigen::Matrix<double, 3, 4>::Identity();
  expected_projection_mat(2, 3) = 1.;
  EXPECT_THAT(pose_optimizer_.GetProjectionMat(),
              mouse_pose::test::EigenMatrixEq(expected_projection_mat));
}

TEST_F(PoseOptimizerTest, LoadMouseTargetPointsFromMap) {
  ASSERT_OK(LoadAndConstructKinematicChain(
      absl::StrCat(test_files_dir_, "test_mouse_skeleton.csv"),
      &(pose_optimizer_.GetKinematicChain())));
  auto keypoint_locations = std::make_unique<
      absl::flat_hash_map<std::string, std::pair<float, float>>>();
  auto keypoint_probs =
      std::make_unique<absl::flat_hash_map<std::string, float>>();
  std::vector<std::string> keypoint_names(
      {"head", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
       "left_fore_paw", "right_fore_paw", "left_hip", "right_hip",
       "left_hind_paw", "right_hind_paw", "root_tail", "mid_tail", "tip_tail"});
  for (auto& keypoint_name : keypoint_names) {
    (*keypoint_locations)[keypoint_name] = std::pair<float, float>(1, 1);
    (*keypoint_probs)[keypoint_name] = 0.3;
  }
  auto keypoint_weights = std::make_unique<std::vector<double>>();
  EXPECT_OK(pose_optimizer_.LoadMouseTargetPointsFromMap(
      *keypoint_locations, *keypoint_probs, keypoint_weights.get()));
  std::vector<double> expected_keypoint_weights(18, 0.3);
  std::vector<int> non_pairing_keypoint_indices({0, 1, 4, 7, 13, 16});
  for (auto& non_pairing_keypoint_index : non_pairing_keypoint_indices) {
    expected_keypoint_weights[non_pairing_keypoint_index] = 0.;
  }
  int neck_index = 2;
  int spine_hip_index = 10;
  expected_keypoint_weights[neck_index] = 1.;
  expected_keypoint_weights[spine_hip_index] = 1.;
  EXPECT_THAT(*keypoint_weights,
              ::testing::Pointwise(::testing::DoubleNear(1e-7),
                                   expected_keypoint_weights));
  for (int i = 0; i < 18; ++i) {
    if (std::find(non_pairing_keypoint_indices.begin(),
                  non_pairing_keypoint_indices.end(),
                  i) != non_pairing_keypoint_indices.end()) {
      EXPECT_THAT(pose_optimizer_.GetTarget2DPoints()[i],
                  mouse_pose::test::EigenMatrixEq(Eigen::Vector2d::Zero()));
    } else {
      EXPECT_THAT(pose_optimizer_.GetTarget2DPoints()[i],
                  mouse_pose::test::EigenMatrixEq(Eigen::Vector2d(1., 1.)));
    }
  }
}

TEST_F(PoseOptimizerTest, LoadMouseTargetPointsFromBodyJoints) {
  ASSERT_OK(LoadAndConstructKinematicChain(
      absl::StrCat(test_files_dir_, "test_mouse_skeleton.csv"),
      &(pose_optimizer_.GetKinematicChain())));

  optical_mouse::BodyJoints bj;
  ::google::protobuf::TextFormat::ParseFromString(
      R"pb(
        key_points {
          name: "nose"
          position_2d: [ 1, 1 ]
        }
        key_points {
          name: "spine_hip"
          position_2d: [ 1, 1 ]
        }
      )pb",
      &bj);
  std::vector<double> keypoint_weights;
  EXPECT_OK(
      pose_optimizer_.LoadTargetPointsFromBodyJoints(bj, &keypoint_weights));
  EXPECT_THAT(pose_optimizer_.GetTarget2DPoints()[2],
              mouse_pose::test::EigenMatrixEq(Eigen::Vector2d(1., 1.)));
}

TEST_F(PoseOptimizerTest, RigidFit) {
  ASSERT_OK(LoadAndConstructKinematicChain(
      absl::StrCat(test_files_dir_, "test_mouse_skeleton.csv"),
      &(pose_optimizer_.GetKinematicChain())));
  ASSERT_OK(LoadTargetPointsFromFile(
      absl::StrCat(test_files_dir_, "test_mouse_skeleton_2d.csv"),
      &(pose_optimizer_.GetTarget2DPoints())));
  ASSERT_OK(LoadProjectionMatFromFile(
      absl::StrCat(test_files_dir_, "test_mouse_camera_P3x4.txt"),
      &(pose_optimizer_.GetProjectionMat())));
  std::vector<double> trans_params(3, 0);
  std::vector<double> rotate_params(3, 0);
  std::vector<double> weights(kNumJointsToOptimize, 1.);
  ceres::Solver::Options options;
  options.num_threads = 5;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.max_num_iterations = 10000;
  options.minimizer_progress_to_stdout = true;
  options.logging_type = ceres::SILENT;
  ASSERT_OK(pose_optimizer_.RigidFit<18>(&trans_params, &rotate_params, weights,
                                         options));
  std::vector<double> expected_trans_params(3, 0);
  std::vector<double> expected_rotate_params(3, 0);
  EXPECT_THAT(trans_params, ::testing::Pointwise(::testing::DoubleNear(1e-5),
                                                 expected_trans_params));
  EXPECT_THAT(rotate_params, ::testing::Pointwise(::testing::DoubleNear(1e-5),
                                                  expected_rotate_params));
  trans_params = std::vector<double>(3, 1.);
  rotate_params = std::vector<double>(3, 1.);
  EXPECT_OK(pose_optimizer_.RigidFit<18>(&trans_params, &rotate_params, weights,
                                         options));
  EXPECT_THAT(trans_params, ::testing::Pointwise(::testing::DoubleNear(1e-5),
                                                 expected_trans_params));
  EXPECT_THAT(rotate_params, ::testing::Pointwise(::testing::DoubleNear(1e-5),
                                                  expected_rotate_params));

  trans_params = std::vector<double>(3, 1.);
  rotate_params = std::vector<double>(3, 1.);
  // Test empty constraints.
  auto trans_constraints =
      std::make_unique<absl::flat_hash_map<int, std::pair<double, double>>>();
  auto rotate_constraints =
      std::make_unique<absl::flat_hash_map<int, std::pair<double, double>>>();
  ASSERT_OK(pose_optimizer_.RigidFit<18>(&trans_params, &rotate_params, weights,
                                         options, trans_constraints.get(),
                                         rotate_constraints.get()));
  EXPECT_THAT(trans_params, ::testing::Pointwise(::testing::DoubleNear(1e-5),
                                                 expected_trans_params));
  EXPECT_THAT(rotate_params, ::testing::Pointwise(::testing::DoubleNear(1e-5),
                                                  expected_rotate_params));
  // Test hard constraints.
  trans_params = std::vector<double>(3, 1.);
  rotate_params = std::vector<double>(3, 1.);
  (*trans_constraints)[0] = std::pair<double, double>(0.99, 1.001);
  ASSERT_OK(pose_optimizer_.RigidFit<18>(&trans_params, &rotate_params, weights,
                                         options, trans_constraints.get(),
                                         rotate_constraints.get()));
  EXPECT_NEAR(trans_params[0], 0.99, 1e-5);
}

TEST_F(PoseOptimizerTest, RigidBodyComputation) {
  ASSERT_OK(LoadAndConstructKinematicChain(
      absl::StrCat(test_files_dir_, "test_mouse_skeleton.csv"),
      &(pose_optimizer_.GetKinematicChain())));
  ASSERT_OK(LoadTargetPointsFromFile(
      absl::StrCat(test_files_dir_, "test_mouse_skeleton_2d.csv"),
      &(pose_optimizer_.GetTarget2DPoints())));
  ASSERT_OK(LoadProjectionMatFromFile(
      absl::StrCat(test_files_dir_, "test_mouse_camera_P3x4.txt"),
      &(pose_optimizer_.GetProjectionMat())));
  std::vector<double> trans_params(3, 0);
  std::vector<double> rotate_params(3, 0);
  std::vector<double> weights(kNumJointsToOptimize, 1.);
  pose_optimizer_.ComputeRigidBodyTransform(weights, &trans_params,
                                            &rotate_params);
  // The expected values are translation and rotation inside
  // test_mouse_camera_P3x4, which ComputeRigidBody ignores and should recover.
  std::vector<double> expected_trans_params{0.29674639, 0.79565492, 4.25483763};
  std::vector<double> expected_rotate_params{1.3278910, -2.8182223, 0.08299581};

  // Check the depth component only.
  EXPECT_NEAR(expected_trans_params[2], trans_params[2], 1e-2);
  EXPECT_THAT(rotate_params, ::testing::Pointwise(::testing::DoubleNear(1e-5),
                                                  expected_rotate_params));
}

TEST_F(PoseOptimizerTest, DISABLED_JointAngleFit) {
  ASSERT_OK(LoadAndConstructKinematicChain(
      absl::StrCat(test_files_dir_, "test_mouse_skeleton.csv"),
      &(pose_optimizer_.GetKinematicChain())));
  ASSERT_OK(LoadTargetPointsFromFile(
      absl::StrCat(test_files_dir_, "test_mouse_skeleton_2d.csv"),
      &(pose_optimizer_.GetTarget2DPoints())));
  ASSERT_OK(LoadProjectionMatFromFile(
      absl::StrCat(test_files_dir_, "test_mouse_camera_P3x4.txt"),
      &(pose_optimizer_.GetProjectionMat())));
  std::vector<double> trans_params(3, 0);
  std::vector<double> rotate_params(3, 0);
  std::vector<double> weights(kNumJointsToOptimize, 1.);
  ceres::Solver::Options options;
  options.num_threads = 5;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.max_num_iterations = 10000;
  options.minimizer_progress_to_stdout = true;
  options.logging_type = ceres::SILENT;
  ASSERT_OK(pose_optimizer_.RigidFit<18>(&trans_params, &rotate_params, weights,
                                         options));
  std::vector<double> alphas(kNumJointAnglesToOptimize, 0);
  std::vector<double> expected_alphas(kNumJointAnglesToOptimize, 0);
  pose_optimizer_.JointAngleFit<18>(Eigen::Vector3d(rotate_params.data()),
                                    Eigen::Vector3d(trans_params.data()),
                                    &alphas, weights);
  EXPECT_THAT(alphas, ::testing::Pointwise(::testing::DoubleNear(1e-3),
                                           expected_alphas));
  alphas = std::vector<double>(kNumJointAnglesToOptimize, 0.01);
  pose_optimizer_.JointAngleFit<18>(Eigen::Vector3d(rotate_params.data()),
                                    Eigen::Vector3d(trans_params.data()),
                                    &alphas, weights);
  EXPECT_THAT(alphas, ::testing::Pointwise(::testing::DoubleNear(0.6),
                                           expected_alphas));

  // Test empty constraints.
  auto angles_constraints =
      std::make_unique<absl::flat_hash_map<int, std::pair<double, double>>>();
  pose_optimizer_.JointAngleFit<18>(Eigen::Vector3d(rotate_params.data()),
                                    Eigen::Vector3d(trans_params.data()),
                                    &alphas, weights, options,
                                    angles_constraints.get());
  EXPECT_THAT(alphas, ::testing::Pointwise(::testing::DoubleNear(0.6),
                                           expected_alphas));
  // Test one hard constraint.
  alphas = std::vector<double>(kNumJointAnglesToOptimize, 0.5);
  (*angles_constraints)[1] = std::pair<double, double>(0.4, 0.6);
  pose_optimizer_.JointAngleFit<18>(Eigen::Vector3d(rotate_params.data()),
                                    Eigen::Vector3d(trans_params.data()),
                                    &alphas, weights, options,
                                    angles_constraints.get());
  EXPECT_NEAR(alphas[1], 0.6, 1e-5);
}

TEST_F(PoseOptimizerTest, RigidFitTo3D) {
  ASSERT_OK(LoadAndConstructKinematicChain(
      absl::StrCat(test_files_dir_, "test_skeleton.csv"),
      &(pose_optimizer_.GetKinematicChain())));
  std::vector<Eigen::Vector3d> target_points =
      pose_optimizer_.GetKinematicChain().GetJoints();
  std::vector<double> weights;
  for (int i = 0; i < target_points.size(); ++i) {
    target_points[i] = target_points[i] + Eigen::Vector3d::Ones();
    weights.push_back(1.0);
  }
  pose_optimizer_.GetTarget3DPoints() = target_points;
  std::vector<double> trans_params(3, 0);
  std::vector<double> rotate_params(3, 0);
  ASSERT_OK(
      pose_optimizer_.RigidFitTo3D(weights, &trans_params, &rotate_params));
  std::vector<double> expected_trans_params(3, 1);
  std::vector<double> expected_rotate_params(3, 0);
  EXPECT_THAT(trans_params, ::testing::Pointwise(::testing::DoubleNear(1e-8),
                                                 expected_trans_params));
  EXPECT_THAT(rotate_params, ::testing::Pointwise(::testing::DoubleNear(1e-8),
                                                  expected_rotate_params));
  trans_params = std::vector<double>(3, 1.);
  rotate_params = std::vector<double>(3, 1.);
  EXPECT_OK(
      pose_optimizer_.RigidFitTo3D(weights, &trans_params, &rotate_params));
  EXPECT_THAT(trans_params, ::testing::Pointwise(::testing::DoubleNear(1e-8),
                                                 expected_trans_params));
  EXPECT_THAT(rotate_params, ::testing::Pointwise(::testing::DoubleNear(1e-8),
                                                  expected_rotate_params));
}

TEST_F(PoseOptimizerTest, JointAngleFitTo3D) {
  ASSERT_OK(LoadAndConstructKinematicChain(
      absl::StrCat(test_files_dir_, "test_mouse_skeleton.csv"),
      &(pose_optimizer_.GetKinematicChain())));
  std::vector<Eigen::Vector3d> target_points =
      pose_optimizer_.GetKinematicChain().GetJoints();
  std::vector<double> weights;
  for (int i = 0; i < target_points.size(); ++i) {
    target_points[i] = target_points[i] + Eigen::Vector3d::Ones();
    weights.push_back(1.0);
  }
  pose_optimizer_.GetTarget3DPoints() = target_points;
  std::vector<double> trans_params(3, 0);
  std::vector<double> rotate_params(3, 0);
  std::vector<double> alphas(kNumJointAnglesToOptimize, 0);
  std::vector<double> expected_alphas(kNumJointAnglesToOptimize, 0);
  ASSERT_OK(
      pose_optimizer_.RigidFitTo3D(weights, &trans_params, &rotate_params));
  pose_optimizer_.JointAngleFitTo3D(weights, &trans_params, &rotate_params,
                                    &alphas);
  EXPECT_THAT(alphas, ::testing::Pointwise(::testing::DoubleNear(1e-3),
                                           expected_alphas));
}

TEST_F(PoseOptimizerTest, JointAngleAndRigidFit) {
  ASSERT_OK(LoadAndConstructKinematicChain(
      absl::StrCat(test_files_dir_, "test_mouse_skeleton.csv"),
      &(pose_optimizer_.GetKinematicChain())));
  ASSERT_OK(LoadTargetPointsFromFile(
      absl::StrCat(test_files_dir_, "test_mouse_skeleton_2d.csv"),
      &(pose_optimizer_.GetTarget2DPoints())));
  ASSERT_OK(LoadProjectionMatFromFile(
      absl::StrCat(test_files_dir_, "test_mouse_camera_P3x4.txt"),
      &(pose_optimizer_.GetProjectionMat())));
  std::vector<double> trans_params(3, 0.);
  std::vector<double> rotate_params(3, 0.);
  std::vector<double> alphas(kNumJointAnglesToOptimize, 0.);
  std::vector<double> expected_alphas(kNumJointAnglesToOptimize, 0.);
  std::vector<double> weights(kNumJointsToOptimize, 1.);
  ceres::Solver::Options options;
  options.num_threads = 5;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.max_num_iterations = 10000;
  options.minimizer_progress_to_stdout = true;
  options.logging_type = ceres::SILENT;
  ASSERT_OK(pose_optimizer_.RigidFit<18>(&trans_params, &rotate_params, weights,
                                         options));
  pose_optimizer_.JointAngleAndRigidFit(&trans_params, &rotate_params, &alphas,
                                        weights);
  // Hard to know what the expected value should be, so just test that it runs
  // and changes the angle value.
  EXPECT_NE(0., alphas[0]);
}

TEST_F(PoseOptimizerTest, JointAnglePosePriorAndRigidFit) {
  ASSERT_OK(LoadAndConstructKinematicChain(
      absl::StrCat(test_files_dir_, "test_mouse_skeleton.csv"),
      &(pose_optimizer_.GetKinematicChain())));
  ASSERT_OK(LoadTargetPointsFromFile(
      absl::StrCat(test_files_dir_, "test_mouse_skeleton_2d.csv"),
      &(pose_optimizer_.GetTarget2DPoints())));
  ASSERT_OK(LoadProjectionMatFromFile(
      absl::StrCat(test_files_dir_, "test_mouse_camera_P3x4.txt"),
      &(pose_optimizer_.GetProjectionMat())));
  ASSERT_OK(pose_optimizer_.LoadGmmFromFile(
      absl::StrCat(test_files_dir_, "gmm_mixture_5.pbtxt")));
  std::vector<double> trans_params(3, 0.);
  std::vector<double> rotate_params(3, 0.);
  std::vector<double> alphas(kNumJointAnglesToOptimize, 0.);
  std::vector<double> expected_alphas(kNumJointAnglesToOptimize, 0.);
  std::vector<double> weights(kNumJointsToOptimize, 1.);
  double pose_prior_weight = 1.;
  ceres::Solver::Options options;
  options.num_threads = 5;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.max_num_iterations = 10000;
  options.minimizer_progress_to_stdout = true;
  options.logging_type = ceres::SILENT;
  ASSERT_OK(pose_optimizer_.RigidFit<18>(&trans_params, &rotate_params, weights,
                                         options));
  pose_optimizer_.JointAnglePosePriorAndRigidFit(
      &trans_params, &rotate_params, &alphas, weights, pose_prior_weight);
  // Hard to know what the expected value should be, so just test that it runs
  // and changes the angle value.
  EXPECT_NE(0., alphas[0]);
}

TEST_F(PoseOptimizerTest, JointAnglePosePriorShapeBasisAndRigidFit) {
  ASSERT_OK(LoadAndConstructKinematicChain(
      absl::StrCat(test_files_dir_, "test_mouse_skeleton.csv"),
      &(pose_optimizer_.GetKinematicChain())));
  ASSERT_OK(LoadTargetPointsFromFile(
      absl::StrCat(test_files_dir_, "test_mouse_skeleton_2d.csv"),
      &(pose_optimizer_.GetTarget2DPoints())));
  ASSERT_OK(LoadProjectionMatFromFile(
      absl::StrCat(test_files_dir_, "test_mouse_camera_P3x4.txt"),
      &(pose_optimizer_.GetProjectionMat())));
  ASSERT_OK(pose_optimizer_.LoadGmmFromFile(
      absl::StrCat(test_files_dir_, "gmm_mixture_5.pbtxt")));
  ASSERT_OK(pose_optimizer_.GetKinematicChain().ReadShapePcaBasisConfig(
      absl::StrCat(test_files_dir_, "pca_bone_shape_config.csv")));

  std::vector<double> trans_params(3, 0.);
  std::vector<double> rotate_params(3, 0.);
  std::vector<double> alphas(kNumJointAnglesToOptimize, 0.);
  std::vector<double> expected_alphas(kNumJointAnglesToOptimize, 0.);
  std::vector<double> weights(kNumJointsToOptimize, 1.);
  std::vector<double> shape_betas(kNumShapeBasisComponents, 0.);
  double pose_prior_weight = 1.;
  double shape_basis_l2_weight = 1.;
  ceres::Solver::Options options;
  options.num_threads = 5;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.max_num_iterations = 10000;
  options.minimizer_progress_to_stdout = true;
  options.logging_type = ceres::SILENT;
  ASSERT_OK(pose_optimizer_.RigidFit<18>(&trans_params, &rotate_params, weights,
                                         options));
  pose_optimizer_.JointAnglePosePriorShapeBasisAndRigidFit(
      &trans_params, &rotate_params, &alphas, &shape_betas, weights,
      pose_prior_weight, shape_basis_l2_weight);
  // Hard to know what the expected value should be, so just test that it runs
  // and changes the angle value.
  EXPECT_NE(0., alphas[0]);
  EXPECT_NE(0., shape_betas[0]);
}

TEST_F(PoseOptimizerTest, JointAnglePcaFit) {
  ASSERT_OK(LoadAndConstructKinematicChain(
      absl::StrCat(test_files_dir_, "test_mouse_skeleton.csv"),
      &(pose_optimizer_.GetKinematicChain())));
  ASSERT_OK(pose_optimizer_.GetKinematicChain().ReadAnglePcaBasisConfig(
      absl::StrCat(test_files_dir_, "pca_bone_config_world.csv")));
  ASSERT_OK(LoadTargetPointsFromFile(
      absl::StrCat(test_files_dir_, "test_mouse_skeleton_2d.csv"),
      &(pose_optimizer_.GetTarget2DPoints())));
  ASSERT_OK(LoadProjectionMatFromFile(
      absl::StrCat(test_files_dir_, "test_mouse_camera_P3x4.txt"),
      &(pose_optimizer_.GetProjectionMat())));
  std::vector<double> trans_params(3, 0.);
  std::vector<double> rotate_params(3, 0.);
  std::vector<double> alphas(kNumJointAnglesToOptimize, 0.);
  std::vector<double> expected_alphas(kNumJointAnglesToOptimize, 0.);
  std::vector<double> weights(kNumJointsToOptimize, 1.);
  ceres::Solver::Options options;
  options.num_threads = 5;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.max_num_iterations = 10000;
  options.minimizer_progress_to_stdout = true;
  options.logging_type = ceres::SILENT;
  ASSERT_OK(pose_optimizer_.RigidFit<18>(&trans_params, &rotate_params, weights,
                                         options));
  pose_optimizer_.JointAnglePcaFit(&trans_params, &rotate_params, &alphas,
                                   weights);
  // Hard to know what the expected value should be, so just test that it runs
  // and changes the angle value.
  EXPECT_NE(0., alphas[0]);
}
}  // namespace
}  // namespace mouse_pose
