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

#include "mouse_pose_analysis/pose_3d/pose_reconstruction.h"

#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "glog/logging.h"
#include "googlemock/include/gmock/gmock.h"
#include "gtest/gtest.h"
#include "mouse_pose_analysis/pose_3d/gtest_matchers.h"
#include "mouse_pose_analysis/pose_3d/gtest_util.h"
#include "mouse_pose_analysis/pose_3d/pose_optimizer.h"
#include "mouse_pose_analysis/pose_3d/pose_reconstruction.pb.h"

namespace mouse_pose {
namespace {
using mouse_pose::optical_mouse::InputParameters;

class PoseReconstructionTest : public testing::Test {
 protected:
  void SetUp() override {
    testdata_path_ = GetTestRootDir() + "pose_3d/testdata/";

    std::string input_text = absl::StrFormat(
        R"pb(
          camera_parameters {
            # Extracted from the projection matrix in the test directory.
            intrinsics: [ 1200, 0, 480.0, 0, 1200, 270.0, 0, 0, 1 ]
          }
          skeleton_config_file: "%1$s/bone_config_world.csv"
          # method: JOINT_ANGLE_AND_RIGID_FIT
          method: JOINT_ANGLE
          pose_prior_weight: 10000
          shape_prior_weight: 10
          mesh_file: "%1$s/simplified_skin_yf_zu.obj"
          vertex_weight_file: "%1$s/vertex_weights_simplified_skin.csv"
          mask_file: "%1$s/synthetic_mask_0611.png"
          pose_pca_file: "%1$s/pca_bone_config_world.csv"
          gmm_file: "%1$s/gmm_mixture_5.pbtxt"
          shape_basis_file: "%1$s/pca_bone_shape_config.csv"
        )pb",
        testdata_path_);
    ::google::protobuf::TextFormat::ParseFromString(input_text,
                                                    &input_parameters_);
  }

  std::string testdata_path_;
  PoseOptimizer optimizer_;
  InputParameters input_parameters_;
};

TEST_F(PoseReconstructionTest, SetUpOptimizerTest) {
  SetUpOptimizer(input_parameters_, &optimizer_);

  // Test kinematic chain is constructed.
  const KinematicChain &chain = optimizer_.GetKinematicChain();
  EXPECT_EQ(18, chain.GetJoints().size());
}

TEST_F(PoseReconstructionTest, RunOptimizerTest) {
  // Note: the keypoint names in the target file have to be in agreement with
  // the bone config file in the proto above.
  const std::string target_2d_points_filename =
      testdata_path_ + "labeled_2dkp.csv";
  std::vector<double> trans_params(3, 0);
  std::vector<double> rotate_params(3, 0);
  std::vector<double> alphas(18 * 3, 0);

  SetUpOptimizer(input_parameters_, &optimizer_);

  ASSERT_OK(mouse_pose::LoadTargetPointsFromFile(
      target_2d_points_filename, &(optimizer_.GetTarget2DPoints())));
  std::vector<double> weights(mouse_pose::kNumJointsToOptimize, 1.0);
  // Test that the optimizer runs.
  ASSERT_OK(ReconstructPose(input_parameters_, &optimizer_, weights,
                            &trans_params, &rotate_params, &alphas, nullptr));

  // Test that the optimizer does the right thing (using global translation as a
  // marker).  The value 4.25 is from decomposing the 3x4 projection matrix in
  // the test data directory.
  // Anything within 1.0 (10 cm) is a sign that the optimizer works sensibly.
  EXPECT_NEAR(4.25, trans_params[2], 1.0);
}

TEST_F(PoseReconstructionTest, RunCeresRigidFitOptimizerTest) {
  // Note: the keypoint names in the target file have to be in agreement with
  // the bone config file in the proto above.
  const std::string target_2d_points_filename =
      testdata_path_ + "labeled_2dkp.csv";
  std::vector<double> trans_params(3, 0);
  std::vector<double> rotate_params(3, 0);
  std::vector<double> alphas(18 * 3, 0);

  auto x_constraint =
      input_parameters_.mutable_optimization_constraints()->add_translation();
  x_constraint->set_lower_bound(-10);
  x_constraint->set_upper_bound(10);
  auto y_constraint =
      input_parameters_.mutable_optimization_constraints()->add_translation();
  y_constraint->set_lower_bound(-10);
  y_constraint->set_upper_bound(10);
  auto z_constraint =
      input_parameters_.mutable_optimization_constraints()->add_translation();
  z_constraint->set_lower_bound(0);
  z_constraint->set_upper_bound(5);
  input_parameters_.set_prefer_ceres_rigid_fit_to_open_cv(true);
  SetUpOptimizer(input_parameters_, &optimizer_);

  ASSERT_OK(mouse_pose::LoadTargetPointsFromFile(
      target_2d_points_filename, &(optimizer_.GetTarget2DPoints())));
  std::vector<double> weights(mouse_pose::kNumJointsToOptimize, 1.0);
  // Test that the optimizer runs.
  ASSERT_OK(ReconstructPose(input_parameters_, &optimizer_, weights,
                            &trans_params, &rotate_params, &alphas, nullptr));

  // Test that the optimizer does the right thing (using global translation as a
  // marker).  The value 4.25 is from decomposing the 3x4 projection matrix in
  // the test data directory.
  // Anything within 0.5 (5 cm) is a sign that the optimizer works sensibly.
  // Note that this test will not pass without the translation constraint
  // because the Ceres rigid fit optimizer does not converge as well.
  EXPECT_NEAR(4.25, trans_params[2], 1.0);
}

TEST_F(PoseReconstructionTest, RunOptimizerTestWithGuess) {
  // Note: the keypoint names in the target file have to be in agreement with
  // the bone config file in the proto above.
  const std::string target_2d_points_filename =
      testdata_path_ + "labeled_2dkp.csv";

  std::vector<double> trans_params(3, 0);
  std::vector<double> rotate_params(3, 0);
  std::vector<double> alphas(18 * 3, 0);

  // Set the z component of translation to 3.
  for (int i = 0; i < 6; i++)
    input_parameters_.add_initial_rigid_body_values(0.0);
  input_parameters_.set_initial_rigid_body_values(5, 3.0);

  SetUpOptimizer(input_parameters_, &optimizer_);

  ASSERT_OK(mouse_pose::LoadTargetPointsFromFile(
      target_2d_points_filename, &(optimizer_.GetTarget2DPoints())));
  std::vector<double> weights(mouse_pose::kNumJointsToOptimize, 1.0);
  // Test that the optimizer runs.
  ASSERT_OK(ReconstructPose(input_parameters_, &optimizer_, weights,
                            &trans_params, &rotate_params, &alphas, nullptr));

  // Test that the optimizer does the right thing (using global translation as a
  // marker).  The value 4.25 is from decomposing the 3x4 projection matrix in
  // the test data directory.
  // Anything within 0.5 (5 cm) is a sign that the optimizer works sensibly.
  EXPECT_NEAR(4.25, trans_params[2], 0.5);
}

TEST_F(PoseReconstructionTest, RunOptimizerTestWithTranslationConstraints) {
  // Note: the keypoint names in the target file have to be in agreement with
  // the bone config file in the proto above.
  const std::string target_2d_points_filename =
      testdata_path_ + "labeled_2dkp.csv";

  std::vector<double> trans_params(3, 0);
  std::vector<double> rotate_params(3, 0);
  std::vector<double> alphas(18 * 3, 0);

  auto x_constraint =
      input_parameters_.mutable_optimization_constraints()->add_translation();
  x_constraint->set_lower_bound(-10);
  x_constraint->set_upper_bound(10);
  auto y_constraint =
      input_parameters_.mutable_optimization_constraints()->add_translation();
  y_constraint->set_lower_bound(-10);
  y_constraint->set_upper_bound(10);
  auto z_constraint =
      input_parameters_.mutable_optimization_constraints()->add_translation();
  z_constraint->set_lower_bound(0);
  z_constraint->set_upper_bound(5);

  SetUpOptimizer(input_parameters_, &optimizer_);

  ASSERT_OK(mouse_pose::LoadTargetPointsFromFile(
      target_2d_points_filename, &(optimizer_.GetTarget2DPoints())));
  std::vector<double> weights(mouse_pose::kNumJointsToOptimize, 1.0);
  // Test that the optimizer runs.
  ASSERT_OK(ReconstructPose(input_parameters_, &optimizer_, weights,
                            &trans_params, &rotate_params, &alphas, nullptr));

  // Test that the optimizer does the right thing (using global translation as a
  // marker).  The value 4.25 is from decomposing the 3x4 projection matrix in
  // the test data directory.
  // Anything within 0.5 (5 cm) is a sign that the optimizer works sensibly.
  EXPECT_NEAR(4.25, trans_params[2], 0.5);
}

TEST_F(PoseReconstructionTest, RunOptimizerTestWithRotationConstraints) {
  // Note: the keypoint names in the target file have to be in agreement with
  // the bone config file in the proto above.
  const std::string target_2d_points_filename =
      testdata_path_ + "labeled_2dkp.csv";

  std::vector<double> trans_params(3, 0);
  std::vector<double> rotate_params(3, 0);
  std::vector<double> alphas(18 * 3, 0);

  auto rx_constraint =
      input_parameters_.mutable_optimization_constraints()->add_rotation();
  rx_constraint->set_lower_bound(-1.0);
  rx_constraint->set_upper_bound(1.0);
  auto ry_constraint =
      input_parameters_.mutable_optimization_constraints()->add_rotation();
  ry_constraint->set_lower_bound(-1.0);
  ry_constraint->set_upper_bound(1.0);
  auto rz_constraint =
      input_parameters_.mutable_optimization_constraints()->add_rotation();
  rz_constraint->set_lower_bound(-1.0);
  rz_constraint->set_upper_bound(1.0);

  SetUpOptimizer(input_parameters_, &optimizer_);

  ASSERT_OK(mouse_pose::LoadTargetPointsFromFile(
      target_2d_points_filename, &(optimizer_.GetTarget2DPoints())));
  std::vector<double> weights(mouse_pose::kNumJointsToOptimize, 1.0);
  // Test that the optimizer runs.
  ASSERT_OK(ReconstructPose(input_parameters_, &optimizer_, weights,
                            &trans_params, &rotate_params, &alphas, nullptr));

  // Test that the optimizer does the right thing (using global translation as a
  // marker).  The value 4.25 is from decomposing the 3x4 projection matrix in
  // the test data directory.
  // Anything within 0.5 (5 cm) is a sign that the optimizer works sensibly.
  EXPECT_NEAR(4.25, trans_params[2], 0.5);
}

TEST_F(PoseReconstructionTest, RunOptimizerTestWithRigidBodyConstraints) {
  // Note: the keypoint names in the target file have to be in agreement with
  // the bone config file in the proto above.
  const std::string target_2d_points_filename =
      testdata_path_ + "labeled_2dkp.csv";

  std::vector<double> trans_params(3, 0);
  std::vector<double> rotate_params(3, 0);
  std::vector<double> alphas(18 * 3, 0);

  auto x_constraint =
      input_parameters_.mutable_optimization_constraints()->add_translation();
  x_constraint->set_lower_bound(-10);
  x_constraint->set_upper_bound(10);
  auto y_constraint =
      input_parameters_.mutable_optimization_constraints()->add_translation();
  y_constraint->set_lower_bound(-10);
  y_constraint->set_upper_bound(10);
  auto z_constraint =
      input_parameters_.mutable_optimization_constraints()->add_translation();
  z_constraint->set_lower_bound(0);
  z_constraint->set_upper_bound(5);
  auto rx_constraint =
      input_parameters_.mutable_optimization_constraints()->add_rotation();
  rx_constraint->set_lower_bound(-1.0);
  rx_constraint->set_upper_bound(1.0);
  auto ry_constraint =
      input_parameters_.mutable_optimization_constraints()->add_rotation();
  ry_constraint->set_lower_bound(-1.0);
  ry_constraint->set_upper_bound(1.0);
  auto rz_constraint =
      input_parameters_.mutable_optimization_constraints()->add_rotation();
  rz_constraint->set_lower_bound(-1.0);
  rz_constraint->set_upper_bound(1.0);

  SetUpOptimizer(input_parameters_, &optimizer_);

  ASSERT_OK(mouse_pose::LoadTargetPointsFromFile(
      target_2d_points_filename, &(optimizer_.GetTarget2DPoints())));
  std::vector<double> weights(mouse_pose::kNumJointsToOptimize, 1.0);
  // Test that the optimizer runs.
  ASSERT_OK(ReconstructPose(input_parameters_, &optimizer_, weights,
                            &trans_params, &rotate_params, &alphas, nullptr));

  // Test that the optimizer does the right thing (using global translation as a
  // marker).  The value 4.25 is from decomposing the 3x4 projection matrix in
  // the test data directory.
  // Anything within 0.5 (5 cm) is a sign that the optimizer works sensibly.
  EXPECT_NEAR(4.25, trans_params[2], 0.5);
}

TEST_F(PoseReconstructionTest, RunOptimizerTestWithJointAngleConstraints) {
  // Note: the keypoint names in the target file have to be in agreement with
  // the bone config file in the proto above.
  const std::string target_2d_points_filename =
      testdata_path_ + "labeled_2dkp.csv";

  std::vector<double> trans_params(3, 0);
  std::vector<double> rotate_params(3, 0);
  std::vector<double> alphas(18 * 3, 0);

  for (int i = 0; i < 18 * 3; ++i) {
    auto a_constraint = input_parameters_.mutable_optimization_constraints()
                            ->add_joint_angles();
    a_constraint->set_lower_bound(-1.0);
    a_constraint->set_upper_bound(1.0);
  }

  SetUpOptimizer(input_parameters_, &optimizer_);

  ASSERT_OK(mouse_pose::LoadTargetPointsFromFile(
      target_2d_points_filename, &(optimizer_.GetTarget2DPoints())));
  std::vector<double> weights(mouse_pose::kNumJointsToOptimize, 1.0);
  // Test that the optimizer runs.
  ASSERT_OK(ReconstructPose(input_parameters_, &optimizer_, weights,
                            &trans_params, &rotate_params, &alphas, nullptr));

  // Test that the optimizer does the right thing (using global translation as a
  // marker).  The value 4.25 is from decomposing the 3x4 projection matrix in
  // the test data directory.
  // Anything within 0.5 (5 cm) is a sign that the optimizer works sensibly.
  EXPECT_NEAR(4.25, trans_params[2], 0.5);
}

}  // namespace

}  // namespace mouse_pose
