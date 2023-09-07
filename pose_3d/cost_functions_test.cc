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

#include "mouse_pose_analysis/pose_3d/cost_functions.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "googlemock/include/gmock/gmock.h"
#include "gtest/gtest.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "mouse_pose_analysis/pose_3d/gmm_prior.h"
#include "mouse_pose_analysis/pose_3d/gtest_matchers.h"
#include "mouse_pose_analysis/pose_3d/gtest_util.h"
#include "mouse_pose_analysis/pose_3d/kinematic_chain.h"
#include "mouse_pose_analysis/pose_3d/pose_optimizer_utils.h"
#include "mouse_pose_analysis/pose_3d/rigged_mesh_utils.h"
#include "third_party/eigen3/Eigen/Core"

namespace mouse_pose {
namespace {

using ::testing::Test;

TEST(CostFunctionsTest, RigidTransCostFunctionTest) {
  std::vector<Eigen::Vector3d> source_points(5, Eigen::Vector3d::Zero());
  std::vector<Eigen::Vector2d> target_points(5, Eigen::Vector2d::Ones());
  Eigen::Matrix<double, 3, 4> projection_mat(
      Eigen::Matrix<double, 3, 4>::Identity());
  projection_mat(2, 3) = 1.;
  std::vector<double> weights(5, 1.);
  RigidTransCostFunction cost_function(source_points, target_points,
                                       projection_mat, weights);
  std::vector<double> rotation(3, 0.);
  std::vector<double> translation(3, 0.);
  std::vector<double> residual(10, 0.);
  EXPECT_TRUE(
      cost_function(rotation.data(), translation.data(), residual.data()));
  std::vector<double> expected(10, -1);
  EXPECT_THAT(residual, ::testing::Pointwise(::testing::DoubleEq(), expected));
}

TEST(CostFunctionsTest, JointAngleCostFunctionTest) {
  KinematicChain chain(GetTestRootDir() + "pose_3d/testdata/test_skeleton.csv");
  EXPECT_OK(chain.ConstructKinematicChain());
  std::vector<double> alphas(3 * chain.GetJoints().size(), 0.);
  std::vector<Eigen::Vector4d> updated_joints;
  std::vector<Eigen::Matrix4d> chain_G;
  // Check when the rotation angle is zero, the joints should not move.
  chain.UpdateKinematicChain(alphas, &updated_joints, &chain_G);
  for (int joint_idx = 0; joint_idx < updated_joints.size(); ++joint_idx) {
    EXPECT_THAT(
        chain.GetJoints()[joint_idx],
        mouse_pose::test::EigenMatrixEq(updated_joints[joint_idx].head<3>()));
  }
}

TEST(CostFunctionsTest, JointAngleAndRigidCostFunctionTest) {
  KinematicChain chain(GetTestRootDir() + "pose_3d/testdata/test_skeleton.csv");
  EXPECT_OK(chain.ConstructKinematicChain());

  std::vector<Eigen::Vector2d> target_points(18, Eigen::Vector2d::Ones());
  Eigen::Matrix<double, 3, 4> projection_mat(
      Eigen::Matrix<double, 3, 4>::Identity());
  projection_mat(2, 3) = 1.;
  std::vector<double> weights(6, 1.);
  JointAngleAndRigidCostFunction cost_function(chain, target_points,
                                               projection_mat, weights);
  std::vector<double> rotation(3, 0.);
  std::vector<double> translation(3, 0.);
  std::vector<double> loadings(14, 0.);
  std::vector<double> residual(18 * 2, 0.);
  EXPECT_TRUE(cost_function(rotation.data(), translation.data(),
                            loadings.data(), residual.data()));
  // Hard to know what the expected value should be, so just test that it runs
  // and changes the residual value.
  EXPECT_NE(residual[0], 0);
}

TEST(CostFunctionsTest, RigidTrans3DCostFunctionTest) {
  std::vector<Eigen::Vector3d> source_points(5, Eigen::Vector3d::Zero());
  std::vector<Eigen::Vector3d> target_points(5, Eigen::Vector3d::Ones());
  std::vector<double> weights(5, 1.0);
  RigidTrans3DCostFunction cost_function(source_points, target_points, weights);
  std::vector<double> rotation(3, 0.);
  std::vector<double> translation(3, 0.);
  std::vector<double> residual(15, 0.);
  EXPECT_TRUE(
      cost_function(rotation.data(), translation.data(), residual.data()));
  std::vector<double> expected(15, -1);
  EXPECT_THAT(residual, ::testing::Pointwise(::testing::DoubleEq(), expected));
}

TEST(CostFunctionsTest, Pose3DToAngleCostFunctionTest) {
  KinematicChain chain(GetTestRootDir() + "pose_3d/testdata/test_skeleton.csv");
  EXPECT_OK(chain.ConstructKinematicChain());
  std::vector<Eigen::Vector3d> target_points = chain.GetJoints();
  std::vector<double> weights;
  for (int i = 0; i < target_points.size(); ++i) {
    target_points[i] = target_points[i] + Eigen::Vector3d::Ones();
    weights.push_back(1.0);
  }
  Pose3DToAngleCostFunction cost_function(chain, target_points, weights);
  std::vector<double> rotation(3, 0.);
  std::vector<double> translation(3, 0.);
  std::vector<double> alphas(5 * 3, 0.);
  std::vector<double> residual(5 * 3, 1.);
  EXPECT_TRUE(cost_function(rotation.data(), translation.data(), alphas.data(),
                            residual.data()));
  std::vector<double> expected(5 * 3, 1);
  EXPECT_THAT(residual, ::testing::Pointwise(::testing::DoubleEq(), expected));

  std::vector<double> perfect_translation(3, 1.);
  EXPECT_TRUE(cost_function(rotation.data(), perfect_translation.data(),
                            alphas.data(), residual.data()));
  std::vector<double> perfect_expected(5 * 3, 0);
  EXPECT_THAT(residual,
              ::testing::Pointwise(::testing::DoubleEq(), perfect_expected));
}

TEST(CostFunctionsTest, JointAnglePosePriorAndRigidCostFunctionTest) {
  KinematicChain chain(GetTestRootDir() +
                       "pose_3d/testdata/test_mouse_skeleton.csv");
  EXPECT_OK(chain.ConstructKinematicChain());
  std::string filepath =
      GetTestRootDir() + "pose_3d/testdata/gmm_mixture_5.pbtxt";
  GaussianMixtureModel gmm;
  gmm.InitializeFromFile(filepath);

  std::vector<Eigen::Vector2d> target_points(18, Eigen::Vector2d::Ones());
  Eigen::Matrix<double, 3, 4> projection_mat(
      Eigen::Matrix<double, 3, 4>::Identity());
  projection_mat(2, 3) = 1.;
  std::vector<double> weights(target_points.size(), 1.);
  JointAnglePosePriorAndRigidCostFunction cost_function(
      chain, target_points, projection_mat, weights, gmm, 1.0);
  std::vector<double> rotation(3, 0.);
  std::vector<double> translation(3, 0.);
  std::vector<double> loadings(14, 0.);
  std::vector<double> residual(18 * 2, 0.);
  EXPECT_TRUE(cost_function(rotation.data(), translation.data(),
                            loadings.data(), residual.data()));
  // Hard to know what the expected value should be, so just test that it runs
  // and changes the residual value.
  EXPECT_NE(residual[0], 0);
}

TEST(CostFunctionsTest, JointAnglePcaCostFunctionTest) {
  KinematicChain chain(GetTestRootDir() + "pose_3d/testdata/test_skeleton.csv");
  EXPECT_OK(chain.ConstructKinematicChain());
  EXPECT_OK(chain.ReadAnglePcaBasisConfig(
      GetTestRootDir() + "pose_3d/testdata/pca_bone_config_world.csv"));

  std::vector<Eigen::Vector2d> target_points(18, Eigen::Vector2d::Ones());
  Eigen::Matrix<double, 3, 4> projection_mat(
      Eigen::Matrix<double, 3, 4>::Identity());
  projection_mat(2, 3) = 1.;
  std::vector<double> weights(target_points.size(), 1.);
  JointAnglePcaCostFunction cost_function(chain, target_points, projection_mat,
                                          weights);
  std::vector<double> rotation(3, 0.);
  std::vector<double> translation(3, 0.);
  std::vector<double> loadings(14, 0.);
  std::vector<double> residual(18 * 2, 0.);
  EXPECT_TRUE(cost_function(rotation.data(), translation.data(),
                            loadings.data(), residual.data()));
  // Hard to know what the expected value should be, so just test that it runs
  // and changes the residual value.
  EXPECT_NE(residual[0], 0);
}

}  // namespace
}  // namespace mouse_pose
