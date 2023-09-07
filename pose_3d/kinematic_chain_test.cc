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

#include "mouse_pose_analysis/pose_3d/kinematic_chain.h"

#include <cmath>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "googlemock/include/gmock/gmock.h"
#include "gtest/gtest.h"
#include "mouse_pose_analysis/pose_3d/gtest_matchers.h"
#include "mouse_pose_analysis/pose_3d/gtest_util.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/MatrixFunctions"

namespace mouse_pose {
namespace {

using ::testing::Test;

class KinematicChainTest : public Test {
 protected:
  KinematicChainTest() {
    root_joint_idx_ = 0;
    bones_.push_back(Eigen::Vector2i(0, 0));
    bones_.push_back(Eigen::Vector2i(1, 4));
    bones_.push_back(Eigen::Vector2i(2, 4));
    bones_.push_back(Eigen::Vector2i(3, 2));
    bones_.push_back(Eigen::Vector2i(4, 0));
    joints_.push_back(Eigen::Vector3d(0., 0., 0.));
    joints_.push_back(Eigen::Vector3d(-1., 2., 0.));
    joints_.push_back(Eigen::Vector3d(1., 2., 0.));
    joints_.push_back(Eigen::Vector3d(2., 3., 0.));
    joints_.push_back(Eigen::Vector3d(0., 1., 0.));
    joint_names_.push_back("root");
    joint_names_.push_back("joint_1");
    joint_names_.push_back("joint_2");
    joint_names_.push_back("joint_3");
    joint_names_.push_back("joint_4");
    test_config_filename_ =
        GetTestRootDir() + "pose_3d/testdata/test_skeleton.csv";
    test_angle_pca_config_filename_ =
        GetTestRootDir() + "pose_3d/testdata/pca_bone_config_world.csv";
    test_shape_pca_config_filename_ =
        GetTestRootDir() + "pose_3d/testdata/pca_bone_shape_config.csv";
    test_mouse_config_filename_ =
        GetTestRootDir() + "pose_3d/testdata/test_mouse_skeleton.csv";
    chain_ = KinematicChain();
  }
  KinematicChain chain_;
  std::vector<Eigen::Vector2i> bones_;
  std::vector<Eigen::Vector3d> joints_;
  std::vector<std::string> joint_names_;
  int root_joint_idx_;
  std::string test_config_filename_;
  std::string test_mouse_config_filename_;
  std::string test_angle_pca_config_filename_;
  std::string test_shape_pca_config_filename_;
};

TEST_F(KinematicChainTest, ReadConfigFile) {
  EXPECT_OK(chain_.ReadSkeletonConfig(test_config_filename_));
  EXPECT_EQ(chain_.GetBones(), bones_);
  EXPECT_EQ(chain_.GetJoints(), joints_);
  EXPECT_EQ(chain_.GetJointNames(), joint_names_);
}

TEST_F(KinematicChainTest, ReadAnglePcaConfigFile) {
  EXPECT_OK(chain_.ReadAnglePcaBasisConfig(test_angle_pca_config_filename_));
  EXPECT_EQ(14, chain_.GetAlphaBasis().rows());
  EXPECT_EQ(54, chain_.GetAlphaBasis().cols());
  EXPECT_EQ(1, chain_.GetAlphaBasisMean().rows());
  EXPECT_EQ(54, chain_.GetAlphaBasisMean().cols());
  EXPECT_NEAR(-0.290036542817, chain_.GetAlphaBasis()(0, 0), 0.001);
}

TEST_F(KinematicChainTest, ReadShapePcaConfigFile) {
  EXPECT_OK(chain_.ReadShapePcaBasisConfig(test_shape_pca_config_filename_));
  EXPECT_EQ(4, chain_.GetShapeBasis().rows());
  EXPECT_EQ(54, chain_.GetShapeBasis().cols());
  EXPECT_EQ(1, chain_.GetShapeBasisMean().rows());
  EXPECT_EQ(54, chain_.GetShapeBasisMean().cols());
  EXPECT_NEAR(1.22589253e-09, chain_.GetShapeBasis()(0, 0), 0.00001);
  EXPECT_NEAR(3.74043300e-04, chain_.GetShapeBasis()(0, 3), 0.00001);
}

TEST_F(KinematicChainTest, ConstructKinematicChain) {
  EXPECT_OK(chain_.ReadSkeletonConfig(test_config_filename_));
  EXPECT_OK(chain_.ConstructKinematicChain());
  EXPECT_EQ(chain_.GetChainJoint2Alpha().size(), joints_.size());
  EXPECT_EQ(chain_.GetChainAlpha2Joint().size(), 3 * (joints_.size() - 1));
  EXPECT_EQ(chain_.GetChainAlphaAxis().size(), 3 * (joints_.size() - 1));
  EXPECT_EQ(chain_.GetChainAlphaParent().size(), 3 * (joints_.size() - 1));
  EXPECT_EQ(chain_.GetChainJointParent().size(), joints_.size());

  // Checks the 3 rotation angle are mapped to the same joint_idx.
  // Checks the 3 rotation angle are mapped to the correct rotation axis.
  for (int j = 0; j < (joints_.size() - 1); ++j) {
    EXPECT_EQ(chain_.GetChainAlpha2Joint()[j * 3],
              chain_.GetChainAlpha2Joint()[j * 3 + 1]);
    EXPECT_EQ(chain_.GetChainAlpha2Joint()[j * 3 + 1],
              chain_.GetChainAlpha2Joint()[j * 3 + 2]);
    EXPECT_EQ(chain_.GetChainAlphaAxis()[j * 3], Eigen::Vector3d(1., 0., 0.));
    EXPECT_EQ(chain_.GetChainAlphaAxis()[j * 3 + 1],
              Eigen::Vector3d(0., 1., 0.));
    EXPECT_EQ(chain_.GetChainAlphaAxis()[j * 3 + 2],
              Eigen::Vector3d(0., 0., 1.));
  }
  // Checks the x angle is always the parent of the y angle.
  for (int j = 1; j < joints_.size(); ++j) {
    int alpha = chain_.GetChainJoint2Alpha()[j];
    EXPECT_EQ(alpha + 1, chain_.GetChainAlphaParent()[alpha]);
  }
  // Checks the rotation angle, alpha is linked correctly.
  // point_0 - point_4 - point_2 - point_3
  //               \
  //               point_1
  EXPECT_EQ(chain_.GetChainJointParent()[0], 0);
  EXPECT_EQ(chain_.GetChainJointParent()[1], 4);
  EXPECT_EQ(chain_.GetChainJointParent()[2], 4);
  EXPECT_EQ(chain_.GetChainJointParent()[3], 2);
  EXPECT_EQ(chain_.GetChainJointParent()[4], 0);
  int point_1_alpha_z_idx = chain_.GetChainJoint2Alpha()[1] + 2;
  int point_3_alpha_z_idx = chain_.GetChainJoint2Alpha()[3] + 2;
  int point_0_alpha_idx = chain_.GetChainJoint2Alpha()[0];
  int point_4_alpha_idx = chain_.GetChainJoint2Alpha()[4];
  int point_2_alpha_idx = chain_.GetChainJoint2Alpha()[2];
  int point_2_alpha_z_idx = point_2_alpha_idx + 2;
  int point_4_alpha_z_idx = point_4_alpha_idx + 2;
  EXPECT_EQ(chain_.GetChainAlphaParent()[point_4_alpha_z_idx],
            point_0_alpha_idx);
  EXPECT_EQ(chain_.GetChainAlphaParent()[point_1_alpha_z_idx],
            point_4_alpha_idx);
  EXPECT_EQ(chain_.GetChainAlphaParent()[point_2_alpha_z_idx],
            point_4_alpha_idx);
  EXPECT_EQ(chain_.GetChainAlphaParent()[point_3_alpha_z_idx],
            point_2_alpha_idx);
}

TEST_F(KinematicChainTest, UpdateKinematicChain) {
  EXPECT_OK(chain_.ReadSkeletonConfig(test_config_filename_));
  EXPECT_OK(chain_.ConstructKinematicChain());
  std::vector<double> alphas(3 * joints_.size(), 0.);
  std::vector<Eigen::Vector4d> updated_joints;
  std::vector<Eigen::Matrix4d> chain_G;

  // Check when the rotation angle is zero, the joints should not move.
  chain_.UpdateKinematicChain(alphas, &updated_joints, &chain_G);
  for (int joint_idx = 0; joint_idx < updated_joints.size(); ++joint_idx) {
    EXPECT_THAT(joints_[joint_idx], mouse_pose::test::EigenMatrixEq(
                                        updated_joints[joint_idx].head<3>()));
  }
  // Rotate point_3 around x axis 90 degree, other joints should not move.
  alphas[9] = 0.5 * M_PI;
  chain_.UpdateKinematicChain(alphas, &updated_joints, &chain_G);
  for (int joint_idx = 0; joint_idx < updated_joints.size(); ++joint_idx) {
    if (joint_idx == 3) {
      EXPECT_THAT(Eigen::Vector3d(2., 2., 1.),
                  mouse_pose::test::EigenMatrixNear(
                      updated_joints[joint_idx].head<3>(), 1e-10));

    } else {
      EXPECT_THAT(joints_[joint_idx], mouse_pose::test::EigenMatrixEq(
                                          updated_joints[joint_idx].head<3>()));
    }
  }

  alphas[10] = 0.5 * M_PI;
  chain_.UpdateKinematicChain(alphas, &updated_joints, &chain_G);
  for (int joint_idx = 0; joint_idx < updated_joints.size(); ++joint_idx) {
    if (joint_idx == 3) {
      EXPECT_THAT(Eigen::Vector3d(2., 2., -1.),
                  mouse_pose::test::EigenMatrixNear(
                      updated_joints[joint_idx].head<3>(), 1e-10));
    } else {
      EXPECT_THAT(joints_[joint_idx], mouse_pose::test::EigenMatrixEq(
                                          updated_joints[joint_idx].head<3>()));
    }
  }
  alphas[chain_.GetChainJoint2Alpha()[1]] = 0.5 * M_PI;
  alphas[9] = alphas[10] = 0.;
  chain_.UpdateKinematicChain(alphas, &updated_joints, &chain_G);
  for (int joint_idx = 0; joint_idx < updated_joints.size(); ++joint_idx) {
    if (joint_idx == 1) {
      EXPECT_THAT(Eigen::Vector3d(-1., 1., 1.),
                  mouse_pose::test::EigenMatrixNear(
                      updated_joints[joint_idx].head<3>(), 1e-10));
    } else {
      EXPECT_THAT(joints_[joint_idx], mouse_pose::test::EigenMatrixEq(
                                          updated_joints[joint_idx].head<3>()));
    }
  }

  // Check the transformation matrix for root should be identity.
  EXPECT_THAT(Eigen::Matrix4d::Identity(),
              mouse_pose::test::EigenMatrixEq(chain_G[0]));
}

TEST_F(KinematicChainTest, UpdateJointXYZLocations) {
  // Tests some inverse projections on mouse skeleton using saved shape basis.
  EXPECT_OK(chain_.ReadShapePcaBasisConfig(test_shape_pca_config_filename_));
  EXPECT_OK(chain_.ReadSkeletonConfig(test_mouse_config_filename_));
  int num_coeff = chain_.GetShapeBasis().rows();
  int num_joints = chain_.GetJoints().size();
  std::vector<double> coords_one(num_coeff, 1.0);  // all ones.
  std::vector<Eigen::Vector3d> updated_joints;
  EXPECT_OK(chain_.UpdateJointXYZLocations(coords_one.data(), &updated_joints));
  // root coordinates do not change with the basis.
  Eigen::Vector3d root_pos(0.18354701, 0.34671989, 0.10455703);
  EXPECT_THAT(root_pos,
              mouse_pose::test::EigenMatrixNear(updated_joints[0], 1e-6));
  EXPECT_THAT(Eigen::Vector3d(0.39348173, 0.62551619, 0.07243943),  // LeftLeg
              mouse_pose::test::EigenMatrixNear(updated_joints[4], 1e-6));

  std::vector<double> coords_zero(num_coeff, 0.0);  // all zeros ==> mean.
  EXPECT_OK(
      chain_.UpdateJointXYZLocations(coords_zero.data(), &updated_joints));
  EXPECT_THAT(root_pos,
              mouse_pose::test::EigenMatrixNear(updated_joints[0], 1e-6));
  Eigen::RowVectorXd mean = chain_.GetShapeBasisMean();
  Eigen::Map<Eigen::MatrixXd> mean_t(mean.data(), 3, num_joints);
  for (int i = 0; i < num_joints; ++i) {
    EXPECT_THAT(mean_t.col(i),
                mouse_pose::test::EigenMatrixNear(updated_joints[i], 1e-6));
  }
}
}  // namespace
}  // namespace mouse_pose
