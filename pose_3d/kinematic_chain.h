/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_KINEMATIC_CHAIN_H_
#define MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_KINEMATIC_CHAIN_H_

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/status/status.h"
#include "mouse_pose_analysis/pose_3d/matrix_util.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/SparseCore"

// Implementation of paper Tracking People with Twists and Exponential Maps
// (https://ieeexplore.ieee.org/abstract/document/698581/).
// The class contains members:
// joints_ (XYZ coordinates of the joints)
// bones_ (joint_idx for each bone)
// angles_ (three joint angles for each joint)
// chain_* (mapping between properties of the chain dynamics computation).
//
// joints_ and bones_ are read from the CSV skeleton configuration file.
// chains_* are constructed based on the CSV skeleton configuration file, once
// the kinematic chain is constructed these do not change.
// alphas_ are from optimizer/other sources.

namespace mouse_pose {

inline constexpr int kNumShapeBasisComponents = 4;

class KinematicChain {
 public:
  KinematicChain() {}
  explicit KinematicChain(std::string config_filename);
  Eigen::Matrix4d ConstructTwist(const Eigen::Vector3d joint,
                                 const Eigen::Vector3d axis);

  // Constructs the parent-child mapping.
  void ConstructSkeleton(std::vector<int>* joint_parent);
  void ConstructSkeletonRecursive(int root_joint_idx,
                                  std::vector<int>* joint_parent);

  // Constructs the kinematic chain represented using twist.
  absl::Status ConstructKinematicChain();

  // Reads the skeleton configuration file to fill the joints_ and bones_.
  absl::Status ReadSkeletonConfig(std::string config_filename);

  // Reads the PCA basis file to fill alpha_basis_ and alpha_basis_mean_.
  inline absl::Status ReadAnglePcaBasisConfig(std::string pca_filename) {
    return ReadPcaBasisConfig(pca_filename, &alpha_basis_, &alpha_basis_mean_);
  }

  // Reads the PCA basis file to fill shape_basis_ and shape_basis_mean_.
  // Note - we assume the shape basis is pre-scaled by sqrt(\lambda), where
  // \lambda is the eigenvalues vector of the covariance matrix.
  inline absl::Status ReadShapePcaBasisConfig(std::string pca_filename) {
    return ReadPcaBasisConfig(pca_filename, &shape_basis_, &shape_basis_mean_);
  }

  // Computes new set of joint x, y and z locations in the rest pose from the
  // shape prior and stores them in joints_. The new locations are controlled by
  // the subspace coordinates here. subspace_coords can be driven by a Ceres
  // optimizer.
  template <typename T>
  absl::Status UpdateJointXYZLocations(
      const T* subspace_coords,
      std::vector<Eigen::Matrix<T, 3, 1>>* updated_joints) const {
    typedef Eigen::Matrix<T, 3, 1> Vector3;
    typedef Eigen::Matrix<T, 1, Eigen::Dynamic> RowVectorX;
    Eigen::Map<const Eigen::Matrix<T, 1, kNumShapeBasisComponents>>
        coefficients(subspace_coords);

    CHECK_EQ(kNumShapeBasisComponents, shape_basis_.rows());
    CHECK_EQ(shape_basis_.cols(), 3 * joints_.size());
    RowVectorX joint_coords = coefficients * shape_basis_.template cast<T>() +
                              shape_basis_mean_.template cast<T>();

    updated_joints->clear();
    for (int j = 0; j < joints_.size(); ++j) {
      Vector3 joint(joint_coords(3 * j + 0), joint_coords(3 * j + 1),
                    joint_coords(3 * j + 2));
      updated_joints->push_back(joint);
    }

    return absl::OkStatus();
  }

  // Computes the 3D location and the transformation matrix of the points
  // on the skeleton based on the
  // input: the constructed kinematic chain, the joint angles (the alphas).
  template <typename T>
  void UpdateKinematicChain(
      const std::vector<T>& alphas,
      std::vector<Eigen::Matrix<T, 4, 1>>* updated_joints,
      std::vector<Eigen::Matrix<T, 4, 4>>* chain_trans_mats,
      std::vector<Eigen::Matrix<T, 3, 1>>* joints = nullptr) const {
    typedef Eigen::Matrix<T, 4, 1> Vector4;
    typedef Eigen::Matrix<T, 3, 1> Vector3;
    typedef Eigen::Matrix<T, 4, 4> Matrix4;

    auto new_joints = std::make_unique<std::vector<Vector3>>();
    if (joints == nullptr) {
      for (int j = 0; j < joints_.size(); ++j) {
        Vector3 jt = joints_[j].template cast<T>();
        new_joints->push_back(jt);
      }
      joints = new_joints.get();
    }

    // Set the default values to these transformation matrix lists.
    // Use the updated joints (chain alpha centers).
    std::vector<std::vector<Matrix4>> chain_Gr(
        joints->size(),
        std::vector<Matrix4>(alphas.size(), Matrix4::Identity()));
    std::vector<std::vector<Matrix4>> chain_Gl(
        joints->size(),
        std::vector<Matrix4>(alphas.size(), Matrix4::Identity()));
    Matrix4 tmp_G = Matrix4::Identity();
    for (int joint_idx = 0; joint_idx < joints->size(); ++joint_idx) {
      int alpha_idx = chain_joint2alpha_[joint_idx];
      tmp_G.setIdentity();
      while (alpha_idx != -1) {
        Vector3 chain_alpha_axis =
            chain_alpha_axis_[alpha_idx].template cast<T>();
        Matrix4 twist = SkewAxis(
            chain_alpha_axis, joints->at(chain_alpha_center_index_[alpha_idx]));

        // The following code is to implement
        // tmp_G = ExponentialMap(twist, alphas[alpha_idx]) * tmp_G;
        T s_value = alphas[alpha_idx];
        T c_value = static_cast<T>(1);
        T factor = static_cast<T>(6);
        for (int i = 1; i < kOrderOfApproximation; i++) {
          if (i == 1) {
            s_value += pow(static_cast<T>(-1), i) *
                       pow(static_cast<T>(alphas[alpha_idx]), 2 * i + 1) /
                       (static_cast<T>(2 * 3));
            c_value += pow(static_cast<T>(-1), i) *
                       pow(static_cast<T>(alphas[alpha_idx]), 2 * i) /
                       (static_cast<T>(2));
          } else {
            factor *= static_cast<T>(2 * i);
            c_value += pow(static_cast<T>(-1), i) *
                       pow(static_cast<T>(alphas[alpha_idx]), 2 * i) / factor;
            factor *= static_cast<T>(2 * i + 1);
            s_value += pow(static_cast<T>(-1), i) *
                       pow(static_cast<T>(alphas[alpha_idx]), 2 * i + 1) /
                       factor;
          }
        }
        Eigen::Matrix<T, 4, 4> tmp_twist = twist * twist;
        twist = (s_value)*twist + (static_cast<T>(1) - c_value) * tmp_twist +
                Eigen::Matrix<T, 4, 4>::Identity();
        tmp_G = twist * tmp_G;
        chain_Gr[joint_idx][alpha_idx] = tmp_G;
        alpha_idx = chain_alpha_parent_[alpha_idx];
      }

      // Set Gl as default values.
      for (int i = 0; i < alphas.size(); ++i) {
        chain_Gl[joint_idx][i] = tmp_G;
      }

      alpha_idx = chain_joint2alpha_[joint_idx];
      while (alpha_idx != -1) {
        chain_Gl[joint_idx][alpha_idx] =
            tmp_G * chain_Gr[joint_idx][alpha_idx].inverse();
        alpha_idx = chain_alpha_parent_[alpha_idx];
      }
    }
    updated_joints->clear();
    if (chain_trans_mats != nullptr) {
      chain_trans_mats->clear();
    }
    for (int joint_idx = 0; joint_idx < joints->size(); ++joint_idx) {
      Vector4 joint(Vector4::Ones());
      joint.template head<3>() = joints->at(joint_idx).template cast<T>();
      Matrix4 rotate_mat = chain_Gl[joint_idx][0] * chain_Gr[joint_idx][0];
      updated_joints->push_back(rotate_mat * joint);
      if (chain_trans_mats != nullptr) {
        chain_trans_mats->push_back(rotate_mat);
      }
    }
  }

  // Accessors.
  // Returns the bones, a list of <joint_1_idx, joint_2_idx>.
  const std::vector<Eigen::Vector2i>& GetBones() const { return bones_; }

  // Returns the 3D position of the joints, a list of <x, y, z>.
  const std::vector<Eigen::Vector3d>& GetJoints() const { return joints_; }

  // Returns the names of the joints.
  const std::vector<std::string>& GetJointNames() const { return joint_names_; }

  // Returns the mapping between the joint idx and the x-axis joint angle idx.
  const std::vector<int>& GetChainJoint2Alpha() const {
    return chain_joint2alpha_;
  }

  // Returns the mapping between the joint angle idx and the joint idx.
  const std::vector<int>& GetChainAlpha2Joint() const {
    return chain_alpha2joint_;
  }

  // Returns the mapping between the joint angle idx and the rotation axis.
  const std::vector<Eigen::Vector3d>& GetChainAlphaAxis() const {
    return chain_alpha_axis_;
  }

  // Returns the mapping between the joint angle and its parent joint angle.
  const std::vector<int>& GetChainAlphaParent() const {
    return chain_alpha_parent_;
  }

  const std::vector<int>& GetChainJointParent() const {
    return chain_joint_parent_;
  }

  const Eigen::MatrixXd& GetAlphaBasis() const { return alpha_basis_; }

  const Eigen::RowVectorXd& GetAlphaBasisMean() const {
    return alpha_basis_mean_;
  }

  const Eigen::MatrixXd& GetShapeBasis() const { return shape_basis_; }

  const Eigen::RowVectorXd& GetShapeBasisMean() const {
    return shape_basis_mean_;
  }

 private:
  // The root joint idx.
  int root_joint_idx_;

  // The 3D position of the joints.
  std::vector<Eigen::Vector3d> joints_;

  // The bones in the skeleton, represented as <joint_idx, parent_joint_idx>.
  std::vector<Eigen::Vector2i> bones_;

  // The joint names.
  std::vector<std::string> joint_names_;

  // The mapping between the joint_idx to rotation angle alpha_idx.
  // Each joint has three rotation angles.
  std::vector<int> chain_joint2alpha_;

  // The mapping between the rotation angle alpha_idx to joint_idx.
  std::vector<int> chain_alpha2joint_;

  // The mapping between rotation angle and the axis (x, y, z).
  std::vector<Eigen::Vector3d> chain_alpha_axis_;

  // The mapping between rotation angle_idx and the index to the center (x, y,
  // z).
  std::vector<int> chain_alpha_center_index_;

  // The chain of rotation angles from child to parent.
  std::vector<int> chain_alpha_parent_;

  // The chain of the skeleton.
  std::vector<int> chain_joint_parent_;

  // The basis space for the angles.
  Eigen::MatrixXd alpha_basis_;
  Eigen::RowVectorXd alpha_basis_mean_;

  // The basis space for the shape prior.
  Eigen::MatrixXd shape_basis_;
  Eigen::RowVectorXd shape_basis_mean_;

  // Helper to load PCA basis CSV files.
  // Read the PCA basis file to fill alpha_basis_ and alpha_basis_mean_.
  absl::Status ReadPcaBasisConfig(std::string pca_filename,
                                  Eigen::MatrixXd* basis,
                                  Eigen::RowVectorXd* basis_mean);
};
}  // namespace mouse_pose

#endif  // MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_KINEMATIC_CHAIN_H_
