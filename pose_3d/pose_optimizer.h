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

#ifndef MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_POSE_OPTIMIZER_H_
#define MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_POSE_OPTIMIZER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "ceres/ceres.h"
#include "ceres/solver.h"
#include "ceres/types.h"
#include "mouse_pose_analysis/pose_3d/cost_functions.h"
#include "mouse_pose_analysis/pose_3d/gmm_prior.h"
#include "mouse_pose_analysis/pose_3d/keypoint.pb.h"
#include "mouse_pose_analysis/pose_3d/kinematic_chain.h"
#include "mouse_pose_analysis/pose_3d/pose_optimizer_utils.h"
#include "mouse_pose_analysis/pose_3d/rigged_mesh.h"
#include "opencv2/core/core.hpp"
#include "third_party/eigen3/Eigen/Core"

namespace mouse_pose {

inline constexpr int kNumJointAnglesToOptimize = 18 * 3;
inline constexpr int kNumAnglePcaCoefficientsToOptimize = 14;
inline constexpr int kNumJointsToOptimize = 18;

class PoseOptimizer {
 public:
  PoseOptimizer() {}

  // Computes the rigid transformation trans_G based on the scaling landmarks
  // , the camera projection matrix, the weights on the target keypoints and
  // constraints.
  template <int number_of_joints>
  absl::Status RigidFit(
      std::vector<double>* trans_params, std::vector<double>* rotate_params,
      const std::vector<double>& weights,
      const ceres::Solver::Options options = ceres::Solver::Options(),
      const absl::flat_hash_map<int, std::pair<double, double>>*
          trans_constraints = nullptr,
      const absl::flat_hash_map<int, std::pair<double, double>>*
          rotate_constraints = nullptr) {
    auto problem = std::make_unique<ceres::Problem>();
    constexpr int residual_dim = number_of_joints * 2;
    std::unique_ptr<ceres::CostFunction> cost_function =
        std::make_unique<ceres::AutoDiffCostFunction<RigidTransCostFunction,
                                                     residual_dim, 3, 3>>(
            new RigidTransCostFunction(chain_.GetJoints(), target_2d_points_,
                                       projection_mat_, weights));
    problem->AddResidualBlock(cost_function.release(), nullptr,
                              rotate_params->data(), trans_params->data());
    if (trans_constraints != nullptr) {
      for (auto const& trans_constraint : *trans_constraints) {
        problem->SetParameterLowerBound(trans_params->data(),
                                        trans_constraint.first,
                                        trans_constraint.second.first);
        problem->SetParameterUpperBound(trans_params->data(),
                                        trans_constraint.first,
                                        trans_constraint.second.second);
      }
    }
    if (rotate_constraints != nullptr) {
      for (auto const& rotate_constraint : *rotate_constraints) {
        problem->SetParameterLowerBound(rotate_params->data(),
                                        rotate_constraint.first,
                                        rotate_constraint.second.first);
        problem->SetParameterUpperBound(rotate_params->data(),
                                        rotate_constraint.first,
                                        rotate_constraint.second.second);
      }
    }

    ceres::Solver::Summary summary;
    Solve(options, problem.get(), &summary);
    return absl::OkStatus();
  }

  // Computes the translation and rotation from 3D joints and corresponding 2D
  // key points.
  void ComputeRigidBodyTransform(const std::vector<double>& joint_weights,
                                 std::vector<double>* translation,
                                 std::vector<double>* rotation,
                                 bool use_init_guess = false);

  // Optimizes for the rotation angles for each joints based on the input:
  // camera projection matrix cam_P_, rigid transformation trans_G_, the
  // kinematic chain and the weights on the target keypoints.
  template <int number_of_joints>
  ceres::Solver::Summary JointAngleFit(
      const Eigen::Vector3d& rotation, const Eigen::Vector3d& translation,
      std::vector<double>* joint_angles, const std::vector<double>& weights,
      const ceres::Solver::Options& options = ceres::Solver::Options(),
      const absl::flat_hash_map<int, std::pair<double, double>>* constraints =
          nullptr) {
    auto problem = std::make_unique<ceres::Problem>();
    constexpr int residual_dim = number_of_joints * 2;
    constexpr int opt_var_dim = number_of_joints * 3;
    std::unique_ptr<ceres::CostFunction> cost_function(
        new ceres::AutoDiffCostFunction<JointAngleCostFunction, residual_dim,
                                        opt_var_dim>(new JointAngleCostFunction(
            chain_, target_2d_points_, projection_mat_, rotation, translation,
            weights)));
    problem->AddResidualBlock(cost_function.release(), nullptr,
                              joint_angles->data());
    if (constraints != nullptr) {
      for (auto const& constraint : *constraints) {
        problem->SetParameterLowerBound(joint_angles->data(), constraint.first,
                                        constraint.second.first);
        problem->SetParameterUpperBound(joint_angles->data(), constraint.first,
                                        constraint.second.second);
      }
    }
    ceres::Solver::Summary summary;
    Solve(options, problem.get(), &summary);
    return summary;
  }

  // Simultaneously solve for the joint angles, rotation, and translation.
  // Joint angles are subject to a GMM pose prior.
  ceres::Solver::Summary JointAnglePosePriorAndRigidFit(
      std::vector<double>* rotate_params, std::vector<double>* trans_params,
      std::vector<double>* joint_angles, const std::vector<double>& weights,
      const double pose_prior_weight,
      const ceres::Solver::Options& options = ceres::Solver::Options(),
      const absl::flat_hash_map<int, std::pair<double, double>>*
          trans_constraints = nullptr,
      const absl::flat_hash_map<int, std::pair<double, double>>*
          rotate_constraints = nullptr,
      const absl::flat_hash_map<int, std::pair<double, double>>*
          alpha_constraints = nullptr);

  // Simultaneously solve for the joint angles, shape, rotation, and translation
  // Joint angles are subject to a GMM pose prior.
  ceres::Solver::Summary JointAnglePosePriorShapeBasisAndRigidFit(
      std::vector<double>* rotate_params, std::vector<double>* trans_params,
      std::vector<double>* joint_angles, std::vector<double>* shape_weights,
      const std::vector<double>& weights, const double pose_prior_weight,
      const double shape_basis_l2_weight,
      const ceres::Solver::Options& options = ceres::Solver::Options(),
      const absl::flat_hash_map<int, std::pair<double, double>>*
          trans_constraints = nullptr,
      const absl::flat_hash_map<int, std::pair<double, double>>*
          rotate_constraints = nullptr,
      const absl::flat_hash_map<int, std::pair<double, double>>*
          alpha_constraints = nullptr);

  // Simultaneously solve for the joint angles, rotation, and translation.
  ceres::Solver::Summary JointAngleAndRigidFit(
      std::vector<double>* rotate_params, std::vector<double>* trans_params,
      std::vector<double>* joint_angles, const std::vector<double>& weights,
      const ceres::Solver::Options& options = ceres::Solver::Options(),
      const absl::flat_hash_map<int, std::pair<double, double>>*
          trans_constraints = nullptr,
      const absl::flat_hash_map<int, std::pair<double, double>>*
          rotate_constraints = nullptr,
      const absl::flat_hash_map<int, std::pair<double, double>>*
          alpha_constraints = nullptr);

  // Combines the JointAngleFit and the ImageSilhouetteFit to optimize the
  // joint angles based on the target 2D keypoints and shape silhouette.
  ceres::Solver::Summary PoseShapeFit(
      const Eigen::Vector3d& rotation, const Eigen::Vector3d& translation,
      const cv::Mat& mask, std::vector<double>* joint_angles,
      const std::vector<double>& weights,
      const ceres::Solver::Options& options = ceres::Solver::Options());

  // Optimizes for the rotation angles for each joints in a basis space from the
  // input: camera projection matrix cam_P_, rigid transformation trans_G_ and
  // the kinematic chain (after calling ReadAnglePcaBasisConfig).
  ceres::Solver::Summary JointAnglePcaFit(
      std::vector<double>* rotation, std::vector<double>* translation,
      std::vector<double>* joint_angles, const std::vector<double>& weights,
      const ceres::Solver::Options& options = ceres::Solver::Options());

  // Computes the rigid transformation trans_G based on the ground truth 3D
  // pose.
  absl::Status RigidFitTo3D(
      const std::vector<double>& weight, std::vector<double>* trans_params,
      std::vector<double>* rotate_params,
      const ceres::Solver::Options& options = ceres::Solver::Options());

  // Optimizes for the rotation angles for each joints based on the input:
  // rigid transformation trans_G_ and the kinematic chain.
  ceres::Solver::Summary JointAngleFitTo3D(
      const std::vector<double>& weights, std::vector<double>* trans_params,
      std::vector<double>* rotate_params, std::vector<double>* joint_angles,
      const ceres::Solver::Options& options = ceres::Solver::Options());

  // Loads a mesh from an OBJ file and its rigging a CSV file.
  absl::Status LoadRiggedMeshFromFiles(std::string obj_filename,
                                       std::string weight_filename);

  absl::Status LoadGmmFromFile(std::string gmm_filename);

  // Loads calico mouse 2D target keypoints and creates keypoint weights from a
  // unordered map of joint names and x,y coordinates.
  absl::Status LoadMouseTargetPointsFromMap(
      const absl::flat_hash_map<std::string, std::pair<float, float>>&
          target_keypoints,
      const absl::flat_hash_map<std::string, float>& input_keypoint_weights,
      std::vector<double>* weights = nullptr);

  // Loads the target 2D points from a BodyJoints proto.
  // Outputs a vector of weights: 1 for key point present in the kinematic
  // chain, 0 otherwise.
  absl::Status LoadTargetPointsFromBodyJoints(
      const mouse_pose::optical_mouse::BodyJoints& joints,
      std::vector<double>* weights = nullptr);

  // Accessors
  // Returns the kinematic chain.
  // There's a way to remove the redundancy of const and non-const accessors.
  // See Effective C++ Item 3.  But the logic is simple here, we'll just type.
  KinematicChain& GetKinematicChain() { return chain_; }
  const KinematicChain& GetKinematicChain() const { return chain_; }

  Eigen::Matrix<double, 3, 4>& GetProjectionMat() { return projection_mat_; }
  const Eigen::Matrix<double, 3, 4>& GetProjectionMat() const {
    return projection_mat_;
  }

  std::vector<Eigen::Vector2d>& GetTarget2DPoints() {
    return target_2d_points_;
  }
  const std::vector<Eigen::Vector2d>& GetTarget2DPoints() const {
    return target_2d_points_;
  }

  std::vector<Eigen::Vector3d>& GetTarget3DPoints() {
    return target_3d_points_;
  }
  const std::vector<Eigen::Vector3d>& GetTarget3DPoints() const {
    return target_3d_points_;
  }

 private:
  KinematicChain chain_;
  std::unique_ptr<RiggedMesh> mesh_;
  std::vector<Eigen::Vector2d> target_2d_points_;
  std::vector<Eigen::Vector3d> target_3d_points_;
  Eigen::Matrix<double, 3, 4> projection_mat_;
  std::unique_ptr<GaussianMixtureModel> gmm_;
};

}  // namespace mouse_pose

#endif  // MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_POSE_OPTIMIZER_H_
