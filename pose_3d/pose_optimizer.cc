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

#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "ceres/ceres.h"
#include "ceres/cost_function.h"
#include "ceres/dynamic_autodiff_cost_function.h"
#include "ceres/problem.h"
#include "ceres/solver.h"
#include "ceres/types.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/imgproc.hpp"
#include "mouse_pose_analysis/pose_3d/pose_optimizer_utils.h"
#include "mouse_pose_analysis/pose_3d/rigged_mesh_utils.h"
#include "third_party/eigen3/Eigen/Core"

namespace mouse_pose {

using mouse_pose::optical_mouse::BodyJoints;
constexpr double kDefaultKeypointWeight = 1.;

absl::Status PoseOptimizer::LoadRiggedMeshFromFiles(
    std::string obj_filename, std::string weight_filename) {
  mesh_ = CreateRiggedMeshFromFiles(obj_filename, weight_filename);

  return absl::OkStatus();
}

absl::Status PoseOptimizer::LoadGmmFromFile(std::string gmm_filename) {
  gmm_ = std::make_unique<GaussianMixtureModel>();
  gmm_->InitializeFromFile(gmm_filename);
  gmm_->CheckIsScaledAndRotated();
  return absl::OkStatus();
}

absl::Status PoseOptimizer::LoadMouseTargetPointsFromMap(
    const absl::flat_hash_map<std::string, std::pair<float, float>>&
        target_keypoints,
    const absl::flat_hash_map<std::string, float>& input_keypoint_weights,
    std::vector<double>* out_keypoint_weights) {
  auto append_key_point = [&](const Eigen::Vector2d pos, double weight) {
    target_2d_points_.push_back(pos);
    if (out_keypoint_weights != nullptr) {
      out_keypoint_weights->push_back(weight);
    }
  };

  target_2d_points_.clear();
  const std::vector<std::string> chain_joint_names = chain_.GetJointNames();
  for (auto& keypoint_name : chain_joint_names) {
    auto it = target_keypoints.find(keypoint_name);
    if (it != target_keypoints.end()) {
      auto [x, y] = it->second;
      double weight = kDefaultKeypointWeight;
      auto it_weight = input_keypoint_weights.find(keypoint_name);
      if (it_weight != input_keypoint_weights.end()) weight = it_weight->second;
      append_key_point(Eigen::Vector2d(x, y), weight);
    } else if (keypoint_name == "neck") {
      auto it_left = target_keypoints.find("left_ear");
      auto it_right = target_keypoints.find("right_ear");
      if (it_left == target_keypoints.end() ||
          it_right == target_keypoints.end()) {
        LOG(ERROR)
            << "Left or right ear used to compute neck position is missing.";
        append_key_point(Eigen::Vector2d::Zero(), 0);
        break;
      }

      auto [lx, ly] = it_left->second;
      auto [rx, ry] = it_right->second;
      Eigen::Vector2d left_ear(lx, ly);
      Eigen::Vector2d right_ear(rx, ry);
      append_key_point(0.5 * (left_ear + right_ear), kDefaultKeypointWeight);
    } else if (keypoint_name == "spine_hip") {
      auto it_left = target_keypoints.find("left_hip");
      auto it_right = target_keypoints.find("right_hip");
      if (it_left == target_keypoints.end() ||
          it_right == target_keypoints.end()) {
        LOG(ERROR)
            << "Left or right hip used to compute spine position is missing.";
        append_key_point(Eigen::Vector2d::Zero(), 0);
        break;
      }

      auto [lx, ly] = it_left->second;
      auto [rx, ry] = it_right->second;
      Eigen::Vector2d left_hip(lx, ly);
      Eigen::Vector2d right_hip(rx, ry);
      append_key_point(0.5 * (left_hip + right_hip), kDefaultKeypointWeight);
    } else {
      LOG(WARNING)
          << "Key point " << keypoint_name
          << " is not in the target. Disabled it by setting weight to 0.";
      append_key_point(Eigen::Vector2d::Zero(), 0);
    }
  }
  return absl::OkStatus();
}

absl::Status PoseOptimizer::LoadTargetPointsFromBodyJoints(
    const BodyJoints& joints, std::vector<double>* keypoint_weights) {
  absl::flat_hash_map<std::string, std::pair<float, float>> target_keypoints;
  absl::flat_hash_map<std::string, float> input_keypoint_weights;
  for (const auto& joint : joints.key_points()) {
    target_keypoints[joint.name()] =
        std::pair(joint.position_2d(0), joint.position_2d(1));
    if (joint.has_weight()) {
      input_keypoint_weights[joint.name()] = joint.weight();
    } else {
      input_keypoint_weights[joint.name()] = kDefaultKeypointWeight;
    }
  }
  return LoadMouseTargetPointsFromMap(target_keypoints, input_keypoint_weights,
                                      keypoint_weights);
}

ceres::Solver::Summary PoseOptimizer::JointAngleAndRigidFit(
    std::vector<double>* rotate_params, std::vector<double>* trans_params,
    std::vector<double>* joint_angles, const std::vector<double>& weights,
    const ceres::Solver::Options& options,
    const absl::flat_hash_map<int, std::pair<double, double>>*
        trans_constraints,
    const absl::flat_hash_map<int, std::pair<double, double>>*
        rotate_constraints,
    const absl::flat_hash_map<int, std::pair<double, double>>*
        alpha_constraints) {
  auto problem = std::make_unique<ceres::Problem>();
  ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<JointAngleAndRigidCostFunction,
                                      ceres::DYNAMIC, 3, 3,
                                      kNumJointAnglesToOptimize>(
          new JointAngleAndRigidCostFunction(chain_, target_2d_points_,
                                             projection_mat_, weights),
          target_2d_points_.size() * 2);
  problem->AddResidualBlock(cost_function, nullptr, rotate_params->data(),
                            trans_params->data(), joint_angles->data());
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
  if (rotate_constraints != nullptr) {
    for (auto const& alpha_constraint : *alpha_constraints) {
      problem->SetParameterLowerBound(joint_angles->data(),
                                      alpha_constraint.first,
                                      alpha_constraint.second.first);
      problem->SetParameterUpperBound(joint_angles->data(),
                                      alpha_constraint.first,
                                      alpha_constraint.second.second);
    }
  }
  ceres::Solver::Summary summary;
  Solve(options, problem.get(), &summary);
  return summary;
}

ceres::Solver::Summary PoseOptimizer::JointAnglePosePriorAndRigidFit(
    std::vector<double>* rotate_params, std::vector<double>* trans_params,
    std::vector<double>* joint_angles, const std::vector<double>& weights,
    const double pose_prior_weight, const ceres::Solver::Options& options,
    const absl::flat_hash_map<int, std::pair<double, double>>*
        trans_constraints,
    const absl::flat_hash_map<int, std::pair<double, double>>*
        rotate_constraints,
    const absl::flat_hash_map<int, std::pair<double, double>>*
        alpha_constraints) {
  auto problem = std::make_unique<ceres::Problem>();
  CHECK(gmm_) << "Must initialize the GMM first.";
  ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<JointAnglePosePriorAndRigidCostFunction,
                                      ceres::DYNAMIC, 3, 3,
                                      kNumJointAnglesToOptimize>(
          new JointAnglePosePriorAndRigidCostFunction(chain_, target_2d_points_,
                                                      projection_mat_, weights,
                                                      *gmm_, pose_prior_weight),
          target_2d_points_.size() * 2);
  problem->AddResidualBlock(cost_function, nullptr, rotate_params->data(),
                            trans_params->data(), joint_angles->data());
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
  if (rotate_constraints != nullptr) {
    for (auto const& alpha_constraint : *alpha_constraints) {
      problem->SetParameterLowerBound(joint_angles->data(),
                                      alpha_constraint.first,
                                      alpha_constraint.second.first);
      problem->SetParameterUpperBound(joint_angles->data(),
                                      alpha_constraint.first,
                                      alpha_constraint.second.second);
    }
  }
  ceres::Solver::Summary summary;
  Solve(options, problem.get(), &summary);
  return summary;
}

ceres::Solver::Summary PoseOptimizer::JointAnglePosePriorShapeBasisAndRigidFit(
    std::vector<double>* rotate_params, std::vector<double>* trans_params,
    std::vector<double>* joint_angles, std::vector<double>* shape_weights,
    const std::vector<double>& weights, const double pose_prior_weight,
    const double shape_basis_l2_weight, const ceres::Solver::Options& options,
    const absl::flat_hash_map<int, std::pair<double, double>>*
        trans_constraints,
    const absl::flat_hash_map<int, std::pair<double, double>>*
        rotate_constraints,
    const absl::flat_hash_map<int, std::pair<double, double>>*
        alpha_constraints) {
  auto problem = std::make_unique<ceres::Problem>();
  CHECK(gmm_) << "Must initialize the GMM first.";
  ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<
      JointAnglePosePriorShapeBasisAndRigidCostFunction, ceres::DYNAMIC, 3, 3,
      kNumJointAnglesToOptimize, kNumShapeBasisComponents>(
      new JointAnglePosePriorShapeBasisAndRigidCostFunction(
          chain_, target_2d_points_, projection_mat_, weights, *gmm_,
          pose_prior_weight, shape_basis_l2_weight),
      target_2d_points_.size() * 2);
  problem->AddResidualBlock(cost_function, nullptr, rotate_params->data(),
                            trans_params->data(), joint_angles->data(),
                            shape_weights->data());
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
  if (rotate_constraints != nullptr) {
    for (auto const& alpha_constraint : *alpha_constraints) {
      problem->SetParameterLowerBound(joint_angles->data(),
                                      alpha_constraint.first,
                                      alpha_constraint.second.first);
      problem->SetParameterUpperBound(joint_angles->data(),
                                      alpha_constraint.first,
                                      alpha_constraint.second.second);
    }
  }
  ceres::Solver::Summary summary;
  Solve(options, problem.get(), &summary);
  return summary;
}

absl::Status PoseOptimizer::RigidFitTo3D(
    const std::vector<double>& weights, std::vector<double>* trans_params,
    std::vector<double>* rotate_params, const ceres::Solver::Options& options) {
  ceres::Problem problem;
  ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<RigidTrans3DCostFunction, ceres::DYNAMIC,
                                      3, 3>(
          new RigidTrans3DCostFunction(chain_.GetJoints(), target_3d_points_,
                                       weights),
          target_3d_points_.size() * 3);
  problem.AddResidualBlock(cost_function, nullptr, rotate_params->data(),
                           trans_params->data());
  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);
  return absl::OkStatus();
}

ceres::Solver::Summary PoseOptimizer::JointAngleFitTo3D(
    const std::vector<double>& weights, std::vector<double>* trans_params,
    std::vector<double>* rotate_params, std::vector<double>* joint_angles,
    const ceres::Solver::Options& options) {
  ceres::Problem problem;
  ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<Pose3DToAngleCostFunction, ceres::DYNAMIC,
                                      3, 3, kNumJointAnglesToOptimize>(
          new Pose3DToAngleCostFunction(chain_, target_3d_points_, weights),
          target_3d_points_.size() * 3);
  problem.AddResidualBlock(cost_function, nullptr, rotate_params->data(),
                           trans_params->data(), joint_angles->data());
  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);
  return summary;
}

ceres::Solver::Summary PoseOptimizer::JointAnglePcaFit(
    std::vector<double>* rotation, std::vector<double>* translation,
    std::vector<double>* joint_angles, const std::vector<double>& weights,
    const ceres::Solver::Options& options) {
  std::vector<double> pca_loadings(kNumAnglePcaCoefficientsToOptimize, 0.0);
  ceres::Problem problem;
  ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<JointAnglePcaCostFunction, ceres::DYNAMIC,
                                      3, 3, kNumAnglePcaCoefficientsToOptimize>(
          new JointAnglePcaCostFunction(chain_, target_2d_points_,
                                        projection_mat_, weights),
          target_2d_points_.size() * 2);
  problem.AddResidualBlock(cost_function, nullptr, rotation->data(),
                           translation->data(), pca_loadings.data());
  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);

  Eigen::Matrix<double, 1, Eigen::Dynamic> loading_mat;
  loading_mat.resize(1, pca_loadings.size());
  for (int j = 0; j < pca_loadings.size(); ++j) {
    loading_mat(0, j) = pca_loadings[j];
  }
  Eigen::Matrix<double, 1, Eigen::Dynamic> angle_mat;
  angle_mat.resize(1, chain_.GetJoints().size() * 3);
  angle_mat = loading_mat * chain_.GetAlphaBasis() + chain_.GetAlphaBasisMean();
  joint_angles->clear();
  joint_angles->insert(joint_angles->begin(), angle_mat.begin(),
                       angle_mat.end());
  return summary;
}

// Finds the initial estimate of translation and rotation.
void PoseOptimizer::ComputeRigidBodyTransform(
    const std::vector<double>& joint_weights, std::vector<double>* translation,
    std::vector<double>* rotation, bool use_init_guess) {
  std::vector<cv::Point2f> image_points;
  std::vector<cv::Point3f> object_points;
  const auto& joints = chain_.GetJoints();
  for (int i = 0; i < target_2d_points_.size(); ++i) {
    auto& p2 = target_2d_points_[i];
    auto& p3 = joints[i];
    // Skip unused joints, which have 0 weight.
    if (joint_weights[i] > 0) {
      image_points.push_back(cv::Point2f(p2[0], p2[1]));
      object_points.push_back(cv::Point3f(p3[0], p3[1], p3[2]));
    }
  }

  cv::Mat projection_matrix;
  cv::eigen2cv(projection_mat_, projection_matrix);

  cv::Mat camera_matrix, rot, trans;
  // Drop the rotation and translation embedded in the projection matrix.
  cv::decomposeProjectionMatrix(projection_matrix, camera_matrix, rot, trans);
  std::vector<double> distortion(5, 0.0);

  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F),
          tvec = cv::Mat::zeros(3, 1, CV_64F);

  if (use_init_guess) {
    tvec.at<double>(0) = (*translation)[0];
    tvec.at<double>(1) = (*translation)[1];
    tvec.at<double>(2) = (*translation)[2];

    rvec.at<double>(0) = (*rotation)[0];
    rvec.at<double>(1) = (*rotation)[1];
    rvec.at<double>(2) = (*rotation)[2];
  }
  cv::solvePnPRansac(object_points, image_points, camera_matrix, distortion,
                     rvec, tvec, use_init_guess);

  (*translation)[0] = tvec.at<double>(0);
  (*translation)[1] = tvec.at<double>(1);
  (*translation)[2] = tvec.at<double>(2);

  (*rotation)[0] = rvec.at<double>(0);
  (*rotation)[1] = rvec.at<double>(1);
  (*rotation)[2] = rvec.at<double>(2);
}
}  // namespace mouse_pose
