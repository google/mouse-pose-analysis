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

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "ceres/types.h"
#include "glog/logging.h"
#include "mouse_pose_analysis/pose_3d/pose_optimizer.h"
#include "mouse_pose_analysis/pose_3d/pose_reconstruction.pb.h"
#include "mouse_pose_analysis/pose_3d/pose_reconstruction_utils.h"

namespace mouse_pose {

using mouse_pose::optical_mouse::CameraParameters;
using mouse_pose::optical_mouse::InputParameters;
using mouse_pose::optical_mouse::OptimizationMethod;
using mouse_pose::optical_mouse::OptimizationOptions;

absl::Status LoadProjectionMatFromCameraParameters(
    const CameraParameters &camera_params,
    Eigen::Matrix<double, 3, 4> *projection_matrix) {
  *projection_matrix = Eigen::Matrix<double, 3, 4>::Zero();
  int i = 0;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      (*projection_matrix)(r, c) = camera_params.intrinsics(i++);
  return absl::OkStatus();
}

void SetUpOptimizer(const InputParameters &input_parameters,
                    PoseOptimizer *pose_optimizer) {
  // Load bone configuration.
  QCHECK_OK(mouse_pose::LoadAndConstructKinematicChain(
      input_parameters.skeleton_config_file(),
      &(pose_optimizer->GetKinematicChain())));
  // Load camera matrix.
  QCHECK_OK(LoadProjectionMatFromCameraParameters(
      input_parameters.camera_parameters(),
      &(pose_optimizer->GetProjectionMat())));

  switch (input_parameters.method()) {
    case OptimizationMethod::JOINT_ANGLE_PCA:
      QCHECK_OK(pose_optimizer->GetKinematicChain().ReadAnglePcaBasisConfig(
          input_parameters.pose_pca_file()));
      break;

    case OptimizationMethod::JOINT_ANGLE_POSE_PRIOR_AND_RIGID_FIT:
      QCHECK_OK(pose_optimizer->LoadGmmFromFile(input_parameters.gmm_file()));
      break;

    case OptimizationMethod::JOINT_ANGLE_POSE_PRIOR_SHAPE_BASIS_AND_RIGID_FIT:
      QCHECK_OK(pose_optimizer->LoadGmmFromFile(input_parameters.gmm_file()));
      QCHECK_OK(pose_optimizer->GetKinematicChain().ReadShapePcaBasisConfig(
          input_parameters.shape_basis_file()));
      break;
    default:
      // Not all methods need set-up.
      break;
  }
}

// Loads translation constraints from an InputParameters structure.
// This assumes all 3 (x, y, z) constraints are provided.
std::unique_ptr<absl::flat_hash_map<int, std::pair<double, double>>>
CreateTranslationConstraints(const InputParameters &input_parameters) {
  if (input_parameters.has_optimization_constraints() &&
      input_parameters.optimization_constraints().translation_size() > 0 &&
      input_parameters.method() !=
          OptimizationMethod::JOINT_ANGLE_POSE_PRIOR_AND_RIGID_FIT) {
    auto trans_constraints =
        std::make_unique<absl::flat_hash_map<int, std::pair<double, double>>>();
    for (int i = 0; i < 3; ++i) {
      double ub = input_parameters.optimization_constraints()
                      .translation(i)
                      .upper_bound();
      double lb = input_parameters.optimization_constraints()
                      .translation(i)
                      .lower_bound();
      (*trans_constraints)[i] = std::pair<double, double>(lb, ub);
    }
    return trans_constraints;
  }
  return nullptr;
}

// Loads rotation constraints from an InputParameters structure.
// This assumes all 3 (rx, ry, rz) constraints are provided.
std::unique_ptr<absl::flat_hash_map<int, std::pair<double, double>>>
CreateRotationConstraints(const InputParameters &input_parameters) {
  // Since we use angle-axis representation, the rotation constraints don't make
  // much sense if set up this way.
  if (input_parameters.has_optimization_constraints() &&
      input_parameters.optimization_constraints().rotation_size() > 0 &&
      input_parameters.method() !=
          OptimizationMethod::JOINT_ANGLE_POSE_PRIOR_AND_RIGID_FIT) {
    auto rotate_constraints =
        std::make_unique<absl::flat_hash_map<int, std::pair<double, double>>>();
    for (int i = 0; i < 3; ++i) {
      double ub =
          input_parameters.optimization_constraints().rotation(i).upper_bound();
      double lb =
          input_parameters.optimization_constraints().rotation(i).lower_bound();
      (*rotate_constraints)[i] = std::pair<double, double>(lb, ub);
    }
    return rotate_constraints;
  }
  return nullptr;
}

// Loads constraints on joints angles.
std::unique_ptr<absl::flat_hash_map<int, std::pair<double, double>>>
CreateAngleConstraints(const InputParameters &input_parameters,
                       int num_joints) {
  auto angles_constraints =
      std::make_unique<absl::flat_hash_map<int, std::pair<double, double>>>();
  if (input_parameters.has_optimization_constraints() &&
      input_parameters.optimization_constraints().joint_angles_size() > 0) {
    CHECK(input_parameters.optimization_constraints().joint_angles_size() ==
          num_joints * 3)
        << "Number of constraints "
        << input_parameters.optimization_constraints().joint_angles_size()
        << " should be 3 times of number of joints " << num_joints;
    for (int i = 0; i < num_joints; ++i) {
      for (int j = 0; j < 3; ++j) {
        int n = i * 3 + j;
        double ub = input_parameters.optimization_constraints()
                        .joint_angles(n)
                        .upper_bound();
        double lb = input_parameters.optimization_constraints()
                        .joint_angles(n)
                        .lower_bound();
        (*angles_constraints)[n] = std::pair<double, double>(lb, ub);
      }
    }

  } else {
    // The hard-coded default constraints.
    std::vector<int> spinal_bone_indices({1, 2, 10});
    for (int i = 0; i < num_joints; ++i) {
      for (int j = 0; j < 3; ++j) {
        (*angles_constraints)[i * 3 + j] = std::pair<double, double>(-0.9, 0.9);
      }
    }
  }

  return angles_constraints;
}

// Sets default optimization options and loads values from supplied
// options.
ceres::Solver::Options SetOptimizationOptions(
    const OptimizationOptions &options) {
  ceres::Solver::Options ceres_options;

  // Set default values.
  ceres_options.num_threads = 5;
  ceres_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  ceres_options.max_num_iterations = 1000;
  ceres_options.minimizer_progress_to_stdout = true;
  ceres_options.logging_type = ceres::SILENT;

  // Override options if provided.
  if (options.num_threads() > 0)
    ceres_options.num_threads = options.num_threads();

  if (options.max_num_iterations() > 0)
    ceres_options.max_num_iterations = options.max_num_iterations();

  if (options.minimizer_progress_to_stdout() == false)
    ceres_options.minimizer_progress_to_stdout = false;

  if (!options.logging_type().empty()) {
    if (options.logging_type() == "per_minimizer_iteration")
      ceres_options.logging_type = ceres::PER_MINIMIZER_ITERATION;
    else if (options.logging_type() == "silent")
      ceres_options.logging_type = ceres::SILENT;
  }

  if (!options.trust_region_strategy_type().empty()) {
    if (options.trust_region_strategy_type() == "dogleg")
      ceres_options.trust_region_strategy_type = ceres::DOGLEG;
    else if (options.trust_region_strategy_type() == "levenberg_marquardt")
      ceres_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  }

  if (options.initial_trust_region_radius() > 0.)
    ceres_options.initial_trust_region_radius =
        options.initial_trust_region_radius();

  return ceres_options;
}

// Reconstructs pose.
// Outputs a global rigid body transform in trans_params and rotate_params,
// joint angles in alphas and coefficients of shape space bases.
absl::Status ReconstructPose(const InputParameters &input_parameters,
                             PoseOptimizer *pose_optimizer,
                             const std::vector<double> &keypoint_weights,
                             std::vector<double> *trans_params,
                             std::vector<double> *rotate_params,
                             std::vector<double> *alphas,
                             std::vector<double> *shape_basis_weights) {
  CHECK(trans_params != nullptr) << "Always fit global translation.";
  CHECK(rotate_params != nullptr) << "Always fit global rotation.";

  ceres::Solver::Options options =
      SetOptimizationOptions(input_parameters.optimization_options());

  CHECK(keypoint_weights.size() == mouse_pose::kNumJointsToOptimize)
      << "Expecting weight size to be " << mouse_pose::kNumJointsToOptimize
      << ". Get " << keypoint_weights.size() << ".";

  std::vector<double> local_alphas(mouse_pose::kNumJointAnglesToOptimize, 0);
  alphas->swap(local_alphas);
  if (shape_basis_weights != nullptr) {
    std::vector<double> local_shape_basis_weights(
        mouse_pose::kNumShapeBasisComponents, 0);
    shape_basis_weights->swap(local_shape_basis_weights);
  }

  bool use_init_guess = false;
  if (input_parameters.initial_rigid_body_values_size() > 0) {
    use_init_guess = true;
    (*rotate_params)[0] = input_parameters.initial_rigid_body_values(0);
    (*rotate_params)[1] = input_parameters.initial_rigid_body_values(1);
    (*rotate_params)[2] = input_parameters.initial_rigid_body_values(2);
    (*trans_params)[0] = input_parameters.initial_rigid_body_values(3);
    (*trans_params)[1] = input_parameters.initial_rigid_body_values(4);
    (*trans_params)[2] = input_parameters.initial_rigid_body_values(5);
  }

  auto trans_constraints = CreateTranslationConstraints(input_parameters);
  auto rotate_constraints = CreateRotationConstraints(input_parameters);

  // Fit rigid transformation parameters.
  if (input_parameters.method() != OptimizationMethod::NONE) {
    if (input_parameters.prefer_ceres_rigid_fit_to_open_cv()) {
      CHECK_OK(pose_optimizer->RigidFit<18>(
          trans_params, rotate_params, keypoint_weights, options,
          trans_constraints.get(), rotate_constraints.get()));
    } else {
      pose_optimizer->ComputeRigidBodyTransform(keypoint_weights, trans_params,
                                                rotate_params, use_init_guess);
    }
  }

  if (alphas == nullptr) {
    LOG(INFO) << "Global rigid body fit only.";
    return absl::OkStatus();
  }

  auto angles_constraints =
      CreateAngleConstraints(input_parameters, alphas->size() / 3);

  switch (input_parameters.method()) {
    case OptimizationMethod::NONE: {
      break;  // Do nothing.
    }
    case OptimizationMethod::RIGID_FIT: {
      break;  // Do nothing because it's already completed.
    }
    case OptimizationMethod::JOINT_ANGLE: {
      auto summary = pose_optimizer->JointAngleFit<18>(
          Eigen::Vector3d(rotate_params->data()),
          Eigen::Vector3d(trans_params->data()), alphas, keypoint_weights,
          options, angles_constraints.get());
      LOG(INFO) << "Summary\n" << summary.BriefReport() << "\n";
      break;
    }

    case OptimizationMethod::JOINT_ANGLE_POSE_PRIOR_AND_RIGID_FIT: {
      auto summary = pose_optimizer->JointAnglePosePriorAndRigidFit(
          rotate_params, trans_params, alphas, keypoint_weights,
          input_parameters.pose_prior_weight(), options,
          trans_constraints.get(), rotate_constraints.get(),
          angles_constraints.get());
      LOG(INFO) << "Summary\n" << summary.BriefReport() << "\n";
      break;
    }
    case OptimizationMethod::JOINT_ANGLE_PCA: {
      options.initial_trust_region_radius = 1e1;
      options.trust_region_strategy_type = ceres::DOGLEG;
      options.max_num_iterations = 50;
      pose_optimizer->JointAnglePcaFit(rotate_params, trans_params, alphas,
                                       keypoint_weights, options);
      break;
    }
    case OptimizationMethod::JOINT_ANGLE_POSE_PRIOR_SHAPE_BASIS_AND_RIGID_FIT: {
      pose_optimizer->JointAnglePosePriorShapeBasisAndRigidFit(
          rotate_params, trans_params, alphas, shape_basis_weights,
          keypoint_weights, input_parameters.pose_prior_weight(),
          input_parameters.shape_prior_weight());
      break;
    }
    case OptimizationMethod::JOINT_ANGLE_AND_RIGID_FIT: {
      pose_optimizer->JointAngleAndRigidFit(
          rotate_params, trans_params, alphas, keypoint_weights, options,
          trans_constraints.get(), rotate_constraints.get(),
          angles_constraints.get());
      break;
    }
    default:
      LOG(FATAL) << "Unknown optimization method:" << input_parameters.method();
      return absl::InvalidArgumentError("Unknown optimization method");
  }

  return absl::OkStatus();
}

}  // namespace mouse_pose
