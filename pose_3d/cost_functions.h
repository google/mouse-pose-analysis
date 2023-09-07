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

#ifndef MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_COST_FUNCTIONS_H_
#define MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_COST_FUNCTIONS_H_
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "absl/strings/str_format.h"
#include "ceres/rotation.h"
#include "opencv2/core/core.hpp"
#include "mouse_pose_analysis/pose_3d/gmm_prior.h"
#include "mouse_pose_analysis/pose_3d/kinematic_chain.h"
#include "mouse_pose_analysis/pose_3d/matrix_util.h"
#include "mouse_pose_analysis/pose_3d/rigged_mesh.h"
#include "mouse_pose_analysis/pose_3d/status.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/MatrixFunctions"

namespace mouse_pose {

// The cost function to optimize the rigid body transformation parameters:
// rotation and translation. The cost function takes, the 3D keypoint locations,
// target 2D keypoint locations and the projection camera matrices as input.
// And it computes the residual as the L1 norm of the difference between
// the projected 3D keypoint locations and the target 2D keypoint locations.
// The residual will be used by CERES to do auto diff to compute the gradients.
class RigidTransCostFunction {
 public:
  // Initializes the cost function with the 3D keypoint locations, the target
  // 2D keypoint locations, the camera projection matrix, and the weight
  // for each target keypoint.
  RigidTransCostFunction(const std::vector<Eigen::Vector3d>& source_points,
                         const std::vector<Eigen::Vector2d>& target_points,
                         const Eigen::Matrix<double, 3, 4>& projection_matrix,
                         const std::vector<double>& weights)
      : source_points_(source_points),
        target_points_(target_points),
        projection_matrix_(projection_matrix),
        weights_(weights) {}

  // Evaluates the residual (or the reprojection_error) based on the rotation
  // translation parameters, 3D keypoint locations, target 2D keypoint locations
  // and the camera projection matrix.
  template <typename T>
  bool operator()(const T* rotation, const T* translation,
                  T* reprojection_error) const {
    typedef Eigen::Matrix<T, 3, 1> Vector3;
    typedef Eigen::Matrix<T, 4, 1> Vector4;
    typedef Eigen::Map<const Vector3> Vector3Ref;

    Vector3Ref rotation_mat(rotation);
    Vector3Ref translation_mat(translation);
    for (size_t i = 0; i < source_points_.size(); ++i) {
      const Vector3 source_point = source_points_[i].template cast<T>();
      Vector3 rotated_point;
      ceres::AngleAxisRotatePoint(rotation_mat.data(), source_point.data(),
                                  rotated_point.data());
      Vector3 translated_point;
      translated_point = rotated_point + translation_mat;
      Vector4 augmented_point;
      augmented_point.block(0, 0, 3, 1) << translated_point;
      augmented_point(3, 0) = static_cast<T>(1.);
      const Eigen::Matrix<T, 2, 1> reprojected_point =
          (projection_matrix_.cast<T>() * augmented_point).hnormalized();
      reprojection_error[i * 2] =
          (reprojected_point[0] - target_points_[i][0]) * weights_[i];
      reprojection_error[i * 2 + 1] =
          (reprojected_point[1] - target_points_[i][1]) * weights_[i];
    }
    return true;
  }

 private:
  // List of the source 3D keypoint locations.
  const std::vector<Eigen::Vector3d> source_points_;

  // List of the target 2D keypoint locations.
  const std::vector<Eigen::Vector2d> target_points_;

  // The input camera projection matrix.
  const Eigen::Matrix<double, 3, 4> projection_matrix_;

  // The weights for each joints.
  const std::vector<double> weights_;
};

// The cost function to optimize the joint angles of an articulated body:
// alphas. The cost function takes, the kinematic chain,
// target 2D keypoint, rigid transformations and the projection camera matrices
// as input. And it computes the residual as the L1 norm of the difference of
// the projected 3D keypoint locations and the target 2D keypoint locations.
// The residual will be used by CERES to do auto diff to compute the gradients.
class JointAngleCostFunction {
 public:
  // Initializes the cost function with the kinematic chain, the target
  // 2D keypoint locations, rigid transformation parmeters,
  // the camera projection matrix and weight for each target keypoint.
  JointAngleCostFunction(const KinematicChain& kinematic_chain,
                         const std::vector<Eigen::Vector2d>& target_points,
                         const Eigen::Matrix<double, 3, 4>& projection_matrix,
                         const Eigen::Vector3d& rotation,
                         const Eigen::Vector3d& translation,
                         const std::vector<double>& weights)
      : kinematic_chain_(kinematic_chain),
        target_points_(target_points),
        projection_matrix_(projection_matrix),
        rotation_(rotation),
        translation_(translation),
        num_alphas_(3 * kinematic_chain.GetJoints().size()),
        weights_(weights) {}

  // Evaluates the residual (or the reprojection_error) based on the joint
  // angles (or the alphas), kinematic chain, target 2D keypoint locations
  // and the camera projection matrix.
  template <typename T>
  bool operator()(const T* alphas, T* reprojection_error) const {
    typedef Eigen::Matrix<T, 4, 1> Vector4;
    typedef Eigen::Matrix<T, 3, 1> Vector3;
    typedef Eigen::Matrix<T, 4, 4> Matrix4;
    // Use unique pointer to avoid atack overflow.
    auto updated_source_points = std::make_unique<std::vector<Vector4>>();
    auto chain_G = std::make_unique<std::vector<Matrix4>>();
    Vector3 rotation_mat;
    rotation_mat(0) = static_cast<T>(rotation_[0]);
    rotation_mat(1) = static_cast<T>(rotation_[1]);
    rotation_mat(2) = static_cast<T>(rotation_[2]);
    Vector3 translation_mat;
    translation_mat(0) = static_cast<T>(translation_[0]);
    translation_mat(1) = static_cast<T>(translation_[1]);
    translation_mat(2) = static_cast<T>(translation_[2]);

    // Compute the 3D joint locations based on the joint angles (the alphas)
    // and the kinematic chain.
    std::vector<T> alphas_vec(alphas, alphas + num_alphas_);
    kinematic_chain_.UpdateKinematicChain(
        alphas_vec, updated_source_points.get(), chain_G.get());

    // Transform the updated 3D joint locations with rotation and translation.
    // Reproject the transformed 3D joint locations to compute the reprojection
    // error.
    for (size_t i = 0; i < updated_source_points->size(); i++) {
      const Vector3 updated_source_point =
          (*updated_source_points)[i].template head<3>();
      Vector3 rotated_point;
      ceres::AngleAxisRotatePoint(rotation_mat.data(),
                                  updated_source_point.data(),
                                  rotated_point.data());
      Vector3 translated_point = rotated_point + translation_mat;
      Vector4 augmented_point;
      augmented_point.block(0, 0, 3, 1) << translated_point;
      augmented_point(3, 0) = static_cast<T>(1.);
      const Eigen::Matrix<T, 2, 1> reprojected_point =
          (projection_matrix_.cast<T>() * augmented_point).hnormalized();
      reprojection_error[i * 2] =
          (reprojected_point[0] - target_points_[i][0]) * weights_[i];
      reprojection_error[i * 2 + 1] =
          (reprojected_point[1] - target_points_[i][1]) * weights_[i];
    }
    return true;
  }

 private:
  // The kinematic chain of the articulated body.
  const KinematicChain kinematic_chain_;

  // The list of target 2D point locations.
  const std::vector<Eigen::Vector2d> target_points_;

  // The camera projection matrix.
  const Eigen::Matrix<double, 3, 4> projection_matrix_;

  // The rigid body rotation parameters.
  const Eigen::Vector3d rotation_;

  // The rigid body translation parameters.
  const Eigen::Vector3d translation_;

  // The number of joint angles to optimize.
  const int num_alphas_;

  // The weights for each joints.
  const std::vector<double> weights_;
};

// The cost function to optimize the joint angles of an articulated body
// as well as the rigid translation and rotation: alphas, rotation, and
// translation. The cost function takes, the kinematic chain, target 2D
// keypoints, rigid transformations and the projection camera matrices as input.
// And it computes the residual as the L1 norm of the difference of
// the projected 3D keypoint locations and the target 2D keypoint locations.
// The residual will be used by CERES to do auto diff to compute the gradients.
class JointAngleAndRigidCostFunction {
 public:
  // Initializes the cost function with the kinematic chain, the target
  // 2D keypoint locations, rigid transformation parmeters,
  // the camera projection matrix and weight for each target keypoint.
  JointAngleAndRigidCostFunction(
      const KinematicChain& kinematic_chain,
      const std::vector<Eigen::Vector2d>& target_points,
      const Eigen::Matrix<double, 3, 4>& projection_matrix,
      const std::vector<double>& weights)
      : kinematic_chain_(kinematic_chain),
        target_points_(target_points),
        projection_matrix_(projection_matrix),
        num_alphas_(3 * kinematic_chain.GetJoints().size()),
        weights_(weights) {}

  // Evaluates the residual (or the reprojection_error) based on the joint
  // angles (or the alphas), kinematic chain, target 2D keypoint locations
  // and the camera projection matrix.
  template <typename T>
  bool operator()(const T* rotation, const T* translation, const T* alphas,
                  T* reprojection_error) const {
    typedef Eigen::Matrix<T, 4, 1> Vector4;
    typedef Eigen::Matrix<T, 3, 1> Vector3;
    typedef Eigen::Matrix<T, 4, 4> Matrix4;
    typedef Eigen::Map<const Vector3> Vector3Ref;
    // Use unique pointer to avoid stack overflow.
    auto updated_source_points = std::make_unique<std::vector<Vector4>>();
    auto chain_G = std::make_unique<std::vector<Matrix4>>();
    Vector3Ref rotation_mat(rotation);
    Vector3Ref translation_mat(translation);

    // Compute the 3D joint locations based on the joint angles (the alphas)
    // and the kinematic chain.
    std::vector<T> alphas_vec(alphas, alphas + num_alphas_);
    kinematic_chain_.UpdateKinematicChain(
        alphas_vec, updated_source_points.get(), chain_G.get());

    // Transform the updated 3D joint locations with rotation and translation.
    // Reproject the transformed 3D joint locations to compute the reprojection
    // error.
    for (size_t i = 0; i < updated_source_points->size(); i++) {
      const Vector3 updated_source_point =
          (*updated_source_points)[i].template head<3>();
      Vector3 rotated_point;
      ceres::AngleAxisRotatePoint(rotation_mat.data(),
                                  updated_source_point.data(),
                                  rotated_point.data());
      Vector3 translated_point = rotated_point + translation_mat;
      Vector4 augmented_point;
      augmented_point.block(0, 0, 3, 1) << translated_point;
      augmented_point(3, 0) = static_cast<T>(1.);
      const Eigen::Matrix<T, 2, 1> reprojected_point =
          (projection_matrix_.cast<T>() * augmented_point).hnormalized();
      reprojection_error[i * 2] =
          (reprojected_point[0] - target_points_[i][0]) * weights_[i];
      reprojection_error[i * 2 + 1] =
          (reprojected_point[1] - target_points_[i][1]) * weights_[i];
    }
    return true;
  }

 private:
  // The kinematic chain of the articulated body.
  const KinematicChain kinematic_chain_;

  // The list of target 2D point locations.
  const std::vector<Eigen::Vector2d> target_points_;

  // The camera projection matrix.
  const Eigen::Matrix<double, 3, 4> projection_matrix_;

  // The number of joint angles to optimize.
  const int num_alphas_;

  // The weights for each joints.
  const std::vector<double> weights_;
};

// The cost function to optimize the joint angles of an articulated body
// as well as the rigid translation and rotation: alphas, rotation, and
// translation. The cost function takes, the kinematic chain, target 2D
// keypoints, rigid transformations and the projection camera matrices as input.
// And it computes the residual as the L1 norm of the difference of
// the projected 3D keypoint locations and the target 2D keypoint locations.
// Regularizes the fit by imposing a GMM pose prior.
// The residual will be used by CERES to do auto diff to compute the gradients.
class JointAnglePosePriorAndRigidCostFunction {
 public:
  // Initializes the cost function with the kinematic chain, the target
  // 2D keypoint locations, rigid transformation parmeters,
  // the camera projection matrix and weight for each target keypoint.
  JointAnglePosePriorAndRigidCostFunction(
      const KinematicChain& kinematic_chain,
      const std::vector<Eigen::Vector2d>& target_points,
      const Eigen::Matrix<double, 3, 4>& projection_matrix,
      const std::vector<double>& weights,
      const GaussianMixtureModel& pose_prior, const double pose_prior_weight)
      : kinematic_chain_(kinematic_chain),
        target_points_(target_points),
        projection_matrix_(projection_matrix),
        num_alphas_(3 * kinematic_chain.GetJoints().size()),
        weights_(weights),
        pose_prior_(pose_prior),
        pose_prior_weight_(pose_prior_weight) {}

  // Evaluates the residual (or the reprojection_error) based on the joint
  // angles (or the alphas), kinematic chain, target 2D keypoint locations
  // and the camera projection matrix.
  template <typename T>
  bool operator()(const T* rotation, const T* translation, const T* alphas,
                  T* reprojection_error) const {
    typedef Eigen::Matrix<T, 4, 1> Vector4;
    typedef Eigen::Matrix<T, 3, 1> Vector3;
    typedef Eigen::Matrix<T, 4, 4> Matrix4;
    typedef Eigen::Map<const Vector3> Vector3Ref;
    // Use unique pointer to avoid stack overflow.
    auto updated_source_points = std::make_unique<std::vector<Vector4>>();
    auto chain_G = std::make_unique<std::vector<Matrix4>>();
    Vector3Ref rotation_mat(rotation);
    Vector3Ref translation_mat(translation);

    // Compute the 3D joint locations based on the joint angles (the alphas)
    // and the kinematic chain.
    std::vector<T> alphas_vec(alphas, alphas + num_alphas_);
    kinematic_chain_.UpdateKinematicChain(
        alphas_vec, updated_source_points.get(), chain_G.get());

    // Transform the updated 3D joint locations with rotation and translation.
    // Reproject the transformed 3D joint locations to compute the reprojection
    // error.
    std::vector<Eigen::Matrix<T, 3, 1>> points_for_gmm;
    for (size_t i = 0; i < updated_source_points->size(); i++) {
      const Vector3 updated_source_point =
          (*updated_source_points)[i].template head<3>();
      Vector3 rotated_point;
      ceres::AngleAxisRotatePoint(rotation_mat.data(),
                                  updated_source_point.data(),
                                  rotated_point.data());
      Vector3 translated_point = rotated_point + translation_mat;
      Vector4 augmented_point;
      augmented_point.block(0, 0, 3, 1) << translated_point;
      augmented_point(3, 0) = static_cast<T>(1.);
      const Eigen::Matrix<T, 2, 1> reprojected_point =
          (projection_matrix_.cast<T>() * augmented_point).hnormalized();
      reprojection_error[i * 2] =
          (reprojected_point[0] - target_points_[i][0]) * weights_[i];
      reprojection_error[i * 2 + 1] =
          (reprojected_point[1] - target_points_[i][1]) * weights_[i];
      points_for_gmm.push_back(translated_point);
    }
    Eigen::Matrix<T, Eigen::Dynamic, 1> flattened;
    pose_prior_.AlignPointsAndFlatten<T>(points_for_gmm, &flattened);
    // NOTE: taking the negative here to do maximize the likelihood.
    reprojection_error[0] +=
        -pose_prior_weight_ * pose_prior_.LogLikelihood<T>(flattened);
    return true;
  }

 private:
  // The kinematic chain of the articulated body.
  const KinematicChain kinematic_chain_;

  // The list of target 2D point locations.
  const std::vector<Eigen::Vector2d> target_points_;

  // The camera projection matrix.
  const Eigen::Matrix<double, 3, 4> projection_matrix_;

  // The number of joint angles to optimize.
  const int num_alphas_;

  // The weights for each joints.
  const std::vector<double> weights_;

  // The GMM pose prior.
  const GaussianMixtureModel pose_prior_;

  // The weight to apply to the pose_prior.
  const double pose_prior_weight_;
};

// The cost function to optimize the shape parameterized and joint angles of an
// articulated body as well as the rigid translation and rotation: alphas,
// rotation, and translation. The cost function takes, the kinematic chain,
// target 2D keypoints, rigid transformations, weights for shape and pose
// priors, and the projection camera matrices as input. And it computes
// the residual as the L1 norm of the difference of
// the projected 3D keypoint locations and the target 2D keypoint locations.
// Regularizes the fit by imposing a GMM pose prior and a PCA shape prior.
// The residual will be used by CERES to do auto diff to compute the gradients.
class JointAnglePosePriorShapeBasisAndRigidCostFunction {
 public:
  // Initializes the cost function with the kinematic chain, the target
  // 2D keypoint locations, rigid transformation parameters,
  // the camera projection matrix and weight for each target keypoint.
  JointAnglePosePriorShapeBasisAndRigidCostFunction(
      const KinematicChain& kinematic_chain,
      const std::vector<Eigen::Vector2d>& target_points,
      const Eigen::Matrix<double, 3, 4>& projection_matrix,
      const std::vector<double>& weights,
      const GaussianMixtureModel& pose_prior, const double pose_prior_weight,
      const double shape_basis_l2_weight)
      : kinematic_chain_(kinematic_chain),
        target_points_(target_points),
        projection_matrix_(projection_matrix),
        num_alphas_(3 * kinematic_chain.GetJoints().size()),
        weights_(weights),
        pose_prior_(pose_prior),
        pose_prior_weight_(pose_prior_weight),
        shape_basis_l2_weight_(shape_basis_l2_weight) {}

  // Evaluates the residual (or the reprojection_error) based on the joint
  // angles (or the alphas), shape coefficients, kinematic chain, target 2D
  // keypoint locations     and the camera projection matrix.
  template <typename T>
  bool operator()(const T* rotation, const T* translation, const T* alphas,
                  const T* shape_weights, T* reprojection_error) const {
    typedef Eigen::Matrix<T, 4, 1> Vector4;
    typedef Eigen::Matrix<T, 3, 1> Vector3;
    typedef Eigen::Matrix<T, 4, 4> Matrix4;
    typedef Eigen::Map<const Vector3> Vector3Ref;
    // Use unique pointer to avoid stack overflow.
    auto updated_source_points = std::make_unique<std::vector<Vector4>>();
    auto chain_G = std::make_unique<std::vector<Matrix4>>();
    auto updated_joints = std::make_unique<std::vector<Vector3>>();
    Vector3Ref rotation_mat(rotation);
    Vector3Ref translation_mat(translation);

    // Compute the 3D joint locations using the current shape basis.
    CHECK_OK(kinematic_chain_.UpdateJointXYZLocations(shape_weights,
                                                      updated_joints.get()));

    // Compute the 3D joint locations based on the joint locations, joint angles
    // (the alphas) and the kinematic chain.
    std::vector<T> alphas_vec(alphas, alphas + num_alphas_);
    kinematic_chain_.UpdateKinematicChain(alphas_vec,
                                          updated_source_points.get(),
                                          chain_G.get(), updated_joints.get());

    // Transform the updated 3D joint locations with rotation and translation.
    // Reproject the transformed 3D joint locations to compute the reprojection
    // error.
    std::vector<Eigen::Matrix<T, 3, 1>> points_for_gmm;
    for (size_t i = 0; i < updated_source_points->size(); i++) {
      const Vector3 updated_source_point =
          (*updated_source_points)[i].template head<3>();
      Vector3 rotated_point;
      ceres::AngleAxisRotatePoint(rotation_mat.data(),
                                  updated_source_point.data(),
                                  rotated_point.data());
      Vector3 translated_point = rotated_point + translation_mat;
      Vector4 augmented_point;
      augmented_point.block(0, 0, 3, 1) << translated_point;
      augmented_point(3, 0) = static_cast<T>(1.);
      const Eigen::Matrix<T, 2, 1> reprojected_point =
          (projection_matrix_.cast<T>() * augmented_point).hnormalized();
      reprojection_error[i * 2] =
          (reprojected_point[0] - target_points_[i][0]) * weights_[i];
      reprojection_error[i * 2 + 1] =
          (reprojected_point[1] - target_points_[i][1]) * weights_[i];
      reprojection_error[i * 2] *= reprojection_error[i * 2];
      reprojection_error[i * 2 + 1] *= reprojection_error[i * 2 + 1];
      points_for_gmm.push_back(translated_point);
    }
    Eigen::Matrix<T, Eigen::Dynamic, 1> flattened;
    pose_prior_.AlignPointsAndFlatten<T>(points_for_gmm, &flattened);
    // NOTE: taking the negative here to do maximize the likelihood.
    reprojection_error[0] +=
        -pose_prior_weight_ * pose_prior_.LogLikelihood<T>(flattened);

    reprojection_error[1] +=
        shape_basis_l2_weight_ *
        ceres::sqrt(
            Eigen::Map<const Eigen::Matrix<T, kNumShapeBasisComponents, 1>>(
                shape_weights)
                .squaredNorm() +
            1e-5);  // Small constant prevents sqrt(0)=nan.
    return true;
  }

 private:
  // The kinematic chain of the articulated body.
  const KinematicChain kinematic_chain_;

  // The list of target 2D point locations.
  const std::vector<Eigen::Vector2d> target_points_;

  // The camera projection matrix.
  const Eigen::Matrix<double, 3, 4> projection_matrix_;

  // The number of joint angles to optimize.
  const int num_alphas_;

  // The weights for each joints.
  const std::vector<double> weights_;

  // The GMM pose prior.
  const GaussianMixtureModel pose_prior_;

  // The weight to apply to the pose_prior.
  const double pose_prior_weight_;

  // The weight to apply to the l2 norm penalty of the shape basis weights.
  const double shape_basis_l2_weight_;
};

// Computes the minimal positive distance from a point to a set of tangent
// lines.
template <typename T>
T ComputePointSilhouetteDistance(const std::vector<Eigen::Vector3d>& silhouette,
                                 Eigen::Vector<T, 2> point) {
  // T min_dist(std::numeric_limits<T>::max());
  const T big_number(100000);
  T min_dist(big_number);
  for (auto line_segment : silhouette) {
    T dist = point(0) * line_segment(0) + point(1) * line_segment(1) +
             line_segment(2);

    if (dist > static_cast<T>(0) && dist < min_dist) {
      min_dist = dist;
    }
  }

  if (min_dist == big_number)
    return static_cast<T>(0);
  else
    return min_dist;
}

// Computes the distance between the mesh and the silhouette, which is the sum
// of all vertex-silhouette distances.
template <typename T>
T ComputeMeshSilhouetteDistance(
    const std::vector<Eigen::Vector3d>& silhouette,
    const Eigen::Matrix<T, 3, 4>& projection_matrix,
    const std::vector<Eigen::Vector<T, 3>>& vertices) {
  T dist(0);

  for (const auto& vertex : vertices) {
    Eigen::Vector<T, 2> image_point = Project3DPoint(projection_matrix, vertex);
    dist += ComputePointSilhouetteDistance(silhouette, image_point);
  }

  return dist;
}

// Cost function on the image. Image cost is the distance between the input mask
// and projected mesh.
class ImageSilhouetteCostFunction {
 public:
  ImageSilhouetteCostFunction(
      const KinematicChain& kinematic_chain, RiggedMesh* mesh,
      const std::vector<Eigen::Vector3d>& silhouette,
      const Eigen::Matrix<double, 3, 4>& projection_matrix,
      const Eigen::Vector3d& rotation, const Eigen::Vector3d& translation)
      : kinematic_chain_(kinematic_chain),
        mesh_(mesh),
        silhouette_(silhouette),
        projection_matrix_(projection_matrix),
        num_alphas_(3 * kinematic_chain.GetJoints().size()),
        rotation_(rotation),
        translation_(translation) {}

  // Evaluates the image error from joint angles (or the alphas).
  template <typename T>
  bool operator()(const T* alphas, T* silhouette_error) const {
    typedef Eigen::Matrix<T, 4, 1> Vector4;
    typedef Eigen::Matrix<T, 3, 1> Vector3;
    typedef Eigen::Matrix<T, 3, 3> Matrix3;
    typedef Eigen::Matrix<T, 4, 4> Matrix4;
    std::vector<Vector4> updated_source_points;
    std::vector<Vector3> updated_vertices;
    std::vector<Matrix4> chain_G;

    // Compute the transformations based on the joint angles (the alphas)
    // and the kinematic chain.
    std::vector<T> alphas_vec(alphas, alphas + num_alphas_);
    kinematic_chain_.UpdateKinematicChain(alphas_vec, &updated_source_points,
                                          &chain_G);

    // Update the vertex positions of the mesh using the chain.
    mesh_->UpdateVertices(chain_G, &updated_vertices);

    Matrix4 rigidbody_xform = ExtractRigidBodyXform(alphas + num_alphas_);

    // Apply the rigid body transform before projection.
    Eigen::Matrix<T, 3, 4> projection =
        projection_matrix_.cast<T>() * rigidbody_xform;
    *silhouette_error = ComputeMeshSilhouetteDistance(silhouette_, projection,
                                                      updated_vertices);

    return true;
  }

 private:
  // The kinematic chain of the articulated body.
  const KinematicChain kinematic_chain_;

  // Rigged mesh of the animal. Changes in the chain will change the mesh.
  RiggedMesh* mesh_;

  // Envelope calculated from input mask.
  std::vector<Eigen::Vector3d> silhouette_;

  // The camera projection matrix.
  const Eigen::Matrix<double, 3, 4> projection_matrix_;

  // The number of joint angles to optimize.
  const int num_alphas_;

  // The rigid body rotation parameters.
  const Eigen::Vector3d rotation_;

  // The rigid body translation parameters.
  const Eigen::Vector3d translation_;
};
// Similar to ImageSilhouetteCostFunction, but only optimizes the rotation and
// translation.
class ImageSilhouetteRigidBodyCostFunction {
 public:
  ImageSilhouetteRigidBodyCostFunction(
      const std::vector<Eigen::Vector3d>& silhouette,
      const Eigen::Matrix<double, 3, 4>& projection_matrix, RiggedMesh* mesh)
      : silhouette_(silhouette),
        projection_matrix_(projection_matrix),
        mesh_(mesh) {}

  // Evaluates the image error from joint angles (or the alphas).
  template <typename T>
  bool operator()(const T* alphas, T* silhouette_error) const {
    typedef Eigen::Matrix<T, 3, 1> Vector3;
    typedef Eigen::Matrix<T, 4, 4> Matrix4;
    std::vector<Vector3> updated_vertices;

    // Update the vertex positions of the mesh using the chain.
    mesh_->CopyVertices(&updated_vertices);

    Matrix4 rigidbody_xform = ExtractRigidBodyXform(alphas);
    // Apply the rigid body transform before projection.
    Eigen::Matrix<T, 3, 4> projection =
        projection_matrix_.cast<T>() * rigidbody_xform;
    *silhouette_error = ComputeMeshSilhouetteDistance(silhouette_, projection,
                                                      updated_vertices);

    return true;
  }

 private:
  // Envelope calculated from input mask.
  std::vector<Eigen::Vector3d> silhouette_;

  // The camera projection matrix.
  const Eigen::Matrix<double, 3, 4> projection_matrix_;

  // Rigged mesh of the animal. Changes in the chain will change the mesh.
  RiggedMesh* mesh_;
};

// The cost function to optimize the rigid body transformation parameters:
// rotation and translation for fitting angles from ground truth joint
// locations. The cost function takes, the 3D keypoint locations and the
// target 3D keypoint locations as input. It computes the residual as the L1
// norm of the difference between the 3D keypoint locations and the target 3D
// keypoint locations. The residual will be used by CERES to do auto diff to
// compute the gradients.
class RigidTrans3DCostFunction {
 public:
  // Initializes the cost function with the 3D keypoint locations, the target
  // 2D keypoint locations and the camera projection matrix.
  RigidTrans3DCostFunction(const std::vector<Eigen::Vector3d>& source_points,
                           const std::vector<Eigen::Vector3d>& target_points,
                           const std::vector<double>& weights)
      : source_points_(source_points),
        target_points_(target_points),
        weights_(weights) {}

  // Evaluates the residual based on the rotation/translation parameters, 3D
  // keypoint locations, and target 3D keypoint locations.
  template <typename T>
  bool operator()(const T* rotation, const T* translation,
                  T* residual_error) const {
    typedef Eigen::Matrix<T, 3, 1> Vector3;
    typedef Eigen::Map<const Vector3> Vector3Ref;

    Vector3Ref rotation_mat(rotation);
    Vector3Ref translation_mat(translation);
    for (size_t i = 0; i < source_points_.size(); ++i) {
      const Vector3 source_point = source_points_[i].template cast<T>();
      Vector3 rotated_point;
      ceres::AngleAxisRotatePoint(rotation_mat.data(), source_point.data(),
                                  rotated_point.data());
      Vector3 translated_point;
      translated_point = rotated_point + translation_mat;
      residual_error[i * 3] =
          weights_[i] * (translated_point[0] - target_points_[i][0]);
      residual_error[i * 3 + 1] =
          weights_[i] * (translated_point[1] - target_points_[i][1]);
      residual_error[i * 3 + 2] =
          weights_[i] * (translated_point[2] - target_points_[i][2]);
    }
    return true;
  }

 private:
  // List of the source 3D keypoint locations.
  const std::vector<Eigen::Vector3d> source_points_;

  // List of the target 2D keypoint locations.
  const std::vector<Eigen::Vector3d> target_points_;

  // The weights for each joints.
  const std::vector<double> weights_;
};

// The cost function to optimizes for the angles of a 3D pose with the same
// kinematic chain. The goal is to find the angles (alphas) used for a ground
// truth example. Comparing the fits using the oracle 3D pose or another
// representation allows us to estimate angular error.
class Pose3DToAngleCostFunction {
 public:
  // Initializes the cost function with the kinematic chain, the target
  // 2D keypoint locations, rigid transformation parameters and
  // the camera projection matrix.
  Pose3DToAngleCostFunction(const KinematicChain& kinematic_chain,
                            const std::vector<Eigen::Vector3d>& target_points,
                            const std::vector<double>& weights)
      : kinematic_chain_(kinematic_chain),
        target_points_(target_points),
        weights_(weights),
        num_alphas_(3 * kinematic_chain.GetJoints().size()) {}

  // Evaluates the Euclidean residual based on the joint angles (or the alphas),
  // kinematic chain, target 3D keypoint locations.
  template <typename T>
  bool operator()(const T* rotation, const T* translation, const T* alphas,
                  T* euclidean_error) const {
    typedef Eigen::Matrix<T, 4, 1> Vector4;
    typedef Eigen::Matrix<T, 3, 1> Vector3;
    typedef Eigen::Matrix<T, 4, 4> Matrix4;
    typedef Eigen::Map<const Vector3> Vector3Ref;
    std::vector<Vector4> updated_source_points;
    std::vector<Matrix4> chain_G;
    Vector3Ref rotation_mat(rotation);
    Vector3Ref translation_mat(translation);

    // Compute the 3D joint locations based on the joint angles (the alphas)
    // and the kinematic chain.
    std::vector<T> alphas_vec(alphas, alphas + num_alphas_);
    kinematic_chain_.UpdateKinematicChain(alphas_vec, &updated_source_points,
                                          &chain_G);

    // Transform the updated 3D joint locations with rotation and translation.
    // Reproject the transformed 3D joint locations to compute the reprojection
    // error.
    for (size_t i = 0; i < updated_source_points.size(); i++) {
      const Vector3 updated_source_point =
          updated_source_points[i].template head<3>();
      Vector3 rotated_point;
      ceres::AngleAxisRotatePoint(rotation_mat.data(),
                                  updated_source_point.data(),
                                  rotated_point.data());
      Vector3 translated_point = rotated_point + translation_mat;
      double L2_ANGLE_LOSS = 1e-4;
      euclidean_error[i * 3] =
          weights_[i] * pow(translated_point[0] - target_points_[i][0], 2) +
          L2_ANGLE_LOSS * pow(alphas[i * 3], 2);
      euclidean_error[i * 3 + 1] =
          weights_[i] * pow(translated_point[1] - target_points_[i][1], 2) +
          L2_ANGLE_LOSS * pow(alphas[i * 3 + 1], 2);
      euclidean_error[i * 3 + 2] =
          weights_[i] * pow(translated_point[2] - target_points_[i][2], 2) +
          L2_ANGLE_LOSS * pow(alphas[i * 3 + 2], 2);
    }
    return true;
  }

 private:
  // The kinematic chain of the articulated body.
  const KinematicChain kinematic_chain_;

  // The list of target 3D point locations.
  const std::vector<Eigen::Vector3d> target_points_;

  // The weights for each joints.
  const std::vector<double> weights_;

  // The number of joint angles to optimize.
  const int num_alphas_;
};

// The cost function to optimize the joint angles of an articulated body in a
// basis space of alpha angles. The cost function takes, the kinematic chain
// (with the basis space), target 2D keypoints, rigid transformations and the
// projection camera matrices as input. And it computes the residual as the L1
// norm of the difference of the projected 3D keypoint locations and the target
// 2D keypoint locations. The residual will be used by CERES to do auto diff to
// compute the gradients.
class JointAnglePcaCostFunction {
 public:
  // Initializes the cost function with the kinematic chain, the target
  // 2D keypoint locations, rigid transformation parameters and
  // the camera projection matrix.
  JointAnglePcaCostFunction(
      const KinematicChain& kinematic_chain,
      const std::vector<Eigen::Vector2d>& target_points,
      const Eigen::Matrix<double, 3, 4>& projection_matrix,
      const std::vector<double>& weights)
      : kinematic_chain_(kinematic_chain),
        target_points_(target_points),
        projection_matrix_(projection_matrix),
        weights_(weights) {}

  // Evaluates the residual (or the reprojection_error) based on the joint
  // angles (or the alphas), kinematic chain, target 2D keypoint locations
  // and the camera projection matrix.
  template <typename T>
  bool operator()(const T* rotation, const T* translation, const T* loadings,
                  T* reprojection_error) const {
    typedef Eigen::Matrix<T, 4, 1> Vector4;
    typedef Eigen::Matrix<T, 3, 1> Vector3;
    typedef Eigen::Matrix<T, 4, 4> Matrix4;
    // Use unique pointer to avoid stack overflow.
    auto updated_source_points = std::make_unique<std::vector<Vector4>>();
    auto chain_G = std::make_unique<std::vector<Matrix4>>();
    typedef Eigen::Map<const Vector3> Vector3Ref;
    Vector3Ref rotation_mat(rotation);
    Vector3Ref translation_mat(translation);

    //                         Resizing is cheap, but cannot be compiled away.
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> basis;
    const auto& kc_basis = kinematic_chain_.GetAlphaBasis();
    basis.resize(kc_basis.rows(), kc_basis.cols());
    for (int i = 0; i < kc_basis.rows(); ++i) {
      for (int j = 0; j < kc_basis.cols(); ++j) {
        basis(i, j) = static_cast<T>(kc_basis(i, j));
      }
    }
    Eigen::Matrix<T, 1, Eigen::Dynamic> basis_mean;
    const auto& kc_basis_mean = kinematic_chain_.GetAlphaBasisMean();
    basis_mean.resize(kc_basis_mean.rows(), kc_basis_mean.cols());
    for (int j = 0; j < kc_basis_mean.cols(); ++j) {
      basis_mean(0, j) = static_cast<T>(kc_basis_mean(0, j));
    }
    Eigen::Matrix<T, 1, Eigen::Dynamic> loadings_mat;
    loadings_mat.resize(1, basis.rows());
    for (int i = 0; i < basis.rows(); ++i) {
      loadings_mat(0, i) = loadings[i];
    }
    Eigen::Matrix<T, Eigen::Dynamic, 1> alphas_mat =
        loadings_mat * basis + basis_mean;

    // Compute the 3D joint locations based on the joint angles (the alphas)
    // and the kinematic chain.
    std::vector<T> alphas_vec(alphas_mat.begin(), alphas_mat.end());
    kinematic_chain_.UpdateKinematicChain(
        alphas_vec, updated_source_points.get(), chain_G.get());

    // Transform the updated 3D joint locations with rotation and translation.
    // Reproject the transformed 3D joint locations to compute the reprojection
    // error.
    for (size_t i = 0; i < updated_source_points->size(); i++) {
      const Vector3 updated_source_point =
          (*updated_source_points)[i].template head<3>();
      Vector3 rotated_point;
      ceres::AngleAxisRotatePoint(rotation_mat.data(),
                                  updated_source_point.data(),
                                  rotated_point.data());
      Vector3 translated_point = rotated_point + translation_mat;
      Vector4 augmented_point;
      augmented_point.block(0, 0, 3, 1) << translated_point;
      augmented_point(3, 0) = static_cast<T>(1.);
      const Eigen::Matrix<T, 2, 1> reprojected_point =
          (projection_matrix_.cast<T>() * augmented_point).hnormalized();
      reprojection_error[i * 2] =
          (reprojected_point[0] - target_points_[i][0]) * weights_[i];
      reprojection_error[i * 2 + 1] =
          (reprojected_point[1] - target_points_[i][1]) * weights_[i];
    }
    return true;
  }

 private:
  // The kinematic chain of the articulated body.
  const KinematicChain kinematic_chain_;

  // The list of target 2D point locations.
  const std::vector<Eigen::Vector2d> target_points_;

  // The camera projection matrix.
  const Eigen::Matrix<double, 3, 4> projection_matrix_;

  // The weights for each joints.
  const std::vector<double> weights_;
};
}  // namespace mouse_pose

#endif  // MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_COST_FUNCTIONS_H_
