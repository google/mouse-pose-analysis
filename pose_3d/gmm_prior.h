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

#ifndef MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_GMM_PRIOR_H_
#define MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_GMM_PRIOR_H_

#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>

#include "ceres/ceres.h"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "mouse_pose_analysis/pose_3d/gmm.pb.h"
#include "third_party/eigen3/Eigen/Core"

namespace mouse_pose {

// Code for Gaussian Mixture Model likelihood computation.
// The goal is that the model can be used as a prior for optimization in
// Ceres, which requires a templated definition and using Ceres versions of
// math routines.
class GaussianMixtureModel {
 public:
  // Initialize the model from a text protobuf file.
  void InitializeFromFile(std::string filepath) {
    std::ifstream proto_file(filepath);
    CHECK(proto_file) << "Cannot open the gmm proto file." << filepath;

    std::stringstream buf;
    buf << proto_file.rdbuf();

    GaussianMixtureModelProto gmmp;
    google::protobuf::TextFormat::ParseFromString(buf.str(), &gmmp);
    InitializeFromProto(gmmp);
  }

  void CheckIsScaledAndRotated() {
    // If correctly constructed, the first three values should be zeros, then
    // 1, 0, 0. This means that the first axis is scaled and rotated as
    // expected.
    for (int i = 0; i < means_.rows(); ++i) {
      CHECK_LE(fabs(means_(i, 0)), 1e-5);
      CHECK_LE(fabs(means_(i, 1)), 1e-5);
      CHECK_LE(fabs(means_(i, 2)), 1e-5);
      CHECK_LE(fabs(means_(i, 3) - 1.), 1e-5);
      CHECK_LE(fabs(means_(i, 4)), 1e-5);
      CHECK_LE(fabs(means_(i, 5)), 1e-5);
    }
  }

  // Initialize the model from a protobuf. This verifies that the dimensions
  // are consistent. The order of each array should be row-major.
  void InitializeFromProto(const GaussianMixtureModelProto& proto) {
    num_components_ = proto.number_of_components();
    num_dimensions_ = proto.number_of_dimensions();
    mixing_proportions_.resize(num_components_, 1);
    means_.resize(num_components_, num_dimensions_);
    CHECK_EQ(proto.mixing_proportions_size(), num_components_);
    CHECK_EQ(proto.means_size(), num_components_ * num_dimensions_);
    CHECK_EQ(proto.precisions_size(),
             num_components_ * num_dimensions_ * num_dimensions_);
    precisions_.resize(num_components_);
    // Doing explicit copies so that the types are converted to Ceres:;jet.
    for (int component = 0; component < num_components_; ++component) {
      precisions_[component].resize(num_dimensions_, num_dimensions_);
      mixing_proportions_(component, 0) = proto.mixing_proportions(component);
      for (int row = 0; row < num_dimensions_; ++row) {
        means_(component, row) = proto.means(component * num_dimensions_ + row);
        for (int column = 0; column < num_dimensions_; ++column) {
          precisions_[component](row, column) =
              proto.precisions(component * num_dimensions_ * num_dimensions_ +
                               row * num_dimensions_ + column);
        }
      }
    }
    LOG(INFO) << "mixture model means:\n" << means_;
  }

  // Calculates the log likelihood for the points under the model.
  template <typename T>
  T LogLikelihood(const Eigen::Matrix<T, Eigen::Dynamic, 1>& points) const {
    CHECK_NE(num_components_, -1) << "Model is not initialized.";
    CHECK_EQ(num_dimensions_, points.rows())
        << "The dimensions do not match the data.";
    // Compute individual weighted log likelihoods.
    std::vector<T> log_likelihoods(num_components_);
    T k_log_2_pi = static_cast<T>(num_dimensions_) * log(2. * M_PI);
    for (int i = 0; i < num_components_; ++i) {
      Eigen::Matrix<T, Eigen::Dynamic, 1> diffs =
          points - means_.row(i).transpose().cast<T>();
      // If I try to do these all as one line, the compiler breaks for reasons
      // I can't understand due to Ceres type conversions and Eigen.
      T sqrt_factor = static_cast<T>(-0.5);
      T norm_constant = static_cast<T>(log(1. / precisions_[i].determinant()));
      T distance = diffs.transpose() * precisions_[i].cast<T>() * diffs;
      T mixing_proportion = static_cast<T>(log(mixing_proportions_[i]));
      T core_factor = norm_constant + distance + k_log_2_pi;
      log_likelihoods[i] = sqrt_factor * core_factor + mixing_proportion;

      if (ceres::IsInfinite(log_likelihoods[i])) {
        LOG(WARNING) << "SUPPRESSING NaN LIKELIHOOD";
        log_likelihoods[i] = static_cast<T>(-1e6);
      }
      CHECK(!ceres::IsInfinite(log_likelihoods[i]))
          << "The pose has an infinite or nan likelihood. "
          << "Abort rather than continuing.";
    }
    // Compute log sum exp in a stable fashion.
    auto max_likelihood =
        std::max_element(log_likelihoods.begin(), log_likelihoods.end());
    T sum_exp = static_cast<T>(0.);
    for (int i = 0; i < num_components_; ++i) {
      sum_exp += exp(log_likelihoods[i] - *max_likelihood);
    }
    T log_sum_exp = *max_likelihood + log(sum_exp);
    return log_sum_exp;
  }

  // Aligns the first and second points to be offset to unit length on the X
  // axis. Translates the first point to the origin. Other points are
  // transformed to maintain relative positions. This matches the preprocessing
  // in create_pose_mixture_model.py.
  template <typename T>
  static void AlignPointsAndFlatten(
      const std::vector<Eigen::Matrix<T, 3, 1>>& points,
      Eigen::Matrix<T, Eigen::Dynamic, 1>* flattened) {
    CHECK_GE(points.size(), 16) << "There must be at least 16 points.";
    flattened->resize(3 * points.size(), 1);
    auto root = points[0];
    auto first = points[1];
    Eigen::Matrix<T, 3, 1> core_axis = first - root;
    // Templating is causing errors resolving square, sqrt, and sum in Eigen, so
    // computed manually here. We need to rely on ceres implementations instead.
    T scale = ceres::sqrt(core_axis(0, 0) * core_axis(0, 0) +
                          core_axis(1, 0) * core_axis(1, 0) +
                          core_axis(2, 0) * core_axis(2, 0));
    Eigen::Matrix<T, 3, 1> scaled_axis = core_axis / scale;
    Eigen::Matrix<T, 3, 1> unit_x = {static_cast<T>(1.), static_cast<T>(0.),
                                     static_cast<T>(0.)};
    // Figure out rotation.
    auto cross_value = scaled_axis.cross(unit_x);
    auto dot_value = scaled_axis.dot(unit_x);
    bool must_rotate_around_y = false;
    Eigen::Matrix<T, 3, 3> y_rotation;
    y_rotation << static_cast<T>(-1.), static_cast<T>(0.), static_cast<T>(0.),
        static_cast<T>(0.), static_cast<T>(1.), static_cast<T>(0.),
        static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(-1.);
    if (dot_value == -1.) {
      must_rotate_around_y = true;
      scaled_axis = y_rotation * scaled_axis;
      dot_value = scaled_axis.dot(unit_x);
      cross_value = scaled_axis.cross(unit_x);
    }
    CHECK_NE(dot_value, -1.) << "We do not yet support reversed axes";
    Eigen::Matrix<T, 3, 3> skew_matrix;
    skew_matrix(0, 0) = static_cast<T>(0.);
    skew_matrix(0, 1) = -cross_value[2];
    skew_matrix(0, 2) = cross_value[1];
    skew_matrix(1, 0) = cross_value[2];
    skew_matrix(1, 1) = static_cast<T>(0.);
    skew_matrix(1, 2) = -cross_value[0];
    skew_matrix(2, 0) = -cross_value[1];
    skew_matrix(2, 1) = cross_value[0];
    skew_matrix(2, 2) = static_cast<T>(0.);
    Eigen::Matrix<T, 3, 3> rotation_matrix =
        skew_matrix.Identity() + skew_matrix +
        (skew_matrix * skew_matrix) / (1. + dot_value);
    if (must_rotate_around_y) {
      rotation_matrix *= y_rotation;
    }

    auto right_shoulder = points[15];
    auto left_shoulder = points[12];
    Eigen::Matrix<T, 3, 1> shoulder_vector =
        rotation_matrix * (right_shoulder - left_shoulder);
    shoulder_vector(0, 0) = static_cast<T>(0.);  // Remove X projection.
    // Templating is causing errors resolving square, sqrt, and sum in Eigen, so
    // computed manually here. We need to rely on ceres implementations instead.
    T shoulder_scale =
        ceres::sqrt(shoulder_vector(0, 0) * shoulder_vector(0, 0) +
                    shoulder_vector(1, 0) * shoulder_vector(1, 0) +
                    shoulder_vector(2, 0) * shoulder_vector(2, 0));
    Eigen::Matrix<T, 3, 1> scaled_shoulder = shoulder_vector / shoulder_scale;
    Eigen::Matrix<T, 3, 1> unit_y = {static_cast<T>(0.), static_cast<T>(1.),
                                     static_cast<T>(0.)};
    // Figure out rotation.
    auto shoulder_cross_value = scaled_shoulder.cross(unit_y);
    auto shoulder_dot_value = scaled_shoulder.dot(unit_y);
    bool must_rotate_around_x = false;
    Eigen::Matrix<T, 3, 3> x_rotation;
    x_rotation << static_cast<T>(1.), static_cast<T>(0.), static_cast<T>(0.),
        static_cast<T>(0.), static_cast<T>(-1.), static_cast<T>(0.),
        static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(-1.);
    if (shoulder_dot_value == -1.) {
      must_rotate_around_x = true;
      scaled_shoulder = x_rotation * scaled_shoulder;
      shoulder_dot_value = scaled_shoulder.dot(unit_y);
      shoulder_cross_value = scaled_shoulder.cross(unit_y);
    }
    Eigen::Matrix<T, 3, 3> shoulder_skew_matrix;
    shoulder_skew_matrix(0, 0) = static_cast<T>(0.);
    shoulder_skew_matrix(0, 1) = -shoulder_cross_value[2];
    shoulder_skew_matrix(0, 2) = shoulder_cross_value[1];
    shoulder_skew_matrix(1, 0) = shoulder_cross_value[2];
    shoulder_skew_matrix(1, 1) = static_cast<T>(0.);
    shoulder_skew_matrix(1, 2) = -shoulder_cross_value[0];
    shoulder_skew_matrix(2, 0) = -shoulder_cross_value[1];
    shoulder_skew_matrix(2, 1) = shoulder_cross_value[0];
    shoulder_skew_matrix(2, 2) = static_cast<T>(0.);
    Eigen::Matrix<T, 3, 3> shoulder_rotation_matrix =
        shoulder_skew_matrix.Identity() + shoulder_skew_matrix +
        (shoulder_skew_matrix * shoulder_skew_matrix) /
            (1. + shoulder_dot_value);
    if (must_rotate_around_x) {
      shoulder_rotation_matrix *= x_rotation;
    }

    rotation_matrix = shoulder_rotation_matrix * rotation_matrix;
    for (int i = 0; i < points.size(); ++i) {
      // subtract root then apply rotation
      Eigen::Matrix<T, 3, 1> transformed_point =
          rotation_matrix * ((points[i] - root) / scale);
      (*flattened)(i * 3 + 0, 0) = transformed_point(0, 0);
      (*flattened)(i * 3 + 1, 0) = transformed_point(1, 0);
      (*flattened)(i * 3 + 2, 0) = transformed_point(2, 0);
    }
  }

 private:
  int num_components_ = -1;
  int num_dimensions_ = -1;
  Eigen::Matrix<double, Eigen::Dynamic, 1> mixing_proportions_;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> means_;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
      precisions_;
};
}  // namespace mouse_pose

#endif  // MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_GMM_PRIOR_H_
