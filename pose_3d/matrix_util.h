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

#ifndef MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_MATRIX_UTIL_H_
#define MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_MATRIX_UTIL_H_

#include <cmath>

#include "ceres/rotation.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/MatrixFunctions"

namespace mouse_pose {

const int kOrderOfApproximation = 10;

template <typename T, int N, int M>
Eigen::Matrix<T, N, M> MatrixPow(Eigen::Matrix<T, N, M> mat, int n) {
  CHECK_GE(n, 0);
  if (n == 0) {
    return Eigen::Matrix<T, N, M>::Identity();
  } else {
    return mat * MatrixPow(mat, n - 1);
  }
}

// Computes the exponential coordinates basis of a twist from the rotation axis
// and the rotation center. The 4 x 4 basis will later by multiplied by a
// rotation angle to form a rotation matrix. Details refer to:
// https://danieltakeshi.github.io/2018/01/11/twists-and-exponential-coordinates/
template <typename T>
Eigen::Matrix<T, 4, 4> SkewAxis(Eigen::Matrix<T, 3, 1> axis,
                                Eigen::Matrix<T, 3, 1> center) {
  Eigen::Matrix<T, 3, 1> translation;
  translation = -axis.cross(center);
  Eigen::Matrix<T, 4, 4> twist;
  auto zero = static_cast<T>(0);
  twist << zero, -axis(2), axis(1), translation(0), axis(2), zero, -axis(0),
      translation(1), -axis(1), axis(0), zero, translation(2), zero, zero, zero,
      zero;
  return twist;
}

// Computes the exponential map based on the rotation angle and the
// rotation matrix U which is computed from the rotation axis using function
// like SkewAxis() above.
// Q(\theta, u) = e^{\thethaU} = I + sU + (1 - c) U^2 in which
// s = \sum_{k=0}((-1)^k\theta^{2k+1})/((2k+1)!)
// c = \sum_{k=0}((-1)^k\theta^{2k})/((2k)!)
template <typename T>
Eigen::Matrix<T, 4, 4> ExponentialMap(Eigen::Matrix<T, 4, 4> u_mat, T theta) {
  Eigen::Matrix<T, 4, 4> result(Eigen::Matrix<T, 4, 4>::Identity());
  T s_value = theta;
  T c_value = static_cast<T>(1);
  T factor = static_cast<T>(6);
  for (int i = 1; i < kOrderOfApproximation; i++) {
    if (i == 1) {
      s_value += pow(static_cast<T>(-1), i) *
                 pow(static_cast<T>(theta), 2 * i + 1) /
                 (static_cast<T>(2 * 3));
      c_value += pow(static_cast<T>(-1), i) *
                 pow(static_cast<T>(theta), 2 * i) / (static_cast<T>(2));
    } else {
      factor *= static_cast<T>(2 * i);
      c_value += pow(static_cast<T>(-1), i) *
                 pow(static_cast<T>(theta), 2 * i) / factor;
      factor *= static_cast<T>(2 * i + 1);
      s_value += pow(static_cast<T>(-1), i) *
                 pow(static_cast<T>(theta), 2 * i + 1) / factor;
    }
  }
  result +=
      (s_value)*u_mat + (static_cast<T>(1) - c_value) * MatrixPow(u_mat, 2);
  return result;
}

// Projects a 3D point to 2D image.
// The projection matrix is 4x3 and the input point needs to homogenousized
// first and the resulting 3-vector needs to be de-homogenousized.
template <typename T>
Eigen::Vector<T, 2> Project3DPoint(
    const Eigen::Matrix<T, 3, 4> &projection_matrix,
    const Eigen::Vector<T, 3> &point) {
  Eigen::Vector<T, 4> augmented_point = point.homogeneous();
  auto projected = projection_matrix * augmented_point;
  return projected.hnormalized();
}

// Applies a 4x4 transformation to a 3-d vertex and returns a 3-d vertex.
template <typename T>
Eigen::Vector<T, 3> TransformVertex(const Eigen::Matrix<T, 4, 4> transformation,
                                    const Eigen::Vector<T, 3> vertex) {
  Eigen::Vector<T, 4> v = vertex.homogeneous();
  v = transformation * v;
  return v.hnormalized();
}

// Extracts the rigid body transform from 'params'.
// The first 3 elements of params are a rotation vector and the next 3 are
// translation vector.
template <typename T>
Eigen::Matrix<T, 4, 4> ExtractRigidBodyXform(const T *params) {
  typedef Eigen::Matrix<T, 3, 3> Matrix3;
  typedef Eigen::Matrix<T, 4, 4> Matrix4;
  Matrix4 rigidbody_xform = Matrix4::Identity();

  Matrix3 rotation_mat;
  ceres::AngleAxisToRotationMatrix(params, rotation_mat.data());

  rigidbody_xform.topLeftCorner(3, 3) = rotation_mat;
  rigidbody_xform(0, 3) = static_cast<T>(params[3]);
  rigidbody_xform(1, 3) = static_cast<T>(params[4]);
  rigidbody_xform(2, 3) = static_cast<T>(params[5]);

  return rigidbody_xform;
}
}  // namespace mouse_pose

#endif  // MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_MATRIX_UTIL_H_
