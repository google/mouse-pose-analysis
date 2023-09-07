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

#ifndef MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_POSE_OPTIMIZER_UTILS_H_
#define MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_POSE_OPTIMIZER_UTILS_H_

#include <vector>

#include "absl/status/status.h"
#include "opencv2/core/core.hpp"
#include "mouse_pose_analysis/pose_3d/kinematic_chain.h"
#include "third_party/eigen3/Eigen/Core"

namespace mouse_pose {
// Loads the skeleton configuration file to construct the kinematic chain.
absl::Status LoadAndConstructKinematicChain(std::string chain_config_filename,
                                            KinematicChain* chain);

// Loads the target 2D points from file to target_points_.
absl::Status LoadTargetPointsFromFile(
    std::string target_points_filename,
    std::vector<Eigen::Vector2d>* target_2d_points);

// Loads the camera projection matrix from file.  The file is in the format
// specified in numpy savetxt().
absl::Status LoadProjectionMatFromFile(
    std::string projection_mat_filename,
    Eigen::Matrix<double, 3, 4>* projection_matrix);

}  // namespace mouse_pose

#endif  // MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_POSE_OPTIMIZER_UTILS_H_
