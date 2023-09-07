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

#ifndef MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_POSE_RECONSTRUCTION_H_
#define MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_POSE_RECONSTRUCTION_H_

#include <vector>

#include "absl/status/status.h"
#include "mouse_pose_analysis/pose_3d/pose_optimizer.h"
#include "mouse_pose_analysis/pose_3d/pose_reconstruction.pb.h"

namespace mouse_pose {

void SetUpOptimizer(
    const mouse_pose::optical_mouse::InputParameters &input_parameters,
    PoseOptimizer *pose_optimizer);

absl::Status ReconstructPose(
    const mouse_pose::optical_mouse::InputParameters &input_parameters,
    PoseOptimizer *pose_optimizer, const std::vector<double> &keypoint_weight,
    std::vector<double> *trans_params, std::vector<double> *rotate_params,
    std::vector<double> *alphas, std::vector<double> *shape_basis_weights);
}  // namespace mouse_pose

#endif  // MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_POSE_RECONSTRUCTION_H_
