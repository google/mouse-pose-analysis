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

#include "mouse_pose_analysis/pose_3d/rigged_mesh.h"

#include <vector>

#include "ceres/rotation.h"
#include "third_party/eigen3/Eigen/Core"

namespace mouse_pose {

RiggedMesh::RiggedMesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                       const std::vector<VertexBoneWeights> &weights)
    : mesh_bone_weights_(weights) {
  CHECK_EQ(V.rows(), weights.size());

  // Copy the vertices.
  for (const auto v : V.rowwise()) vertices_.push_back(v);

  // Copy the faces.
  for (const Face f : F.rowwise()) faces_.push_back(f);
}

void RiggedMesh::DeformMesh(
    const std::vector<Eigen::Matrix4d> &chain_transformations) {
  for (int i = 0; i < vertices_.size(); ++i) {
    VertexBoneWeights ws = mesh_bone_weights_[i];

    Eigen::Matrix4d weighted_transform(Eigen::Matrix4d::Zero());
    CHECK_EQ(ws.size(), chain_transformations.size());
    for (int wi = 0; wi < ws.size(); ++wi) {
      weighted_transform += ws[wi] * chain_transformations[wi];
    }
    Eigen::Vector4d v = vertices_[i].homogeneous();
    v = weighted_transform * v;
    vertices_[i] = v.hnormalized();
  }
}

void RiggedMesh::TransformMesh(const Eigen::Vector3f &rotation,
                               const Eigen::Vector3f &translation) {
  for (int i = 0; i < vertices_.size(); i++) {
    Eigen::Vector3f rotated_vertex;
    Eigen::Vector3f vertex_f =
        Eigen::Map<const Eigen::Vector3d>(vertices_[i].data(), 3).cast<float>();
    ceres::AngleAxisRotatePoint(rotation.data(), vertex_f.data(),
                                rotated_vertex.data());
    Eigen::Vector3f translated_vertex;
    translated_vertex = rotated_vertex + translation;
    vertices_[i] =
        Eigen::Map<const Eigen::Vector3f>(translated_vertex.data(), 3)
            .cast<double>();
  }
}

}  // namespace mouse_pose
