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

#ifndef MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_RIGGED_MESH_H_
#define MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_RIGGED_MESH_H_

#include <vector>

#include "mouse_pose_analysis/pose_3d/matrix_util.h"
#include "third_party/eigen3/Eigen/Core"

namespace mouse_pose {

// Bone weight of each vertex in a skinned model.  The size of the vertex N
// tells the number of bones.  The weights should add up to 1.  One way to
// enforce this is to store only N-1 weights and the last weight is
// 1-sum(weight(1:N-1)).  For simplicity and performance, we make a tacit
// agreement with the user.
typedef std::vector<double> VertexBoneWeights;

// A face is a vector of indices, each of which indexes to a vertex.
// Face f;
// f(0)=3;
// vertices[f(0)] points to the 3rd vertex.
typedef Eigen::Vector3i Face;

// A stripped down implementation of a rigged mesh, which only has faces and
// vertices.  Normals and textures will be added if needed later.  Each vertex
// has a set of bone weights w_i^j, i=1...N.  When bones are transformed by T_i,
// vertex j will transformed by \sum w_i^j T_i.
class RiggedMesh {
 public:
  explicit RiggedMesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                      const std::vector<VertexBoneWeights> &weights);
  RiggedMesh() {}

  // No copy and move for now.
  RiggedMesh(const RiggedMesh &) = delete;
  RiggedMesh &operator=(const RiggedMesh &) = delete;

  // Copies mesh to a templated container. Equivalent to applying an identify
  // function.
  template <typename T>
  void CopyVertices(std::vector<Eigen::Vector<T, 3>> *updated_vertices) {
    updated_vertices->clear();
    for (const auto &v : vertices_) {
      Eigen::Vector<T, 3> vc = v.cast<T>();
      updated_vertices->push_back(vc);
    }
  }

  // Updates vertex positions when bones are transformed.
  template <typename T>
  void UpdateVertices(const std::vector<Eigen::Matrix<T, 4, 4>> &transforms,
                      std::vector<Eigen::Vector<T, 3>> *updated_vertices) {
    updated_vertices->clear();
    for (int i = 0; i < vertices_.size(); ++i) {
      VertexBoneWeights ws = mesh_bone_weights_[i];

      Eigen::Matrix<T, 4, 4> weighted_transform(Eigen::Matrix<T, 4, 4>::Zero());
      for (int wi = 0; wi < ws.size(); ++wi) {
        weighted_transform += static_cast<T>(ws[wi]) * transforms[wi];
      }

      Eigen::Vector<T, 3> v = vertices_[i].cast<T>();
      v = TransformVertex(weighted_transform, v);
      updated_vertices->push_back(v);
    }
  }

  void DeformMesh(const std::vector<Eigen::Matrix4d> &chain_transformations);
  void TransformMesh(const Eigen::Vector3f &rotation,
                     const Eigen::Vector3f &translation);
  void SetVertices(const std::vector<Eigen::Vector3d> &vertices) {
    vertices_ = vertices;
  }
  void SetFaces(const std::vector<Face> &faces) { faces_ = faces; }
  // Returns the mesh.
  const std::vector<Eigen::Vector3d> GetVertices() const { return vertices_; }
  const std::vector<Face> GetFaces() const { return faces_; }

 private:
  std::vector<Eigen::Vector3d> vertices_;
  std::vector<Face> faces_;
  std::vector<VertexBoneWeights> mesh_bone_weights_;
};

}  // namespace mouse_pose

#endif  // MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_RIGGED_MESH_H_
