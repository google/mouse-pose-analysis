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

#ifndef MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_RIGGED_MESH_UTILS_H_
#define MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_RIGGED_MESH_UTILS_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mouse_pose_analysis/pose_3d/matrix_util.h"
#include "mouse_pose_analysis/pose_3d/rigged_mesh.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "third_party/eigen3/Eigen/Core"

namespace mouse_pose {
// Writes a RiggedMesh to a Wavefront OBJ file.
absl::Status WriteObjFile(std::string filename, const RiggedMesh &mesh);

// Creates a rigged mesh from an OBJ file and a weight file.
std::unique_ptr<RiggedMesh> CreateRiggedMeshFromFiles(
    std::string obj_filename, std::string weight_filename);

// Reads a CSV file, each of row of which contains weights for each vertex.
absl::Status ReadBoneWeights(std::string weight_filename,
                             std::vector<VertexBoneWeights> *mesh_bone_weights);

// Projects a RiggedMesh face (only triangles are supported for now) to image
// and fills inside with 255.
template <typename T>
void ProjectFaceToImage(const std::vector<Eigen::Vector<T, 3>> &vertices,
                        const Eigen::Matrix<T, 3, 4> &projection_matrix,
                        const Face &face, cv::Mat *image) {
  std::vector<cv::Point> image_points;
  for (int i = 0; i < 3; ++i) {
    Eigen::Vector<T, 3> vertex;
    vertex = vertices[face[i]];
    Eigen::Vector<T, 2> image_point = Project3DPoint(projection_matrix, vertex);
    image_points.push_back(cv::Point(static_cast<int>(image_point(0)),
                                     static_cast<int>(image_point(1))));
  }
  cv::fillConvexPoly(*image, image_points.data(), 3, cv::Scalar(255));
}

// Projects the mesh to an image of specified dimension.
// This is the most crude implementation.  A slightly faster algorithm is to
// check the normal of each face before projection.
template <typename T>
cv::Mat ProjectMeshToImage(const RiggedMesh &rigged_mesh,
                           const Eigen::Matrix<T, 3, 4> &projection_matrix,
                           int image_width, int image_height) {
  cv::Mat image(image_height, image_width, CV_8UC1, cv::Scalar(0));

  for (const Face &face : rigged_mesh.GetFaces()) {
    ProjectFaceToImage(rigged_mesh.GetVertices(), projection_matrix, face,
                       &image);
  }

  return image;
}

}  // namespace mouse_pose

#endif  // MOUSE_POSE_OPTICAL_MOUSE_POSE_3D_RIGGED_MESH_UTILS_H_
