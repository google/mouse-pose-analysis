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

#include "mouse_pose_analysis/pose_3d/rigged_mesh_utils.h"

#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "glog/logging.h"
#include "mouse_pose_analysis/pose_3d/rigged_mesh.h"

namespace mouse_pose {

absl::Status ReadBoneWeights(
    std::string weight_filename,
    std::vector<VertexBoneWeights> *mesh_bone_weights) {
  mesh_bone_weights->clear();
  std::ifstream weight_file(weight_filename);
  std::string line;
  while (std::getline(weight_file, line)) {
    if (line[0] == '#') continue;

    VertexBoneWeights bws;
    double weight_sum = 0;
    std::vector<std::string> tokens = absl::StrSplit(line, ',');
    // Ignore the first item, which is the vertex index.
    for (int i = 1; i < tokens.size(); ++i) {
      double w;
      absl::SimpleAtod(tokens[i], &w);
      weight_sum += w;
      bws.push_back(w);
    }
    CHECK(std::abs(weight_sum - 1) <= std::numeric_limits<float>::epsilon())
        << "Weights don't sum to 1.";

    mesh_bone_weights->push_back(bws);
  }

  return absl::OkStatus();
}

absl::Status WriteObjFile(std::string filename, const RiggedMesh &mesh) {
  std::ofstream obj_file(filename);
  for (const auto &vertex : mesh.GetVertices()) {
    std::string v =
        absl::StrFormat("v %f %f %f\n", vertex[0], vertex[1], vertex[2]);
    obj_file << v;
  }

  for (const Face &face : mesh.GetFaces()) {
    std::string f =
        absl::StrFormat("f %d %d %d\n", face[0] + 1, face[1] + 1, face[2] + 1);
    obj_file << f;
  }
  return absl::OkStatus();
}

absl::Status ReadObj(std::string filename, Eigen::MatrixXd &V,
                     Eigen::MatrixXi &F) {
  std::vector<std::vector<double>> vertices;
  std::vector<std::vector<int>> faces;
  std::ifstream obj_file(filename);
  std::string line;
  while (std::getline(obj_file, line)) {
    char leader = line[0];
    if (leader == 'v') {
      std::vector<double> coords;
      std::vector<std::string> tokens =
          absl::StrSplit(line.substr(1), ' ', absl::SkipEmpty());
      for (auto t : tokens) {
        double c;
        CHECK(absl::SimpleAtod(t, &c)) << "Invalid OBJ vertex line.";
        coords.push_back(c);
      }
      CHECK_EQ(coords.size(), 3) << "Only support 3-D coordinates.";
      vertices.push_back(coords);
    } else if (leader == 'f') {
      std::vector<int> indices;
      std::vector<std::string> tokens =
          absl::StrSplit(line.substr(1), ' ', absl::SkipEmpty());
      for (auto t : tokens) {
        int i;
        CHECK(absl::SimpleAtoi(t, &i)) << "Invalid OBJ face line.";
        indices.push_back(i);
      }
      CHECK_EQ(indices.size(), 3) << "Only support triangles.";
      faces.push_back(indices);
    } else if (leader == '#' || leader == 's' || leader == 'g') {
      // Noop for comments and a few unused options that may appear.
    } else {
      return absl::InvalidArgumentError("Unsupported OBJ format.");
    }
  }

  // Only support 3-vectors for now.
  V.resize(vertices.size(), 3);
  for (int i = 0; i < vertices.size(); ++i) {
    V(i, 0) = vertices[i][0];
    V(i, 1) = vertices[i][1];
    V(i, 2) = vertices[i][2];
  }
  F.resize(faces.size(), 3);
  for (int i = 0; i < faces.size(); ++i) {
    F(i, 0) = faces[i][0] - 1;
    F(i, 1) = faces[i][1] - 1;
    F(i, 2) = faces[i][2] - 1;
  }
  return absl::OkStatus();
}

std::unique_ptr<RiggedMesh> CreateRiggedMeshFromFiles(
    std::string obj_filename, std::string weight_filename) {
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  ReadObj(obj_filename, V, F);

  std::vector<VertexBoneWeights> weights;
  ReadBoneWeights(weight_filename, &weights).IgnoreError();

  return std::make_unique<RiggedMesh>(V, F, weights);
}
}  // namespace mouse_pose
