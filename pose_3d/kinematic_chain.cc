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

#include "mouse_pose_analysis/pose_3d/kinematic_chain.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_split.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/MatrixFunctions"

namespace mouse_pose {

KinematicChain::KinematicChain(std::string config_filename) {
  ReadSkeletonConfig(config_filename).IgnoreError();
}

absl::Status KinematicChain::ReadSkeletonConfig(std::string config_filename) {
  std::ifstream config_file(config_filename);
  std::string line;
  while (std::getline(config_file, line)) {
    if (line[0] == '#') continue;

    std::vector<std::string> tokens = absl::StrSplit(line, ',');
    CHECK(tokens.size() == 6) << "Format error.";
    joint_names_.push_back(tokens[0]);
    Eigen::Vector2i bone;
    CHECK(absl::SimpleAtoi(tokens[1], &bone[0]));
    CHECK(absl::SimpleAtoi(tokens[2], &bone[1]));
    bones_.push_back(bone);
    Eigen::Vector3d joint;
    CHECK(absl::SimpleAtod(tokens[3], &joint[0]));
    CHECK(absl::SimpleAtod(tokens[4], &joint[1]));
    CHECK(absl::SimpleAtod(tokens[5], &joint[2]));
    joints_.push_back(joint);
  }
  root_joint_idx_ = 0;
  return absl::OkStatus();
}

void KinematicChain::ConstructSkeletonRecursive(
    int root_joint_idx, std::vector<int>* joint_parent) {
  // Pre-order, breadth first search through the joint tree.
  std::vector<int> child_joints;
  // Iterate through all the bones to find the child joints of root_joint_idx.
  for (const auto& bone : bones_) {
    int child = -1;
    if (bone[0] == root_joint_idx) {
      child = bone[1];
    }
    if (bone[1] == root_joint_idx) {
      child = bone[0];
    }
    // This child is connected (!= -1), hasn't been visited.
    if (child != -1 && (*joint_parent)[child] == -1) {
      (*joint_parent)[child] = root_joint_idx;
      child_joints.push_back(child);
    }
  }
  for (int j = 0; j < child_joints.size(); ++j) {
    ConstructSkeletonRecursive(child_joints[j], joint_parent);
  }
}

void KinematicChain::ConstructSkeleton(std::vector<int>* joint_parent) {
  // Constructs the joint to joint_parent mapping.
  ConstructSkeletonRecursive(root_joint_idx_, joint_parent);
}

absl::Status KinematicChain::ConstructKinematicChain() {
  if (joints_.empty() || bones_.empty() || joints_.size() != bones_.size()) {
    return absl::InternalError(
        "The joints or bones are incorrectly configured.");
  }
  chain_joint_parent_ = std::vector<int>(std::vector<int>(joints_.size(), -1));

  // Construct the pre-order breadth first traverse of the joint indices.
  ConstructSkeleton(&chain_joint_parent_);
  chain_joint2alpha_ = std::vector<int>(joints_.size(), -1);
  Eigen::Vector3d axis;
  int axis_idx = 0;
  std::vector<int> visited;
  for (int j = 0; j < joints_.size(); ++j) {
    int joint_idx = j;
    // Backtrack the joints via the joint_parent mapping to
    while (joint_idx != root_joint_idx_ &&
           std::find(visited.begin(), visited.end(), joint_idx) ==
               visited.end()) {
      visited.push_back(joint_idx);
      chain_joint2alpha_[joint_idx] = axis_idx;

      chain_alpha2joint_.push_back(joint_idx);
      chain_alpha_center_index_.push_back(chain_joint_parent_[joint_idx]);
      axis << 1., 0., 0.;
      chain_alpha_axis_.push_back(axis);
      chain_alpha2joint_.push_back(joint_idx);
      chain_alpha_parent_.push_back(++axis_idx);
      chain_alpha_center_index_.push_back(chain_joint_parent_[joint_idx]);
      axis << 0., 1., 0.;
      chain_alpha_axis_.push_back(axis);
      chain_alpha2joint_.push_back(joint_idx);
      chain_alpha_parent_.push_back(++axis_idx);
      chain_alpha_center_index_.push_back(chain_joint_parent_[joint_idx]);
      axis << 0., 0., 1.;
      chain_alpha_axis_.push_back(axis);
      ++axis_idx;

      int joint_parent_idx = chain_joint_parent_[joint_idx];
      if (joint_parent_idx == root_joint_idx_) {
        // This is the end of the chain.
        chain_alpha_parent_.push_back(-1);
      } else {
        if (std::find(visited.begin(), visited.end(), joint_parent_idx) !=
            visited.end()) {
          // If the parent joint has been visited, link the alpha to the
          // alpha idx of its parent's last alpha idx.
          chain_alpha_parent_.push_back(chain_joint2alpha_[joint_parent_idx]);
        } else {
          // if the parent joint has not been visited.
          chain_alpha_parent_.push_back(axis_idx);
        }
      }
      joint_idx = joint_parent_idx;
    }
  }
  return absl::OkStatus();
}

absl::Status KinematicChain::ReadPcaBasisConfig(
    std::string pca_filename, Eigen::MatrixXd* basis,
    Eigen::RowVectorXd* basis_mean) {
  std::ifstream pca_file(pca_filename);
  int row = 0;
  std::vector<std::vector<float>> rows;

  std::string line;
  while (std::getline(pca_file, line)) {
    if (line[0] == '#') continue;

    std::vector<std::string> tokens = absl::StrSplit(line, ',');
    if (row > 0 && !tokens.empty()) {
      auto& row = rows.emplace_back();
      for (const auto& value : tokens) {
        double as_double;
        CHECK(absl::SimpleAtod(value, &as_double));
        row.push_back(as_double);
      }
      CHECK_EQ(rows[0].size(), rows[rows.size() - 1].size())
          << "All basis rows must be the same size. Instead found sizes "
          << rows[rows.size() - 1].size() << " which does not match "
          << rows[0].size();
    }
    ++row;
  }
  basis_mean->resize(1, rows[0].size());
  for (int j = 0; j < rows[0].size(); ++j) {
    (*basis_mean)(0, j) = rows[0][j];
  }
  basis->resize(rows.size() - 1, rows[0].size());
  for (int i = 1; i < rows.size(); ++i) {
    for (int j = 0; j < rows[0].size(); ++j) {
      (*basis)(i - 1, j) = rows[i][j];  // Note: i is offset by 1.
    }
  }
  return absl::OkStatus();
}

}  // namespace mouse_pose
