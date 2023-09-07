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

#include "mouse_pose_analysis/pose_3d/pose_optimizer_utils.h"

#include <cstdint>
#include <fstream>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "opencv2/imgproc.hpp"
#include "mouse_pose_analysis/pose_3d/status.h"

namespace mouse_pose {
absl::Status LoadAndConstructKinematicChain(std::string chain_config_filename,
                                            KinematicChain* chain) {
  CHECK_OK(chain->ReadSkeletonConfig(chain_config_filename));
  CHECK_OK(chain->ConstructKinematicChain());
  return absl::OkStatus();
}

absl::Status LoadTargetPointsFromFile(
    std::string target_points_filename,
    std::vector<Eigen::Vector2d>* target_2d_points) {
  std::ifstream csv_file(target_points_filename);
  CHECK(csv_file) << "Failed to open file:" << target_points_filename;
  target_2d_points->clear();
  std::string line;
  while (std::getline(csv_file, line)) {
    if (line[0] == '#') continue;
    std::vector<std::string> tokens = absl::StrSplit(line, ',');
    CHECK_EQ(tokens.size(), 4) << "Format error of line:" << line;
    Eigen::Vector2d point;
    CHECK(absl::SimpleAtod(tokens[2], &point[0]));
    CHECK(absl::SimpleAtod(tokens[3], &point[1]));
    target_2d_points->push_back(point);
  }
  return absl::OkStatus();
}

absl::Status LoadProjectionMatFromFile(
    std::string projection_mat_filename,
    Eigen::Matrix<double, 3, 4>* projection_matrix) {
  std::ifstream csv_file(projection_mat_filename);
  CHECK(csv_file) << "Failed to open file:" << projection_mat_filename;
  int row_index = 0;
  std::string line;
  while (std::getline(csv_file, line)) {
    if (line[0] == '#') continue;
    std::vector<std::string> tokens =
        absl::StrSplit(line, ' ', absl::SkipEmpty());
    CHECK_EQ(tokens.size(), 4) << "Input matrix's number of"
                               << "columns is not 4";
    CHECK(row_index < 4) << "Input matrix has more than 3 rows.";
    for (int col_index = 0; col_index < tokens.size(); ++col_index) {
      CHECK(absl::SimpleAtod(tokens[col_index],
                             &((*projection_matrix)(row_index, col_index))));
    }
    row_index++;
  }
  return absl::OkStatus();
}

}  // namespace mouse_pose
