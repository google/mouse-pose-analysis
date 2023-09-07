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

#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mouse_pose_analysis/pose_3d/cost_functions.h"
#include "mouse_pose_analysis/pose_3d/pose_optimizer.h"
#include "mouse_pose_analysis/pose_3d/pose_optimizer_utils.h"
#include "mouse_pose_analysis/pose_3d/pose_reconstruction.h"
#include "mouse_pose_analysis/pose_3d/pose_reconstruction.pb.h"
#include "mouse_pose_analysis/pose_3d/pose_reconstruction_utils.h"
#include "mouse_pose_analysis/pose_3d/rigged_mesh.h"
#include "mouse_pose_analysis/pose_3d/rigged_mesh_utils.h"
#include "mouse_pose_analysis/pose_pipeline/2d_keypoints_to_3d_mesh_calculator.pb.h"
#include "opencv2/imgproc.hpp"
#include "third_party/eigen3/Eigen/Core"

namespace mediapipe {

namespace {

using Point2d = std::pair<float, float>;

constexpr char k2DKeypointLocationsTag[] = "KEYPOINT_LOCATIONS";
constexpr char k2DKeypointLocationsSetTag[] = "KEYPOINT_LOCATIONS_SET";
constexpr char k3DKeypointLocationsTag[] = "KEYPOINT_XYZS";
constexpr char kKeypointProbsTag[] = "KEYPOINT_PROBS";
constexpr char kTrackIdsTag[] = "TRACK_IDS";
constexpr char kOptimizedKeypointLocationsTag[] =
    "OPTIMIZED_KEYPOINT_LOCATIONS";
constexpr char kRotationTranslationAlphasTag[] = "ROT_TRANS_ALPHA";
constexpr char kXYZTag[] = "XYZ";
constexpr char kKeypointsVisualizationTag[] = "KEYPOINTS_VISUALIZATION";
constexpr char kImageFrameTag[] = "IMAGE_FRAME";
constexpr char kMeshOutputDirTag[] = "MESH_OUTPUT_DIR";
constexpr char kRiggedMeshTag[] = "RIGGED_MESH";

// Side packets.
constexpr char kInputParametersFile[] = "INPUT_PARAMETERS_FILE";

void DeformAndTransformMesh(
    const std::vector<Eigen::Matrix4d>& chain_transformations,
    const std::vector<double>& trans_params,
    const std::vector<double>& rotate_params,
    mouse_pose::RiggedMesh* rigged_mesh) {
  rigged_mesh->DeformMesh(chain_transformations);
  rigged_mesh->TransformMesh(
      Eigen::Vector3f(rotate_params[0], rotate_params[1], rotate_params[2]),
      Eigen::Vector3f(trans_params[0], trans_params[1], trans_params[2]));
}

}  // namespace

// 3D skeletal mesh reconstruction from the 2D keypoints locations and camera
// matrices. (The output of the mesh is optional.)
// If output_directory is provided, save the 3D meshes to the output_directory.
//
// This code uses a mixture of floats and doubles along the following lines:
// The downstream calculators take vector<floats>, so the outputs are mostly
// floats. The optimizer uses doubles internally to avoid accumulation of errors
// when rotating chains of joints.
//
// Example config:
// node {
//   calculator: "KeypointsTo3DPoseShapeCalculator"
//   input_stream: "KEYPOINT_LOCATIONS:keypoint_locations"
//   input_stream: "KEYPOINT_WEIGHTS:keypoint_weights"
//   input_stream: "IMAGE_FRAME:image_frame"
//   input_side_packet: "MESH_OUTPUT_DIR:mesh_output_dir"
//   output_stream: "OPTIMIZED_KEYPOINT_POSITIONS:optimized_keypoint_positions"
//   output_stream: "KEYPOINTS_VISUALIZATION:keypoints_visualization"
//   output_stream: "RIGGED_MESH:rigged_mesh"
//   output_stream: "ROT_TRANS_ALPHA:rot_trans_alpha"
//   options {
//     [mediapipe.KeypointsTo3DPoseShapeCalculatorOptions] {
//       config_filename: some_reconstruction_config_filename
//     }
//   }
// }
class KeypointsTo3DPoseShapeCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;

  absl::Status Process(CalculatorContext* cc) override;

  const std::vector<Eigen::Vector3i> kColors = {{255, 0, 0},   {0, 255, 0},
                                                {0, 0, 255},   {0, 255, 255},
                                                {255, 0, 255}, {255, 255, 0}};

 private:
  absl::Status MousePoseShapeOptimization(
      mouse_pose::PoseOptimizer* pose_optimizer,
      const absl::flat_hash_map<std::string, Point2d>& input_keypoints,
      const absl::flat_hash_map<std::string, float>& input_keypoint_weights,
      std::vector<double>* keypoint_weights,
      std::vector<Eigen::Vector2d>* optimized_2d_keypoints,
      std::vector<Eigen::Matrix4d>* chain_transformations,
      std::vector<double>* rotation, std::vector<double>* translation,
      std::vector<double>* alphas, std::vector<float>* xyz,
      std::vector<double>* shape_basis_weights);

  std::unique_ptr<mouse_pose::RiggedMesh> rigged_mesh_;
  // Saves the template mesh vertices to reset the mesh on each frame.
  std::vector<Eigen::Vector3d> rigged_mesh_template_vertices_;
  mouse_pose::optical_mouse::InputParameters input_parameters_;
};
REGISTER_CALCULATOR(KeypointsTo3DPoseShapeCalculator);

absl::Status KeypointsTo3DPoseShapeCalculator::GetContract(
    CalculatorContract* cc) {
  if (cc->Inputs().HasTag(k2DKeypointLocationsTag)) {
    cc->Inputs()
        .Tag(k2DKeypointLocationsTag)
        .Set<absl::flat_hash_map<std::string, Point2d>>();
  }

  if (cc->Inputs().HasTag(k2DKeypointLocationsSetTag)) {
    cc->Inputs()
        .Tag(k2DKeypointLocationsSetTag)
        .Set<absl::flat_hash_map<std::string, std::vector<Point2d>>>();
  }
  if (cc->Inputs().HasTag(kKeypointProbsTag)) {
    cc->Inputs()
        .Tag(kKeypointProbsTag)
        .Set<absl::flat_hash_map<std::string, std::vector<float>>>();
  }
  if (cc->Inputs().HasTag(kImageFrameTag)) {
    cc->Inputs().Tag(kImageFrameTag).Set<ImageFrame>();
    cc->Outputs().Tag(kKeypointsVisualizationTag).Set<ImageFrame>();
  }
  if (cc->InputSidePackets().HasTag(kMeshOutputDirTag)) {
    cc->InputSidePackets().Tag(kMeshOutputDirTag).Set<std::string>();
  }
  if (cc->InputSidePackets().HasTag(kInputParametersFile)) {
    cc->InputSidePackets().Tag(kInputParametersFile).Set<std::string>();
  }
  if (cc->Inputs().HasTag(kTrackIdsTag)) {
    cc->Inputs().Tag(kTrackIdsTag).Set<std::vector<int>>();
  }
  if (cc->Outputs().HasTag(kRiggedMeshTag)) {
    cc->Outputs().Tag(kRiggedMeshTag).Set<mouse_pose::RiggedMesh>();
  }
  if (cc->Outputs().HasTag(kOptimizedKeypointLocationsTag)) {
    cc->Outputs()
        .Tag(kOptimizedKeypointLocationsTag)
        .Set<std::vector<std::vector<Eigen::Vector2d>>>();
  }
  if (cc->Outputs().HasTag(kRotationTranslationAlphasTag)) {
    cc->Outputs().Tag(kRotationTranslationAlphasTag).Set<std::vector<float>>();
  }
  if (cc->Outputs().HasTag(kXYZTag)) {
    cc->Outputs().Tag(kXYZTag).Set<std::vector<float>>();
  }
  if (cc->Outputs().HasTag(k3DKeypointLocationsTag)) {
    cc->Outputs()
        .Tag(k3DKeypointLocationsTag)
        .Set<absl::flat_hash_map<std::string, std::vector<float>>>();
  }
  return absl::OkStatus();
}

absl::Status KeypointsTo3DPoseShapeCalculator::Open(CalculatorContext* cc) {
  const auto& options = cc->Options<KeypointsTo3DPoseShapeCalculatorOptions>();
  std::string config_file;
  if (cc->InputSidePackets().HasTag(kInputParametersFile)) {
    config_file =
        cc->InputSidePackets().Tag(kInputParametersFile).Get<std::string>();
  } else {
    config_file = options.config_filename();
  }

  std::string config_text;
  CHECK_OK(file::GetContents(config_file, &config_text, false));
  CHECK(ParseTextProto<mouse_pose::optical_mouse::InputParameters>(
      config_text, &input_parameters_));
  LOG(INFO) << "Pose Optimizer Configuration: \n"
            << input_parameters_.DebugString();
  rigged_mesh_ = mouse_pose::CreateRiggedMeshFromFiles(
      input_parameters_.mesh_file(), input_parameters_.vertex_weight_file());
  rigged_mesh_template_vertices_ = rigged_mesh_->GetVertices();
  cc->SetOffset(0);
  return absl::OkStatus();
}

// Mouse specific 3D pose optimization.
absl::Status KeypointsTo3DPoseShapeCalculator::MousePoseShapeOptimization(
    mouse_pose::PoseOptimizer* pose_optimizer,
    const absl::flat_hash_map<std::string, Point2d>& input_keypoints,
    const absl::flat_hash_map<std::string, float>& input_keypoint_weights,
    std::vector<double>* keypoint_weights,
    std::vector<Eigen::Vector2d>* optimized_2d_keypoints,
    std::vector<Eigen::Matrix4d>* chain_transformations,
    std::vector<double>* rotation, std::vector<double>* translation,
    std::vector<double>* alphas, std::vector<float>* xyz,
    std::vector<double>* shape_basis_weights) {
  RET_CHECK_OK(pose_optimizer->LoadMouseTargetPointsFromMap(
      input_keypoints, input_keypoint_weights, keypoint_weights));

  rotation->resize(3, 0);
  translation->resize(3, 0);

  RET_CHECK_OK(ReconstructPose(input_parameters_, pose_optimizer,
                               *keypoint_weights, translation, rotation, alphas,
                               shape_basis_weights));

  RET_CHECK_OK(UpdateChainAndProjectKeypoints(
      *alphas, Eigen::Vector3d(rotation->data()),
      Eigen::Vector3d(translation->data()), *pose_optimizer,
      optimized_2d_keypoints, chain_transformations, xyz));
  return absl::OkStatus();
}

absl::Status WriteKeypointsToFile(
    std::string filename,
    const absl::flat_hash_map<std::string, std::vector<float>>& keypoints) {
  std::ofstream f(filename);
  for (const auto& [name, xyz] : keypoints) {
    int len = xyz.size();
    CHECK_GE(len, 3) << "Should be multiple of 3 and number of animals.";
    float x = xyz[len - 3];
    float y = xyz[len - 2];
    float z = xyz[len - 1];
    std::string line = absl::StrFormat("%s %f %f %f\n", name, x, y, z);
    f << line;
  }

  return absl::OkStatus();
}

// Draws rectangles over keypoints when shape is 0, circles otherwise.
absl::Status Visualize2DKeypoints(
    const std::vector<Eigen::Vector2d>* keypoint_locations,
    ImageFrame* output_image, const int radius, const Eigen::Vector3i& color,
    int shape = 1, int highlight_index = -1) {
  cv::Mat output_mat = formats::MatView(output_image);
  for (auto& keypoint : *keypoint_locations) {
    if (shape) {
      cv::circle(output_mat, cv::Point(keypoint[0], keypoint[1]), radius,
                 CV_RGB(color[0], color[1], color[2]), -1);
    } else {
      cv::rectangle(output_mat,
                    cv::Point(keypoint[0] - radius, keypoint[1] - radius),
                    cv::Point(keypoint[0] + radius, keypoint[1] + radius),
                    CV_RGB(color[0], color[1], color[2]), 2);
    }
  }

  if (highlight_index > 0) {
    auto highlighted_point = (*keypoint_locations)[highlight_index];
    cv::circle(output_mat,
               cv::Point(highlighted_point[0], highlighted_point[1]),
               radius + 1, CV_RGB(0, 255, color[2]), -1);
  }

  return absl::OkStatus();
}

void ProjectFaceToImage(const std::vector<Eigen::Vector<double, 3>>& vertices,
                        const Eigen::Matrix<double, 3, 4>& projection_matrix,
                        const mouse_pose::Face& face, cv::Mat* image) {
  std::vector<cv::Point> image_points;
  for (int i = 0; i < 3; ++i) {
    Eigen::Vector<double, 3> vertex;
    vertex = vertices[face[i]];
    Eigen::Vector<double, 2> image_point =
        mouse_pose::Project3DPoint(projection_matrix, vertex);
    image_points.push_back(cv::Point(static_cast<int>(image_point(0)),
                                     static_cast<int>(image_point(1))));
  }
  Eigen::Vector<double, 3> normal =
      (vertices[face[1]] - vertices[face[0]])
          .cross(vertices[face[2]] - vertices[face[0]])
          .normalized();
  Eigen::Vector<double, 3> view_ray;
  view_ray << 0, 0, 1;
  // If face normal is parallel, scale is 1, if face normal is perpendicular,
  // is 0.
  float scale = normal.dot(view_ray);
  if (scale > 0.) {
    cv::fillConvexPoly(*image, image_points.data(), 3,
                       cv::Scalar(255 * scale, 255 * scale, 255 * scale));
  }
}

absl::Status VisualizeRiggedMesh(
    const mouse_pose::RiggedMesh& rigged_mesh,
    const Eigen::Matrix<double, 3, 4>& projection_matrix,
    const ImageFrame& input_image, ImageFrame* output_image) {
  cv::Mat input_mat = formats::MatView(&input_image);
  cv::Mat output_mat = formats::MatView(output_image);
  for (const auto& face : rigged_mesh.GetFaces()) {
    ProjectFaceToImage(rigged_mesh.GetVertices(), projection_matrix, face,
                       &output_mat);
  }
  cv::addWeighted(input_mat, 0.7, output_mat, 0.3, 0, output_mat);
  return absl::OkStatus();
}

absl::flat_hash_map<std::string, Point2d> ChooseFirstKeypointsSet(
    const absl::flat_hash_map<std::string, std::vector<Point2d>>&
        keypoints_set) {
  absl::flat_hash_map<std::string, Point2d> first_keypoint_set;
  for (auto keypoints : keypoints_set) {
    first_keypoint_set[keypoints.first] = keypoints.second[0];
  }
  return first_keypoint_set;
}

absl::Status KeypointsTo3DPoseShapeCalculator::Process(CalculatorContext* cc) {
  LOG(INFO) << "InputTimestamp: " << cc->InputTimestamp();

  absl::flat_hash_map<std::string, std::vector<Point2d>> input_keypoints;
  if (cc->Inputs().HasTag(k2DKeypointLocationsSetTag)) {
    input_keypoints =
        cc->Inputs()
            .Tag(k2DKeypointLocationsSetTag)
            .Get<absl::flat_hash_map<std::string, std::vector<Point2d>>>();
  }
  // Keypoints from a single animal.  For backward compatibility.
  if (cc->Inputs().HasTag(k2DKeypointLocationsTag)) {
    auto single_set_input_keypoints =
        cc->Inputs()
            .Tag(k2DKeypointLocationsTag)
            .Get<absl::flat_hash_map<std::string, Point2d>>();
    for (const auto& keypoints : single_set_input_keypoints) {
      input_keypoints[keypoints.first].push_back(keypoints.second);
    }
  }

  absl::flat_hash_map<std::string, std::vector<float>> input_keypoint_probs;
  if (cc->Inputs().HasTag(kKeypointProbsTag)) {
    input_keypoint_probs =
        cc->Inputs()
            .Tag(kKeypointProbsTag)
            .Get<absl::flat_hash_map<std::string, std::vector<float>>>();
  }

  int num_animals = input_keypoints.begin()->second.size();
  LOG(INFO) << num_animals << " animal(s) detected.";

  std::vector<int> track_ids;
  if (cc->Inputs().HasTag(kTrackIdsTag)) {
    track_ids = cc->Inputs().Tag(kTrackIdsTag).Get<std::vector<int>>();
  }

  // Prepare the accumulators for the final outputs.
  std::vector<float> all_xyzs;
  std::vector<double> all_alphas;
  std::vector<double> all_rotations, all_translations;
  std::unique_ptr<ImageFrame> output_image = nullptr;
  std::vector<std::vector<Eigen::Vector2d>> all_optimized_keypoints;
  std::vector<std::string> chain_joint_names;
  std::string output_mesh_dir =
      cc->InputSidePackets().Tag(kMeshOutputDirTag).Get<std::string>();
  absl::flat_hash_map<std::string, std::vector<float>> all_keyed_xyzs;

  for (int i = 0; i < num_animals; ++i) {
    // Create a separate optimizer for each animal.
    auto pose_optimizer = std::make_unique<mouse_pose::PoseOptimizer>();
    SetUpOptimizer(input_parameters_, pose_optimizer.get());

    if (chain_joint_names.empty()) {
      chain_joint_names = pose_optimizer->GetKinematicChain().GetJointNames();
    }

    absl::flat_hash_map<std::string, Point2d> single_set_keypoints;
    absl::flat_hash_map<std::string, float> single_set_keypoint_probs;
    for (const auto& kp : input_keypoints) {
      single_set_keypoints[kp.first] = kp.second[i];
    }
    for (const auto& kp : input_keypoint_probs) {
      single_set_keypoint_probs[kp.first] = kp.second[i];
    }

    int animal_id = i;
    if (!track_ids.empty()) {
      animal_id = track_ids[i];
    }

    auto animal_id_prefix = absl::StrFormat("%02d", animal_id);
    auto keypoint_weights = std::make_unique<std::vector<double>>();
    std::vector<Eigen::Vector2d> optimized_2d_keypoints;
    auto chain_transformations =
        std::make_unique<std::vector<Eigen::Matrix4d>>();
    auto shape_basis_weights = std::make_unique<std::vector<double>>();
    auto rotation = std::make_unique<std::vector<double>>();
    auto translation = std::make_unique<std::vector<double>>();
    auto alphas = std::make_unique<std::vector<double>>();
    std::vector<float> xyz;
    RET_CHECK_OK(MousePoseShapeOptimization(
        pose_optimizer.get(), single_set_keypoints, single_set_keypoint_probs,
        keypoint_weights.get(), &optimized_2d_keypoints,
        chain_transformations.get(), rotation.get(), translation.get(),
        alphas.get(), &xyz, shape_basis_weights.get()));
    all_optimized_keypoints.push_back(optimized_2d_keypoints);

    // Accumulate the per animal values.
    absl::c_copy(xyz, std::back_inserter(all_xyzs));
    absl::c_copy(*alphas, std::back_inserter(all_alphas));
    absl::c_copy(*rotation, std::back_inserter(all_rotations));
    absl::c_copy(*translation, std::back_inserter(all_translations));
    int xyz_index = 0;
    for (const auto& joint_name : chain_joint_names) {
      all_keyed_xyzs[joint_name].push_back(xyz[xyz_index++]);
      all_keyed_xyzs[joint_name].push_back(xyz[xyz_index++]);
      all_keyed_xyzs[joint_name].push_back(xyz[xyz_index++]);
    }

    // Reset the vertices of the mesh then deform the template 3D mesh based on
    // the optimized joint angles and rigid body transformations.
    rigged_mesh_->SetVertices(rigged_mesh_template_vertices_);
    DeformAndTransformMesh(*chain_transformations, *translation, *rotation,
                           rigged_mesh_.get());

    if (cc->InputSidePackets().HasTag(kMeshOutputDirTag)) {
      std::string output_mesh_filename, output_keypoint_filename;
      output_mesh_filename =
          file::JoinPath(output_mesh_dir, cc->InputTimestamp().DebugString() +
                                              "of" + animal_id_prefix + ".obj");
      output_keypoint_filename = file::JoinPath(
          output_mesh_dir, cc->InputTimestamp().DebugString() + "of" +
                               animal_id_prefix + ".keypoint.txt");
      RET_CHECK_OK(WriteObjFile(output_mesh_filename, *rigged_mesh_));
      RET_CHECK_OK(
          WriteKeypointsToFile(output_keypoint_filename, all_keyed_xyzs));
    }

    if (cc->Inputs().HasTag(kImageFrameTag)) {
      const auto& input_image =
          cc->Inputs().Tag(kImageFrameTag).Get<ImageFrame>();
      auto color = kColors[animal_id % kColors.size()];
      if (output_image == nullptr) {
        output_image = std::make_unique<ImageFrame>(
            ImageFormat::SRGB, input_image.Width(), input_image.Height());
        output_image->CopyFrom(input_image,
                               ImageFrame::kDefaultAlignmentBoundary);
      }
      RET_CHECK_OK(VisualizeRiggedMesh(*rigged_mesh_,
                                       pose_optimizer->GetProjectionMat(),
                                       input_image, output_image.get()));

      RET_CHECK_OK(Visualize2DKeypoints(&pose_optimizer->GetTarget2DPoints(),
                                        output_image.get(), 5, color, 1));
    }
  }

  // Outputs.
  if (cc->Outputs().HasTag(kRotationTranslationAlphasTag)) {
    auto rot_trans_alpha = std::make_unique<std::vector<float>>(
        all_rotations.size() + all_translations.size() + all_alphas.size());
    int position = 0;
    for (int i = 0; i < all_rotations.size(); ++i, ++position) {
      (*rot_trans_alpha)[position] = static_cast<float>(all_rotations.at(i));
    }
    for (int i = 0; i < all_translations.size(); ++i, ++position) {
      (*rot_trans_alpha)[position] = static_cast<float>(all_translations.at(i));
    }
    for (int i = 0; i < all_alphas.size(); ++i, ++position) {
      (*rot_trans_alpha)[position] = static_cast<float>(all_alphas.at(i));
    }
    cc->Outputs()
        .Tag(kRotationTranslationAlphasTag)
        .Add(rot_trans_alpha.release(), cc->InputTimestamp());
  }
  if (cc->Outputs().HasTag(kXYZTag)) {
    auto xyzs_ptr = std::make_unique<std::vector<float>>(all_xyzs);
    cc->Outputs().Tag(kXYZTag).Add(xyzs_ptr.release(), cc->InputTimestamp());
  }
  if (cc->Outputs().HasTag(k3DKeypointLocationsTag)) {
    auto keyed_xyzs =
        std::make_unique<absl::flat_hash_map<std::string, std::vector<float>>>(
            all_keyed_xyzs);

    cc->Outputs()
        .Tag(k3DKeypointLocationsTag)
        .Add(keyed_xyzs.release(), cc->InputTimestamp());
  }

  if (cc->Outputs().HasTag(kRiggedMeshTag)) {
    auto reconstructed_rigged_mesh = std::make_unique<mouse_pose::RiggedMesh>();
    reconstructed_rigged_mesh->SetVertices(rigged_mesh_->GetVertices());
    reconstructed_rigged_mesh->SetFaces(rigged_mesh_->GetFaces());
    cc->Outputs()
        .Tag(kRiggedMeshTag)
        .Add(reconstructed_rigged_mesh.release(), cc->InputTimestamp());
  }

  if (cc->Inputs().HasTag(kImageFrameTag)) {
    CHECK(output_image) << "Null output_image.";
    cc->Outputs()
        .Tag(kKeypointsVisualizationTag)
        .Add(output_image.release(), cc->InputTimestamp());
  }

  if (cc->Outputs().HasTag(kOptimizedKeypointLocationsTag)) {
    auto optimized_keypoints_ptr =
        std::make_unique<std::vector<std::vector<Eigen::Vector2d>>>(
            all_optimized_keypoints);
    cc->Outputs()
        .Tag(kOptimizedKeypointLocationsTag)
        .Add(optimized_keypoints_ptr.release(), cc->InputTimestamp());
  }
  return absl::OkStatus();
}
}  // namespace mediapipe
