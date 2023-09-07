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

#include "mouse_pose_analysis/pose_3d/gmm_prior.h"

#include <string>
#include <vector>

#include "glog/logging.h"
#include "googlemock/include/gmock/gmock.h"
#include "gtest/gtest.h"
#include "mouse_pose_analysis/pose_3d/gmm.pb.h"
#include "mouse_pose_analysis/pose_3d/gtest_matchers.h"
#include "mouse_pose_analysis/pose_3d/gtest_util.h"
#include "mouse_pose_analysis/pose_3d/kinematic_chain.h"

namespace mouse_pose {
namespace {

using ::testing::Test;

GaussianMixtureModelProto CreateSimpleProto() {
  GaussianMixtureModelProto proto;
  proto.set_number_of_components(2);
  proto.set_number_of_dimensions(2);
  std::vector<double> mixing_proprotions = {0.45, 0.55};
  proto.mutable_mixing_proportions()->Add(mixing_proprotions.begin(),
                                          mixing_proprotions.end());

  std::vector<double> means = {0.0, 0.0, 2.0, 2.0};
  proto.mutable_means()->Add(means.begin(), means.end());

  std::vector<double> precisions = {1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0};
  proto.mutable_precisions()->Add(precisions.begin(), precisions.end());
  return proto;
}

GaussianMixtureModelProto CreateComplicatedProto() {
  GaussianMixtureModelProto proto;
  proto.set_number_of_components(3);
  proto.set_number_of_dimensions(2);
  std::vector<double> mixing_proprotions = {0.45, 0.35, 0.2};
  proto.mutable_mixing_proportions()->Add(mixing_proprotions.begin(),
                                          mixing_proprotions.end());

  std::vector<double> means = {0.0, 0.0, 2.0, 2.0, 0.1, 0.57};
  proto.mutable_means()->Add(means.begin(), means.end());

  std::vector<double> precisions = {1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
                                    0.0, 1.0, 1.0, 0.5, 0.5, 1.0};
  proto.mutable_precisions()->Add(precisions.begin(), precisions.end());
  return proto;
}

TEST(GmmPriorTest, GmmProtoInitializationTest) {
  GaussianMixtureModel gmm;
  GaussianMixtureModelProto proto = CreateSimpleProto();
  gmm.InitializeFromProto(proto);
}

TEST(GmmPriorTest, GmmFileInitializationTest) {
  GaussianMixtureModelProto proto = CreateSimpleProto();
  std::string filepath = "/tmp/test.pbtxt";
  std::string buf;
  ASSERT_TRUE(google::protobuf::TextFormat::PrintToString(proto, &buf));
  std::ofstream proto_file(filepath);
  proto_file << buf;
  GaussianMixtureModel gmm;
  gmm.InitializeFromFile(filepath);
}

TEST(GmmPriorTest, GmmFileInitializationFromPythonTest) {
  std::string filepath =
      GetTestRootDir() + "pose_3d/testdata/gmm_mixture_5.pbtxt";
  GaussianMixtureModel gmm;
  gmm.InitializeFromFile(filepath);
  gmm.CheckIsScaledAndRotated();
}

TEST(GmmPriorTest, SimpleLikelihoodTest) {
  GaussianMixtureModel gmm;
  GaussianMixtureModelProto proto = CreateSimpleProto();
  gmm.InitializeFromProto(proto);
  Eigen::Matrix<double, 1, 2> point_0 = {0.f, 0.f};
  Eigen::Matrix<double, 1, 2> point_1 = {1.f, 1.f};
  Eigen::Matrix<double, 1, 2> point_2 = {2.f, 2.f};

  LOG(INFO) << "L(p0) = " << gmm.LogLikelihood<double>(point_0);
  LOG(INFO) << "L(p1) = " << gmm.LogLikelihood<double>(point_1);
  LOG(INFO) << "L(p2) = " << gmm.LogLikelihood<double>(point_2);

  // Golden values are from scipy.stats.multivariate_normal and logsumexp.
  ASSERT_NEAR(-2.614245865688321, gmm.LogLikelihood<double>(point_0), 1e-5);
  ASSERT_NEAR(-2.8378770664093453, gmm.LogLikelihood<double>(point_1), 1e-5);
  ASSERT_NEAR(-2.4208397180959462, gmm.LogLikelihood<double>(point_2), 1e-5);
}

TEST(GmmPriorTest, ComplicatedLikelihoodTest) {
  GaussianMixtureModel gmm;
  GaussianMixtureModelProto proto = CreateComplicatedProto();
  gmm.InitializeFromProto(proto);
  Eigen::Matrix<double, 1, 2> point_0 = {0.f, 0.f};
  Eigen::Matrix<double, 1, 2> point_1 = {1.f, 1.f};
  Eigen::Matrix<double, 1, 2> point_2 = {2.f, 2.f};

  LOG(INFO) << "L(p0) = " << gmm.LogLikelihood<double>(point_0);
  LOG(INFO) << "L(p1) = " << gmm.LogLikelihood<double>(point_1);
  LOG(INFO) << "L(p2) = " << gmm.LogLikelihood<double>(point_2);

  // Golden values are from scipy.stats.multivariate_normal and logsumexp.
  ASSERT_NEAR(-2.3507142317786283, gmm.LogLikelihood<double>(point_0), 1e-5);
  ASSERT_NEAR(-2.8025795012667194, gmm.LogLikelihood<double>(point_1), 1e-5);
  ASSERT_NEAR(-2.8570976665719083, gmm.LogLikelihood<double>(point_2), 1e-5);
}

TEST(GmmPriorTest, AlignJointsNoChangeTest) {
  GaussianMixtureModel gmm;
  std::vector<Eigen::Matrix<double, 3, 1>> points = {
      {0., 0., 0.},  {1., 0., 0.}, {10., 0., 0.}, {0., 1., 0.},
      {0., 1., 0.},  {0., 1., 0.}, {0., 1., 0.},  {0., 1., 0.},
      {0., 1., 0.},  {0., 1., 0.}, {0., 1., 0.},  {0., 1., 0.},
      {0., -1., 0.}, {0., 1., 0.}, {0., 1., 0.},  {0., 1., 0.}};
  Eigen::Matrix<double, Eigen::Dynamic, 1> flattened;
  gmm.AlignPointsAndFlatten<double>(points, &flattened);

  ASSERT_EQ(48, flattened.rows());
  ASSERT_NEAR(0., flattened(0, 0), 1e-5);
  ASSERT_NEAR(0., flattened(1, 0), 1e-5);
  ASSERT_NEAR(0., flattened(2, 0), 1e-5);
  ASSERT_NEAR(1., flattened(3, 0), 1e-5);
  ASSERT_NEAR(0., flattened(4, 0), 1e-5);
  ASSERT_NEAR(0., flattened(5, 0), 1e-5);
  ASSERT_NEAR(10, flattened(6, 0), 1e-5);
  ASSERT_NEAR(0., flattened(7, 0), 1e-5);
  ASSERT_NEAR(0., flattened(8, 0), 1e-5);
}

TEST(GmmPriorTest, AlignJointsScaleTest) {
  GaussianMixtureModel gmm;
  std::vector<Eigen::Matrix<double, 3, 1>> points = {
      {0., 0., 0.},  {0.5, 0., 0.}, {0., 10., 0.}, {0., 1., 0.},
      {0., 1., 0.},  {0., 1., 0.},  {0., 1., 0.},  {0., 1., 0.},
      {0., 1., 0.},  {0., 1., 0.},  {0., 1., 0.},  {0., 1., 0.},
      {0., -1., 0.}, {0., 1., 0.},  {0., 1., 0.},  {0., 1., 0.}};
  Eigen::Matrix<double, Eigen::Dynamic, 1> flattened;
  gmm.AlignPointsAndFlatten<double>(points, &flattened);

  ASSERT_EQ(48, flattened.rows());
  ASSERT_NEAR(0., flattened(0, 0), 1e-5);
  ASSERT_NEAR(0., flattened(1, 0), 1e-5);
  ASSERT_NEAR(0., flattened(2, 0), 1e-5);
  ASSERT_NEAR(1., flattened(3, 0), 1e-5);
  ASSERT_NEAR(0., flattened(4, 0), 1e-5);
  ASSERT_NEAR(0., flattened(5, 0), 1e-5);
  ASSERT_NEAR(0, flattened(6, 0), 1e-5);
  ASSERT_NEAR(20., flattened(7, 0), 1e-5);
  ASSERT_NEAR(0., flattened(8, 0), 1e-5);
}

TEST(GmmPriorTest, AlignJointsRotateTest) {
  GaussianMixtureModel gmm;
  std::vector<Eigen::Matrix<double, 3, 1>> points = {
      {0., 0., 0.},    {0.5, 0.5, 0.}, {-0.5, -0.5, 0.}, {-0.5, 0.5, 0.},
      {0.5, -0.5, 0.}, {0., 1., 0.},   {0., 1., 0.},     {0., 1., 0.},
      {0., 1., 0.},    {0., 1., 0.},   {0., 1., 0.},     {0., 1., 0.},
      {0., -1., 0.},   {0., 1., 0.},   {0., 1., 0.},     {0., 1., 0.}};
  Eigen::Matrix<double, Eigen::Dynamic, 1> flattened;
  gmm.AlignPointsAndFlatten<double>(points, &flattened);

  ASSERT_EQ(48, flattened.rows());
  ASSERT_NEAR(0., flattened(0, 0), 1e-5);
  ASSERT_NEAR(0., flattened(1, 0), 1e-5);
  ASSERT_NEAR(0., flattened(2, 0), 1e-5);
  ASSERT_NEAR(1., flattened(3, 0), 1e-5);
  ASSERT_NEAR(0., flattened(4, 0), 1e-5);
  ASSERT_NEAR(0., flattened(5, 0), 1e-5);
  ASSERT_NEAR(-1, flattened(6, 0), 1e-5);
  ASSERT_NEAR(0., flattened(7, 0), 1e-5);
  ASSERT_NEAR(0., flattened(8, 0), 1e-5);
  ASSERT_NEAR(0, flattened(9, 0), 1e-5);
  ASSERT_NEAR(1., flattened(10, 0), 1e-5);
  ASSERT_NEAR(0., flattened(11, 0), 1e-5);
  ASSERT_NEAR(0, flattened(12, 0), 1e-5);
  ASSERT_NEAR(-1., flattened(13, 0), 1e-5);
  ASSERT_NEAR(0., flattened(14, 0), 1e-5);
}

TEST(GmmPriorTest, AlignJointsOffsetRotateTest) {
  GaussianMixtureModel gmm;
  std::vector<Eigen::Matrix<double, 3, 1>> points = {
      {1., 1., 1.},   {1.5, 1.5, 1.}, {0.5, 0.5, 1.}, {0.5, 1.5, 1.},
      {1.5, 0.5, 1.}, {0., 1., 0.},   {0., 1., 0.},   {0., 1., 0.},
      {0., 1., 0.},   {0., 1., 0.},   {0., 1., 0.},   {0., 1., 0.},
      {0., -1., 0.},  {0., 1., 0.},   {0., 1., 0.},   {0., 1., 0.}};
  Eigen::Matrix<double, Eigen::Dynamic, 1> flattened;
  gmm.AlignPointsAndFlatten<double>(points, &flattened);

  ASSERT_EQ(48, flattened.rows());
  ASSERT_NEAR(0., flattened(0, 0), 1e-5);
  ASSERT_NEAR(0., flattened(1, 0), 1e-5);
  ASSERT_NEAR(0., flattened(2, 0), 1e-5);
  ASSERT_NEAR(1., flattened(3, 0), 1e-5);
  ASSERT_NEAR(0., flattened(4, 0), 1e-5);
  ASSERT_NEAR(0., flattened(5, 0), 1e-5);
  ASSERT_NEAR(-1, flattened(6, 0), 1e-5);
  ASSERT_NEAR(0., flattened(7, 0), 1e-5);
  ASSERT_NEAR(0., flattened(8, 0), 1e-5);
  ASSERT_NEAR(0, flattened(9, 0), 1e-5);
  ASSERT_NEAR(1., flattened(10, 0), 1e-5);
  ASSERT_NEAR(0., flattened(11, 0), 1e-5);
  ASSERT_NEAR(0, flattened(12, 0), 1e-5);
  ASSERT_NEAR(-1., flattened(13, 0), 1e-5);
  ASSERT_NEAR(0., flattened(14, 0), 1e-5);
}

TEST(GmmPriorTest, AlignJointsOffsetRotateScaleTest) {
  GaussianMixtureModel gmm;
  std::vector<Eigen::Matrix<double, 3, 1>> points = {
      {1., 1., 1.},  {1.5, 1.5, 1.5}, {-9., -9., -9.}, {0., 1., 0.},
      {0., 1., 0.},  {0., 1., 0.},    {0., 1., 0.},    {0., 1., 0.},
      {0., 1., 0.},  {0., 1., 0.},    {0., 1., 0.},    {0., 1., 0.},
      {0., -1., 0.}, {0., 1., 0.},    {0., 1., 0.},    {0., 1., 0.}};
  Eigen::Matrix<double, Eigen::Dynamic, 1> flattened;
  gmm.AlignPointsAndFlatten<double>(points, &flattened);

  ASSERT_EQ(48, flattened.rows());
  ASSERT_NEAR(0., flattened(0, 0), 1e-5);
  ASSERT_NEAR(0., flattened(1, 0), 1e-5);
  ASSERT_NEAR(0., flattened(2, 0), 1e-5);
  ASSERT_NEAR(1., flattened(3, 0), 1e-5);
  ASSERT_NEAR(0., flattened(4, 0), 1e-5);
  ASSERT_NEAR(0., flattened(5, 0), 1e-5);
  ASSERT_NEAR(-20, flattened(6, 0), 1e-5);
  ASSERT_NEAR(0., flattened(7, 0), 1e-5);
  ASSERT_NEAR(0., flattened(8, 0), 1e-5);
}

TEST(GmmPriorTest, AlignJointsAlongYTest) {
  GaussianMixtureModel gmm;
  std::vector<Eigen::Matrix<double, 3, 1>> points = {
      {0., 0., 0.},          {1., 0., 0.}, {10., 0., 0.},
      {0., 1., 0.},          {0., 1., 0.}, {0., 1., 0.},
      {0., 1., 0.},          {0., 1., 0.}, {0., 1., 0.},
      {0., 1., 0.},          {0., 1., 0.}, {0., 1., 0.},
      {0., 1.4142, 1.4142},  {0., 1., 0.}, {0., 1., 0.},
      {0., -1.4142, -1.4142}};
  Eigen::Matrix<double, Eigen::Dynamic, 1> flattened;
  gmm.AlignPointsAndFlatten<double>(points, &flattened);

  ASSERT_EQ(48, flattened.rows());
  ASSERT_NEAR(0., flattened(0, 0), 1e-5);
  ASSERT_NEAR(0., flattened(1, 0), 1e-5);
  ASSERT_NEAR(0., flattened(2, 0), 1e-5);
  ASSERT_NEAR(1., flattened(3, 0), 1e-5);
  ASSERT_NEAR(0., flattened(4, 0), 1e-5);
  ASSERT_NEAR(0., flattened(5, 0), 1e-5);
  ASSERT_NEAR(10, flattened(6, 0), 1e-5);
  ASSERT_NEAR(0., flattened(7, 0), 1e-5);
  ASSERT_NEAR(0., flattened(8, 0), 1e-5);
  ASSERT_NEAR(0., flattened(36, 0), 1e-5);
  // The y and z magnitudes are only specified to 1e-4, so only require
  // precision of those points to 1e-3.
  ASSERT_NEAR(-2, flattened(37, 0), 1e-3);
  ASSERT_NEAR(-0., flattened(38, 0), 1e-3);
  ASSERT_NEAR(0., flattened(45, 0), 1e-5);
  ASSERT_NEAR(2., flattened(46, 0), 1e-3);
  ASSERT_NEAR(0., flattened(47, 0), 1e-3);
}

TEST(GmmPriorTest, AlignJointsOppositeXTest) {
  GaussianMixtureModel gmm;
  std::vector<Eigen::Matrix<double, 3, 1>> points = {
      {0., 0., 0.},  {-1., 0., 0.}, {10., 0., 0.}, {0., 1., 0.},
      {0., 1., 0.},  {0., 1., 0.},  {0., 1., 0.},  {0., 1., 0.},
      {0., 1., 0.},  {0., 1., 0.},  {0., 1., 0.},  {0., 1., 0.},
      {0., -1., 0.}, {0., 1., 0.},  {0., 1., 0.},  {0., 1., 0.}};
  Eigen::Matrix<double, Eigen::Dynamic, 1> flattened;
  gmm.AlignPointsAndFlatten<double>(points, &flattened);

  ASSERT_EQ(48, flattened.rows());
  ASSERT_NEAR(0., flattened(0, 0), 1e-5);
  ASSERT_NEAR(0., flattened(1, 0), 1e-5);
  ASSERT_NEAR(0., flattened(2, 0), 1e-5);
  ASSERT_NEAR(1., flattened(3, 0), 1e-5);
  ASSERT_NEAR(0., flattened(4, 0), 1e-5);
  ASSERT_NEAR(0., flattened(5, 0), 1e-5);
}

TEST(GmmPriorTest, RunGmmFromFileWithGoldenBoundTest) {
  std::string filepath =
      GetTestRootDir() + "pose_3d/testdata/gmm_mixture_5.pbtxt";
  GaussianMixtureModel gmm;
  gmm.InitializeFromFile(filepath);
  gmm.CheckIsScaledAndRotated();

  std::string bone_filepath =
      GetTestRootDir() + "pose_3d/testdata/test_mouse_skeleton.csv";
  KinematicChain chain;
  EXPECT_OK(chain.ReadSkeletonConfig(bone_filepath));
  Eigen::Matrix<double, Eigen::Dynamic, 1> flattened;
  gmm.AlignPointsAndFlatten<double>(chain.GetJoints(), &flattened);
  LOG(INFO) << "L = " << gmm.LogLikelihood<double>(flattened);
  EXPECT_LT(4, gmm.LogLikelihood<double>(flattened));
}

TEST(GmmPriorTest, RunGmmFromFileWithKnownFailureTest) {
  std::string filepath =
      GetTestRootDir() + "pose_3d/testdata/gmm_mixture_20_fixed.pbtxt";
  GaussianMixtureModel gmm;
  gmm.InitializeFromFile(filepath);
  gmm.CheckIsScaledAndRotated();

  Eigen::Matrix<double, Eigen::Dynamic, 1> flattened;  // need 54 values
  flattened.resize(54, 1);
  flattened << 0., 0., 0., 1., 0., 0., 1.94395, 0.00587299, -0.657225, 1.93288,
      -0.275046, -0.655325, 1.62518, -0.656085, -0.886929, 2.35544, -0.464833,
      -1.26966, 1.94378, 0.271091, -0.654789, 1.68949, 0.659682, -0.994304,
      2.39855, 0.420478, -1.27893, 2.51549, 0.00533256, -1.01684, -0.303174,
      -0.00218298, 0.244327, -1.13041, -0.00202348, 0.226631, -0.202506,
      -0.296871, 0.0780242, -0.294489, -0.579717, -0.182936, -0.923323,
      -0.557961, 0.0399649, -0.189598, 0.278692, 0.0780242, -0.125297, 0.601074,
      -0.23979, -0.717636, 0.536861, -0.0228448;
  ASSERT_EQ(flattened.rows(), 54);
  LOG(INFO) << "L = " << gmm.LogLikelihood<double>(flattened);
  EXPECT_LT(gmm.LogLikelihood<double>(flattened), 0);
}

}  // namespace
}  // namespace mouse_pose
