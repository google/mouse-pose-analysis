# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code for pose reconstruction with optimization.
licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "status",
    srcs = [
        "status.cc",
    ],
    hdrs = [
        "status.h",
    ],
    deps = [
        "//third_party:glog",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "kinematic_chain",
    srcs = ["kinematic_chain.cc"],
    hdrs = ["kinematic_chain.h"],
    deps = [
        ":matrix_util",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
    ],
)

cc_test(
    name = "kinematic_chain_test",
    srcs = ["kinematic_chain_test.cc"],
    data = [
        "//mouse_pose_analysis/pose_3d/testdata:data",
    ],
    deps = [
        ":gtest_matchers_lib",
        ":gtest_util_lib",
        ":kinematic_chain",
        "@com_google_absl//absl/flags:flag",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "pose_reconstruction_utils",
    srcs = ["pose_reconstruction_utils.cc"],
    hdrs = ["pose_reconstruction_utils.h"],
    deps = [
        ":kinematic_chain",
        ":pose_optimizer",
        ":rigged_mesh",
        "//third_party:opencv",
        "@ceres_solver//:ceres",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@eigen_archive//:eigen3",
    ],
)

cc_test(
    name = "pose_reconstruction_utils_test",
    srcs = ["pose_reconstruction_utils_test.cc"],
    data = [
        "//mouse_pose_analysis/pose_3d/testdata:data",
    ],
    deps = [
        ":gtest_matchers_lib",
        ":gtest_util_lib",
        ":kinematic_chain",
        ":pose_reconstruction_utils",
        ":rigged_mesh",
        ":status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

proto_library(
    name = "keypoint_proto",
    srcs = ["keypoint.proto"],
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "keypoint_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":keypoint_proto"],
)

proto_library(
    name = "pose_reconstruction_proto",
    srcs = ["pose_reconstruction.proto"],
    visibility = ["//visibility:public"],
    deps = [":keypoint_proto"],
)

cc_proto_library(
    name = "pose_reconstruction_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":pose_reconstruction_proto"],
)

cc_library(
    name = "pose_reconstruction",
    srcs = ["pose_reconstruction.cc"],
    hdrs = ["pose_reconstruction.h"],
    deps = [
        ":kinematic_chain",
        ":pose_optimizer",
        ":pose_reconstruction_cc_proto",
        ":pose_reconstruction_utils",
        "//third_party:glog",
        "//third_party:opencv",
        "@ceres_solver//:ceres",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@eigen_archive//:eigen3",
    ],
)

cc_test(
    name = "pose_reconstruction_test",
    size = "medium",
    srcs = ["pose_reconstruction_test.cc"],
    args = [
        "--fibers_default_thread_stack_size=4048576",
    ],
    data = [
        "//mouse_pose_analysis/pose_3d/testdata:data",
    ],
    deps = [
        ":gtest_matchers_lib",
        ":gtest_util_lib",
        ":kinematic_chain",
        ":pose_optimizer",
        ":pose_reconstruction",
        ":pose_reconstruction_cc_proto",
        ":pose_reconstruction_utils",
        ":rigged_mesh",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_binary(
    name = "pose_reconstruction_main",
    srcs = ["pose_reconstruction_main.cc"],
    data = [
        "//mouse_pose_analysis/pose_3d/testdata:data",
    ],
    deps = [
        ":kinematic_chain",
        ":pose_optimizer",
        ":pose_reconstruction",
        ":pose_reconstruction_cc_proto",
        ":pose_reconstruction_utils",
        ":rigged_mesh",
        "//third_party:opencv",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings:str_format",
        "@eigen_archive//:eigen3",
    ],
)

cc_test(
    name = "cost_functions_test",
    srcs = ["cost_functions_test.cc"],
    data = [
        "//mouse_pose_analysis/pose_3d/testdata:data",
    ],
    deps = [
        ":gmm_prior",
        ":gtest_matchers_lib",
        ":gtest_util_lib",
        ":kinematic_chain",
        ":pose_optimizer",
        ":rigged_mesh",
        ":status",
        "//third_party:opencv",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "pose_optimizer",
    srcs = [
        "pose_optimizer.cc",
        "pose_optimizer_utils.cc",
    ],
    hdrs = [
        "cost_functions.h",
        "pose_optimizer.h",
        "pose_optimizer_utils.h",
    ],
    deps = [
        ":gmm_prior",
        ":gtest_matchers_lib",
        ":keypoint_cc_proto",
        ":kinematic_chain",
        ":matrix_util",
        ":rigged_mesh",
        ":status",
        "//third_party:opencv",
        "@ceres_solver//:ceres",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@eigen_archive//:eigen3",
    ],
)

cc_test(
    name = "pose_optimizer_test",
    srcs = ["pose_optimizer_test.cc"],
    data = [
        "//mouse_pose_analysis/pose_3d/testdata:data",
    ],
    deps = [
        ":gtest_matchers_lib",
        ":gtest_util_lib",
        ":pose_optimizer",
        ":pose_reconstruction_cc_proto",
        ":rigged_mesh",
        "//third_party:opencv",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "matrix_util",
    hdrs = ["matrix_util.h"],
    deps = [
        "@ceres_solver//:ceres",
        "@eigen_archive//:eigen3",
    ],
)

cc_test(
    name = "matrix_util_test",
    srcs = ["matrix_util_test.cc"],
    deps = [
        ":gtest_matchers_lib",
        ":matrix_util",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen3",
    ],
)

cc_test(
    name = "rigged_mesh_test",
    srcs = ["rigged_mesh_test.cc"],
    data = [
        "//mouse_pose_analysis/pose_3d/testdata:data",
    ],
    deps = [
        ":gtest_matchers_lib",
        ":gtest_util_lib",
        ":rigged_mesh",
        "@com_google_absl//absl/flags:flag",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gtest_matchers_lib",
    hdrs = ["gtest_matchers.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "gtest_util_lib",
    srcs = ["gtest_util.cc"],
    hdrs = ["gtest_util.h"],
    deps = [
    ],
)

cc_library(
    name = "rigged_mesh",
    srcs = [
        "rigged_mesh.cc",
        "rigged_mesh_utils.cc",
    ],
    hdrs = [
        "rigged_mesh.h",
        "rigged_mesh_utils.h",
    ],
    deps = [
        ":matrix_util",
        "//third_party:glog",
        "//third_party:opencv",
        "@ceres_solver//:ceres",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@eigen_archive//:eigen3",
    ],
)

proto_library(
    name = "gmm_proto",
    srcs = ["gmm.proto"],
)

cc_proto_library(
    name = "gmm_cc_proto",
    deps = [":gmm_proto"],
)

proto_library(
    name = "bones_gmm_proto",
    srcs = ["bones_gmm.proto"],
)

cc_proto_library(
    name = "bones_gmm_cc_proto",
    deps = [":bones_gmm_proto"],
)

proto_library(
    name = "bones_pca_proto",
    srcs = ["bones_pca.proto"],
)

cc_proto_library(
    name = "bones_pca_cc_proto",
    deps = [":bones_pca_proto"],
)

cc_library(
    name = "gmm_prior",
    hdrs = ["gmm_prior.h"],
    deps = [
        ":gmm_cc_proto",
        "@ceres_solver//:ceres",
        "@com_google_protobuf//:protobuf",
        "@eigen_archive//:eigen3",
    ],
)

cc_test(
    name = "gmm_prior_test",
    srcs = ["gmm_prior_test.cc"],
    data = [
        "//mouse_pose_analysis/pose_3d/testdata:data",
    ],
    deps = [
        ":gmm_cc_proto",
        ":gmm_prior",
        ":gtest_matchers_lib",
        ":gtest_util_lib",
        ":kinematic_chain",
        "@com_google_googletest//:gtest_main",
    ],
)
