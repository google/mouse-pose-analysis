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

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

mediapipe_proto_library(
    name = "tflite_image_inference_calculator_proto",
    srcs = ["tflite_image_inference_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "tflite_image_inference_calculator",
    srcs = ["tflite_image_inference_calculator.cc"],
    deps = [
        ":tflite_image_inference_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/stream_handler:fixed_size_input_stream_handler",
        "//mediapipe/util:cpu_util",
        "//mediapipe/util/tflite:tflite_model_loader",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "2d_keypoints_to_3d_mesh_calculator_proto",
    srcs = ["2d_keypoints_to_3d_mesh_calculator.proto"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

cc_library(
    name = "2d_keypoints_to_3d_mesh_calculator",
    srcs = ["2d_keypoints_to_3d_mesh_calculator.cc"],
    copts = [
        "$(STACK_FRAME_UNLIMITED)",
    ],
    deps = [
        ":2d_keypoints_to_3d_mesh_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:parse_text_proto",
        "//mouse_pose_analysis/pose_3d:pose_optimizer",
        "//mouse_pose_analysis/pose_3d:pose_reconstruction",
        "//mouse_pose_analysis/pose_3d:pose_reconstruction_cc_proto",
        "//mouse_pose_analysis/pose_3d:pose_reconstruction_utils",
        "//mouse_pose_analysis/pose_3d:rigged_mesh",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@eigen_archive//:eigen3",
    ],
    alwayslink = 1,
)

cc_test(
    name = "2d_keypoints_to_3d_mesh_calculator_test",
    size = "enormous",
    srcs = ["2d_keypoints_to_3d_mesh_calculator_test.cc"],
    args = [
        "--fibers_default_thread_stack_size=4048576",
    ],
    data = [
        "//mouse_pose_analysis/pose_pipeline/testdata:data",
    ],
    deps = [
        ":2d_keypoints_to_3d_mesh_calculator",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:status",
        "//mouse_pose_analysis/pose_3d:gtest_matchers_lib",
        "//mouse_pose_analysis/pose_3d:gtest_util_lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
        "@org_tensorflow//tensorflow/core:framework",
    ],
)

cc_library(
    name = "simple_run_graph_main",
    srcs = ["simple_run_graph_main.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:map_util",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
)

mediapipe_proto_library(
    name = "detection_to_keypoints_calculator_proto",
    srcs = ["detection_to_keypoints_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "detection_to_keypoints_calculator",
    srcs = ["detection_to_keypoints_calculator.cc"],
    deps = [
        ":detection_to_keypoints_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "pose_pipeline_calculators",
    deps = [
        ":2d_keypoints_to_3d_mesh_calculator",
        ":detection_to_keypoints_calculator",
        ":tflite_image_inference_calculator",
        "//mediapipe/calculators/video:opencv_video_decoder_calculator",
        "//mediapipe/calculators/video:opencv_video_encoder_calculator",
        "//mediapipe/calculators/video:video_pre_stream_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        #"//mediapipe/calculators/tflite:tflite_inference_calculator",
    ],
)

cc_binary(
    name = "run_mouse_pose",
    deps = [
        ":pose_pipeline_calculators",
        ":simple_run_graph_main",
    ],
)
