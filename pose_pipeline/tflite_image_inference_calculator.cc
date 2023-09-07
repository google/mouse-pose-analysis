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

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/tflite/tflite_model_loader.h"
#include "mouse_pose_analysis/pose_pipeline/tflite_image_inference_calculator.pb.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

namespace {
constexpr char kImageTag[] = "IMAGE";
constexpr char kTensorsTag[] = "TENSORS";
}  // namespace

namespace mediapipe {

// Runs inference on the provided image input and TFLite model.
//
// Input:
//  IMAGE - Image.
//
// Output:
//  TENSORS - Vector of TfLiteTensor of type kTfLiteFloat32 or kTfLiteUInt8
//
// Example use:
// node {
//   calculator: "TfLiteImageInferenceCalculator"
//   input_stream: "IMAGE:image"
//   output_stream: "TENSORS:tensors"
//   options: {
//     [mediapipe.TfLiteImageInferenceCalculatorOptions.ext] {
//       model_path: "modelname.tflite"
//     }
//   }
// }
//
//  All output TfLiteTensors will be destroyed when the graph closes,
//  (i.e. after calling graph.WaitUntilDone()).
//  This calculator uses FixedSizeInputStreamHandler by default.
//
class TfLiteImageInferenceCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status LoadModel(CalculatorContext* cc);
  absl::StatusOr<Packet> GetModelAsPacket(const CalculatorContext& cc);
  absl::Status ProcessInputsCpu(CalculatorContext* cc);
  absl::Status ProcessOutputsCpu(
      CalculatorContext* cc,
      std::unique_ptr<std::vector<TfLiteTensor>> output_tensors_cpu);

  absl::Status RunInContextIfNeeded(std::function<absl::Status(void)> f) {
    return f();
  }

  Packet model_packet_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
};
REGISTER_CALCULATOR(TfLiteImageInferenceCalculator);

absl::Status TfLiteImageInferenceCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kImageTag));
  RET_CHECK(cc->Outputs().HasTag(kTensorsTag));

  const auto& options =
      cc->Options<::mediapipe::TfLiteImageInferenceCalculatorOptions>();
  RET_CHECK(!options.model_path().empty())
      << "Either model as side packet or model path in options is required.";

  if (cc->Inputs().HasTag(kImageTag))
    cc->Inputs().Tag(kImageTag).Set<ImageFrame>();
  if (cc->Outputs().HasTag(kTensorsTag))
    cc->Outputs().Tag(kTensorsTag).Set<std::vector<TfLiteTensor>>();

  // Assign this calculator's default InputStreamHandler.
  cc->SetInputStreamHandler("FixedSizeInputStreamHandler");

  return absl::OkStatus();
}

absl::Status TfLiteImageInferenceCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  MP_RETURN_IF_ERROR(LoadModel(cc));
  return absl::OkStatus();
}

absl::Status TfLiteImageInferenceCalculator::Process(CalculatorContext* cc) {
  return RunInContextIfNeeded([this, cc]() -> absl::Status {
    auto output_tensors_cpu = absl::make_unique<std::vector<TfLiteTensor>>();

    MP_RETURN_IF_ERROR(ProcessInputsCpu(cc));

    RET_CHECK_EQ(interpreter_->Invoke(), kTfLiteOk);

    MP_RETURN_IF_ERROR(ProcessOutputsCpu(cc, std::move(output_tensors_cpu)));

    return absl::OkStatus();
  });
}

absl::Status TfLiteImageInferenceCalculator::Close(CalculatorContext* cc) {
  return RunInContextIfNeeded([this]() -> absl::Status {
    interpreter_ = nullptr;
    return absl::OkStatus();
  });
}

absl::Status TfLiteImageInferenceCalculator::ProcessInputsCpu(
    CalculatorContext* cc) {
  if (cc->Inputs().Tag(kImageTag).IsEmpty()) {
    return absl::OkStatus();
  }
  // Convert the input image into tensor.
  const auto& image_frame = cc->Inputs().Tag(kImageTag).Get<ImageFrame>();
  const int height = image_frame.Height();
  const int width = image_frame.Width();
  const int channels = image_frame.NumberOfChannels();
  const int channels_preserved = std::min(channels, 3);
  const mediapipe::ImageFormat::Format format = image_frame.Format();
  if (!(format == mediapipe::ImageFormat::SRGBA ||
        format == mediapipe::ImageFormat::SRGB ||
        format == mediapipe::ImageFormat::GRAY8 ||
        format == mediapipe::ImageFormat::VEC32F1))
    RET_CHECK_FAIL() << "Unsupported CPU input format.";

  const int tensor_idx = interpreter_->inputs()[0];
  TfLiteTensor* tensor = interpreter_->tensor(tensor_idx);

  if (cc->Options<mediapipe::TfLiteImageInferenceCalculatorOptions>()
          .add_batch_dim())
    interpreter_->ResizeInputTensor(tensor_idx,
                                    {1, height, width, channels_preserved});
  else
    interpreter_->ResizeInputTensor(tensor_idx,
                                    {height, width, channels_preserved});
  interpreter_->AllocateTensors();

  const int width_padding =
      image_frame.WidthStep() / image_frame.ByteDepth() - width * channels;
  const uint8_t* image_buffer =
      reinterpret_cast<const uint8_t*>(image_frame.PixelData());
  uint8_t* tensor_buffer = tensor->data.uint8;
  RET_CHECK(tensor_buffer);
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      for (int channel = 0; channel < channels_preserved; ++channel) {
        *tensor_buffer++ = image_buffer[channel];
      }
      image_buffer += channels;
    }
    image_buffer += width_padding;
  }

  return absl::OkStatus();
}

absl::Status TfLiteImageInferenceCalculator::ProcessOutputsCpu(
    CalculatorContext* cc,
    std::unique_ptr<std::vector<TfLiteTensor>> output_tensors_cpu) {
  const auto& tensor_indexes = interpreter_->outputs();
  for (int i = 0; i < tensor_indexes.size(); ++i) {
    TfLiteTensor* tensor = interpreter_->tensor(tensor_indexes[i]);
    output_tensors_cpu->emplace_back(*tensor);
  }
  cc->Outputs()
      .Tag(kTensorsTag)
      .Add(output_tensors_cpu.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

absl::Status TfLiteImageInferenceCalculator::LoadModel(CalculatorContext* cc) {
  ASSIGN_OR_RETURN(model_packet_, GetModelAsPacket(*cc));
  const auto& model = *model_packet_.Get<TfLiteModelPtr>();

  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates
      default_op_resolver;
  auto op_resolver_ptr =
      static_cast<const tflite::ops::builtin::BuiltinOpResolver*>(
          &default_op_resolver);

  tflite::InterpreterBuilder(model, *op_resolver_ptr)(&interpreter_);
  RET_CHECK(interpreter_);
  interpreter_->SetNumThreads(
      cc->Options<mediapipe::TfLiteImageInferenceCalculatorOptions>()
          .cpu_num_thread());

  RET_CHECK_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  return absl::OkStatus();
}

absl::StatusOr<Packet> TfLiteImageInferenceCalculator::GetModelAsPacket(
    const CalculatorContext& cc) {
  const auto& options =
      cc.Options<mediapipe::TfLiteImageInferenceCalculatorOptions>();
  if (!options.model_path().empty()) {
    return TfLiteModelLoader::LoadFromPath(options.model_path());
  }
  return absl::Status(absl::StatusCode::kNotFound,
                      "Must specify TFLite model path.");
}

}  // namespace mediapipe
