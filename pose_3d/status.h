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


#ifndef MEDIAPIPE_DEPS_STATUS_H_
#define MEDIAPIPE_DEPS_STATUS_H_

#include <functional>
#include <iosfwd>
#include <memory>
#include <string>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "glog/logging.h"

namespace mouse_pose {

using Status ABSL_DEPRECATED("Use absl::Status directly") = absl::Status;
using StatusCode ABSL_DEPRECATED("Use absl::StatusCode directly") =
    absl::StatusCode;

ABSL_DEPRECATED("Use absl::OkStatus directly")
inline absl::Status OkStatus() { return absl::OkStatus(); }

extern std::string* CheckOpHelperOutOfLine(const absl::Status& v,
                                           const char* msg);

inline std::string* CheckOpHelper(absl::Status v, const char* msg) {
  if (v.ok()) return nullptr;
  return CheckOpHelperOutOfLine(v, msg);
}

#define DO_CHECK_OK(val, level)                                \
  while (auto _result = mouse_pose::CheckOpHelper(val, #val)) \
  LOG(level) << *(_result)

#define CHECK_OK(val) DO_CHECK_OK(val, FATAL)
#define QCHECK_OK(val) DO_CHECK_OK(val, FATAL)

#ifndef NDEBUG
#define DCHECK_OK(val) CHECK_OK(val)
#else
#define DCHECK_OK(val) \
  while (false && (absl::OkStatus() == (val))) LOG(FATAL)
#endif

}  // namespace mouse_pose

#endif
