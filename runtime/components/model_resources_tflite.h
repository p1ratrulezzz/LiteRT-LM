// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_TFLITE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_TFLITE_H_

#include <memory>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/tokenizer.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"

namespace litert::lm {

// Model resources for a tflite model.
class ModelResourcesTflite : public ModelResources {
 public:
  static absl::StatusOr<std::unique_ptr<ModelResources>> Create(
      std::shared_ptr<ScopedFile> model_file);

  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      ModelType model_type) override;

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      ModelType model_type) override;

  absl::StatusOr<Tokenizer*> GetTokenizer() override;

  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override;

 private:
  explicit ModelResourcesTflite(
      std::unique_ptr<MemoryMappedFile> mapped_model_file);

  absl::Status InitLlmMetadata();

  std::unique_ptr<MemoryMappedFile> mapped_model_file_;
  proto::LlmMetadata llm_metadata_;
  Model model_;
  std::unique_ptr<Tokenizer> tokenizer_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_TFLITE_H_
