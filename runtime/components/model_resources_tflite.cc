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

#include "runtime/components/model_resources_tflite.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/tokenizer.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/metadata_util.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"

#ifdef ENABLE_SENTENCEPIECE_TOKENIZER
#include "runtime/components/sentencepiece_tokenizer.h"
#endif  // ENABLE_SENTENCEPIECE_TOKENIZER

namespace litert::lm {
namespace {
constexpr char kLlmParametersMetadataName[] = "odml.infra.proto.LlmParameters";
constexpr char kSpmVocabMetadataName[] = "spm_vocab_model";
}  // namespace

// static
absl::StatusOr<std::unique_ptr<ModelResources>> ModelResourcesTflite::Create(
    std::shared_ptr<ScopedFile> model_file) {
  ASSIGN_OR_RETURN(auto mapped_model_file,
                   MemoryMappedFile::Create(model_file->file()));
  auto model_resources =
      absl::WrapUnique(new ModelResourcesTflite(std::move(mapped_model_file)));
  RETURN_IF_ERROR(model_resources->InitLlmMetadata());
  return model_resources;
}

ModelResourcesTflite::ModelResourcesTflite(
    std::unique_ptr<MemoryMappedFile> mapped_model_file)
    : mapped_model_file_(std::move(mapped_model_file)) {}

absl::StatusOr<const litert::Model*> ModelResourcesTflite::GetTFLiteModel(
    ModelType model_type) {
  if (model_type != ModelType::kTfLitePrefillDecode) {
    return absl::NotFoundError(
        "Only prefill decode is supported for tflite models.");
  }
  return &model_;
}

absl::StatusOr<absl::string_view> ModelResourcesTflite::GetTFLiteModelBuffer(
    ModelType model_type) {
  if (model_type != ModelType::kTfLitePrefillDecode) {
    return absl::NotFoundError(
        "Only prefill decode is supported for tflite models.");
  }
  return absl::string_view(
      reinterpret_cast<const char*>(mapped_model_file_->data()),
      mapped_model_file_->length());
}

absl::StatusOr<Tokenizer*> ModelResourcesTflite::GetTokenizer() {
#ifdef ENABLE_SENTENCEPIECE_TOKENIZER
  if (!tokenizer_) {
    Expected<absl::Span<const uint8_t>> serialized_model =
        model_.Metadata(kSpmVocabMetadataName);
    RET_CHECK(serialized_model)
        << "Failed to get tokenizer model: " << serialized_model.Error();
    ASSIGN_OR_RETURN(
        tokenizer_, SentencePieceTokenizer::CreateFromBuffer(absl::string_view(
                        reinterpret_cast<const char*>(serialized_model->data()),
                        serialized_model->size())));
  }
  return tokenizer_.get();
#else
  return absl::UnimplementedError("Model does not contain a tokenizer.");
#endif  // ENABLE_SENTENCEPIECE_TOKENIZER
}

absl::StatusOr<const proto::LlmMetadata*>
ModelResourcesTflite::GetLlmMetadata() {
  return &llm_metadata_;
}

absl::Status ModelResourcesTflite::InitLlmMetadata() {
  ASSIGN_OR_RETURN(auto model_buffer,
                   GetTFLiteModelBuffer(ModelType::kTfLitePrefillDecode));
  BufferRef<uint8_t> buffer_ref(model_buffer.data(), model_buffer.size());
  LITERT_ASSIGN_OR_RETURN(model_, litert::Model::CreateFromBuffer(buffer_ref));

  LITERT_ASSIGN_OR_RETURN(absl::Span<const uint8_t> parameters,
                          model_.Metadata(kLlmParametersMetadataName));
  ASSIGN_OR_RETURN(llm_metadata_,
                   ExtractOrConvertLlmMetadata(std::string(
                       reinterpret_cast<const char*>(parameters.data()),
                       parameters.size())));
  return absl::OkStatus();
}

}  // namespace litert::lm
