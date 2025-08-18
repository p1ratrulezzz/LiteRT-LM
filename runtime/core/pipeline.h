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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_PIPELINE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_PIPELINE_H_

#include <stdbool.h>

#include <optional>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/sampler.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/components/tokenizer.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

// Runs the pipeline to prefill the input prompt.
// - executor: The executor that calls the core LLM model.
// - inputs: The inputs for the executor, containing the prompt and other
//   necessary data.
// - wait_for_completion: If true, wait for the prefill to complete before
//   returning.
// - benchmark_info: Optional benchmark info to record performance metrics.
// Returns the last token id of the prefill ids. It is used for
//   the next decode process to determine the token id to start from.
absl::StatusOr<int> Prefill(LlmExecutor& executor, ExecutorInputs& inputs,
                            bool wait_for_completion,
                            std::optional<BenchmarkInfo>& benchmark_info);

// Runs the pipeline to decode the input prompt.
// - executor: The executor that call the core LLM model.
// - tokenizer: The tokenizer to decode the token ids into text.
// - stop_token_ids: The token ids to stop the decoding process.
// - benchmark_info: The benchmark info to record the performance metrics.
absl::StatusOr<Responses> Decode(LlmExecutor& executor, Tokenizer& tokenizer,
                                 const StopTokenDetector& stop_token_detector,
                                 std::optional<BenchmarkInfo>& benchmark_info);

// Runs the pipeline to decode the input prompt. The function is similar to
// Decode, but it outputs the result using the observer to achieve streaming
// behavior.
// - observer: The inference observer to receive the intermediate results.
absl::Status DecodeStreaming(LlmExecutor& executor, Tokenizer& tokenizer,
                             const StopTokenDetector& stop_token_detector,
                             std::optional<BenchmarkInfo>& benchmark_info,
                             InferenceObservable* observer);

// Runs the pipeline to decode the input prompt.
// - executor: The executor that call the core LLM model.
// - tokenizer: The tokenizer to decode the token ids into text.
// - stop_token_ids: The token ids to stop the decoding process.
// - num_output_candidates: The number of output candidates to generate.
// - sampler: The sampler to sample the token ids from the logits.
// - decoded_ids: The decoded token ids from the external sampling process.
//   The supported shape is [num_output_candidates, 1].
// - benchmark_info: The benchmark info to record the performance metrics.
absl::StatusOr<Responses> DecodeCustomSampling(
    LlmExecutor& executor, Tokenizer& tokenizer,
    const StopTokenDetector& stop_token_detector, int num_output_candidates,
    Sampler& sampler, litert::TensorBuffer& decoded_ids,
    std::optional<BenchmarkInfo>& benchmark_info);

// Runs the pipeline to decode the input prompt. The function is similar to
// DecodeCustomSampling, but it outputs the result using the observer to achieve
// streaming behavior.
// - observer: The inference observer to receive the intermediate results.
absl::Status DecodeCustomSamplingStreaming(
    LlmExecutor& executor, Tokenizer& tokenizer,
    const StopTokenDetector& stop_token_detector, int num_output_candidates,
    Sampler& sampler, litert::TensorBuffer& decoded_ids,
    std::optional<BenchmarkInfo>& benchmark_info,
    InferenceObservable* observer);

// Runs the pipeline to score the input prompt.
// - executor: The executor that call the core LLM model.
// - tokenizer: The tokenizer to encode the text into token ids.
// - stop_token_ids: The token ids to stop the decoding process.
// - target_text: The vector of target texts to score across the batch.
// - sampler: The sampler to sample the token ids from the logits. This is
//   used to get the temperature and thus needs to be a TopPSampler.
// - decoded_ids: The decoded token ids from the external sampling process.
//   The supported shape is [num_output_candidates, 1].
// - benchmark_info: The benchmark info to record the performance metrics.
absl::StatusOr<std::vector<float>> ScoreCustomSampling(
    LlmExecutor& executor, Tokenizer& tokenizer,
    const StopTokenDetector& stop_token_detector,
    const std::vector<absl::string_view>& target_text, Sampler& sampler,
    litert::TensorBuffer& decoded_ids);
}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_PIPELINE_H_
