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

#include "runtime/core/pipeline.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/sampler.h"
#include "runtime/components/scoring_cpu_util.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/components/tokenizer.h"
#include "runtime/components/top_p_cpu_sampler.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/litert_status_util.h"
#include "runtime/util/status_macros.h"  //NOLINT

namespace litert::lm {
namespace {

// TODO(b/423364170): all LLM Executors should respect the max number of tokens
// returned by the model. We should remove this default value once all Executors
// are compliant with the max number of tokens.
constexpr int kDefaultMaxNumTokens = 4096;
int TryGetMaxNumTokens(const LlmExecutor& executor) {
  auto settings = executor.GetExecutorSettings();
  if (!settings.ok()) {
    // If the executor settings are not available, we will use the default
    // value.
    ABSL_LOG(WARNING) << "Failed to get executor settings: "
                      << settings.status();
    return kDefaultMaxNumTokens;
  }
  return settings->GetMaxNumTokens();
}

// Check whether the decoding loop should stop.
bool ShouldStop(bool hit_stop_tokens, int benchmark_decode_token_count,
                int num_decoded_steps, int current_step, int max_num_tokens,
                InferenceObservable* observer) {
  // Stopping conditions.
  if (hit_stop_tokens && benchmark_decode_token_count == 0) {
    // Only early stop if no decode step
    // is requested by benchmark.
    return true;
  } else if (benchmark_decode_token_count > 0 &&
             num_decoded_steps >= benchmark_decode_token_count) {
    // Stop when the number of decode steps is equal to the
    // benchmark_decode_token_count (when specified).
    return true;
  } else if (current_step >= max_num_tokens) {
    // Reaching maximum number of kv-cache size.
    if (observer != nullptr) {
      observer->OnError(absl::InternalError("Maximum kv-cache size reached."));
    }
    return true;
  }
  return false;
}

// A wrapper class to run one step of the decode process, handling both internal
// and external sampling.
class DecodeOneStep {
 public:
  DecodeOneStep(LlmExecutor* absl_nonnull executor,
                Tokenizer* absl_nonnull tokenizer, int num_output_candidates,
                const StopTokenDetector& stop_token_detector,
                std::optional<BenchmarkInfo>& benchmark_info,
                std::optional<Sampler*> sampler)
      : executor_(*executor),
        tokenizer_(*tokenizer),
        num_output_candidates_(num_output_candidates),
        sampler_(sampler),
        benchmark_info_(benchmark_info),
        stop_token_detector_(stop_token_detector) {
    if (!sampler_.has_value()) {  // Internal sampling setup
      auto output_tokens = CreateTensorBuffer<int>({num_output_candidates_, 1});
      output_tokens_ = std::move(*output_tokens);
    } else {  // External sampling setup
      auto scores_tensor = CreateTensorBuffer<float>({num_output_candidates_});
      scores_tensor_ = std::move(*scores_tensor);
    }
    result_text_ = std::vector<std::string>(num_output_candidates_, "");
    bpe_partial_token_ids_ =
        std::vector<std::vector<int>>(num_output_candidates_);
    pending_stop_tokens_ =
        std::vector<std::queue<std::string>>(num_output_candidates_);
  }

  // Runs one step of the decode process and returns if all stops for all
  // candidates have been found.
  // For external sampling, `decoded_ids` must be provided and will be updated.
  // For internal sampling, `decoded_ids` is ignored.
  absl::StatusOr<bool> Run(
      std::optional<litert::TensorBuffer*> decoded_ids = std::nullopt) {
    ASSIGN_OR_RETURN(litert::TensorBuffer * next_tokens_buffer,
                     DecodeAndSample(decoded_ids));

    // Post-processing the next tokens.
    ASSIGN_OR_RETURN(auto token_ids,
                     tokenizer_.TensorBufferToTokenIds(*next_tokens_buffer));

    // Merge BPE partial token ids with the next token ids if any.
    ASSIGN_OR_RETURN(
        token_ids, tokenizer_.MergeTokenIds(bpe_partial_token_ids_, token_ids));

    // Regardless of BPE, we always process the next tokens to detect stop
    // tokens.
    LITERT_ASSIGN_OR_RETURN_ABSL(
        auto next_tokens_span,
        ReferTensorBufferAsSpan<int>(*next_tokens_buffer));
    RETURN_IF_ERROR(stop_token_detector_.ProcessTokens(next_tokens_span));

    auto decoded_result =
        tokenizer_.TokenIdsToTexts(num_output_candidates_, token_ids);
    for (int i = 0; i < num_output_candidates_; ++i) {
      result_text_[i] = "";
      if (Tokenizer::IsIncompleteBpeSequence(decoded_result.value()[i])) {
        bpe_partial_token_ids_[i] = token_ids[i];
      } else if (!stop_token_detector_.GetStopTokensFound()[i]) {
        bpe_partial_token_ids_[i].clear();

        // Handle partial stop tokens.
        int max_length = stop_token_detector_.MaxPartialStopTokenLength(i);
        if (max_length > 0) {
          pending_stop_tokens_[i].push(decoded_result.value()[i].value());
        }
        // We only need the latest max_length tokens for partial stop tokens.
        // Add the extra ones to the result text tand we could keep only the
        // latest max_length stop tokens in the queue.
        while (pending_stop_tokens_[i].size() > max_length) {
          result_text_[i] += pending_stop_tokens_[i].front();
          pending_stop_tokens_[i].pop();
        }

        // No partial stop token is found - add the current token to the result
        // text directly - this is the most common case.
        if (max_length == 0) {
          result_text_[i] += decoded_result.value()[i].value();
        }
      }
    }

    if (sampler_.has_value()) {
      LITERT_ASSIGN_OR_RETURN_ABSL(
          scores_span_, ReferTensorBufferAsSpan<float>(scores_tensor_));
    }

    return stop_token_detector_.AllDone();
  }

  absl::Span<float> GetScores() { return scores_span_; }

  const std::vector<std::string>& GetResultText() const { return result_text_; }

  absl::StatusOr<std::vector<float>> computeConfidencesForBatch(
      const std::vector<int> sampled_id, litert::TensorBuffer* decoded_ids) {
    if (sampler_.has_value()) {
      TopPSampler* top_p_sampler = dynamic_cast<TopPSampler*>(sampler_.value());
      if (top_p_sampler == nullptr) {
        return absl::InternalError(
            "getBatchPerplexity is only supported for external sampling.");
      }
      // Overwrite the decoded_ids with the text id for the input text.
      LITERT_ASSIGN_OR_RETURN(auto duplicate_decoded_ids,
                              decoded_ids->Duplicate());
      ExecutorInputs inputs(ExecutorTextData(std::move(duplicate_decoded_ids)),
                            std::nullopt, std::nullopt);
      // Decoding section.
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("executor_decode"));
      }
      ASSIGN_OR_RETURN(auto output_logits, executor_.DecodeLogits(inputs));
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("executor_decode"));
      }
      float temperature = top_p_sampler->getTemperature();
      decoded_ids->Write(absl::MakeConstSpan(sampled_id));

      auto logits_data_or = ReferTensorBufferAsSpan<float>(output_logits);
      absl::Span<float> logits_data;
      std::vector<float> logits_data_buffer;
      if (!logits_data_or) {  // Download the data if it is not in host memory.
        LITERT_ASSIGN_OR_RETURN(auto logits_size, output_logits.PackedSize());
        if (logits_data_buffer.size() != logits_size / sizeof(float)) {
          logits_data_buffer = std::vector<float>(logits_size / sizeof(float));
        }
        TensorBuffer& mutable_logits_tensor =
            const_cast<TensorBuffer&>(output_logits);
        mutable_logits_tensor.Read(absl::MakeSpan(logits_data_buffer));
        logits_data = absl::MakeSpan(logits_data_buffer);
      } else {
        logits_data = logits_data_or.Value();
      }
      return ComputeBatchConfidences(logits_data, sampled_id, temperature);
    } else {
      return absl::InternalError(
          "getBatchPerplexity is only supported for external sampling.");
    }
  }

 private:
  // Runs the core decoding and sampling step, for either internal or external
  // sampling. Returns a pointer to the tensor buffer containing the next token
  // IDs.
  absl::StatusOr<litert::TensorBuffer*> DecodeAndSample(
      std::optional<litert::TensorBuffer*> decoded_ids) {
    if (sampler_) {  // External sampling path
      if (!decoded_ids) {
        return absl::InternalError(
            "decoded_ids must be provided for external sampling.");
      }
      LITERT_ASSIGN_OR_RETURN(auto duplicate_decoded_ids,
                              decoded_ids.value()->Duplicate());
      ExecutorInputs inputs(ExecutorTextData(std::move(duplicate_decoded_ids)),
                            std::nullopt, std::nullopt);
      // Decoding section.
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("executor_decode"));
      }
      ASSIGN_OR_RETURN(auto output_logits, executor_.DecodeLogits(inputs));
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("executor_decode"));
      }

      // Samping section.
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("sampling"));
      }
      RETURN_IF_ERROR(sampler_.value()->SampleToIdAndScoreBuffer(
          output_logits, *decoded_ids.value(), &scores_tensor_));
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("sampling"));
      }

      return decoded_ids.value();
    } else {  // Internal sampling path
      // Benchmark executor_decode_and_sample section.
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(
            benchmark_info_->TimeMarkDelta("executor_decode_and_sample"));
      }
      RETURN_IF_ERROR(executor_.Decode(output_tokens_));
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(
            benchmark_info_->TimeMarkDelta("executor_decode_and_sample"));
      }
      return &output_tokens_;
    }
  }

  LlmExecutor& executor_;
  Tokenizer& tokenizer_;
  const int num_output_candidates_;
  std::optional<Sampler*> sampler_;
  std::optional<BenchmarkInfo> benchmark_info_;
  StopTokenDetector stop_token_detector_;

  // For internal sampling.
  // Holds the output token IDs. Dim: {num_output_candidates, 1}
  litert::TensorBuffer output_tokens_;

  // For external sampling.
  // Holds the scores for the output candidates. Dim: {num_output_candidates}
  litert::TensorBuffer scores_tensor_;
  absl::Span<float> scores_span_;

  // Common state
  std::vector<std::vector<int>> bpe_partial_token_ids_;
  std::vector<std::queue<std::string>> pending_stop_tokens_;
  std::vector<std::string> result_text_;
};

absl::StatusOr<Responses> DecodeLoop(
    LlmExecutor& executor, Tokenizer& tokenizer,
    const StopTokenDetector& stop_token_detector, int num_output_candidates,
    std::optional<BenchmarkInfo>& benchmark_info,
    std::optional<Sampler*> sampler,
    std::optional<litert::TensorBuffer*> decoded_ids,
    std::optional<InferenceObservable*> observer) {
  const bool is_streaming = observer.has_value();
  const bool is_custom_sampling = sampler.has_value();

  int benchmark_decode_token_count = 0;
  if (benchmark_info.has_value()) {
    benchmark_decode_token_count =
        benchmark_info->GetBenchmarkParams().num_decode_tokens();
    RETURN_IF_ERROR(benchmark_info->TimeDecodeTurnStart());
  }

  Responses final_responses(num_output_candidates);
  std::vector<float> accumulated_scores(num_output_candidates, 0.0f);
  std::vector<int> num_decoded_tokens(num_output_candidates, 0);

  int num_decode_steps = 0;
  const int max_num_tokens = TryGetMaxNumTokens(executor);
  DecodeOneStep run_one_step(&executor, &tokenizer, num_output_candidates,
                             stop_token_detector, benchmark_info, sampler);
  while (true) {
    absl::StatusOr<bool> all_done = run_one_step.Run(decoded_ids);
    if (!all_done.ok()) {
      if (is_streaming) observer.value()->OnError(all_done.status());
      return all_done.status();
    }
    num_decode_steps++;
    Responses step_responses(num_output_candidates);
    bool any_updates = false;
    for (int j = 0; j < num_output_candidates; ++j) {
      std::string output_text = run_one_step.GetResultText()[j];
      if (output_text.empty()) {
        // No output text for this candidate - could be due to
        // 1. early stopping.
        // 2. partial BPE sequence.
        // 3. matching partial stop tokens.
        continue;
      }
      any_updates = true;
      // The tokenizer may return a token with a special character "▁" that
      // should be replaced with a space.
      std::string result_text = absl::StrReplaceAll(output_text, {{"▁", " "}});
      if (is_streaming) {
        step_responses.GetMutableResponseTexts()[j] = result_text;
        if (is_custom_sampling) {
          step_responses.GetMutableScores()[j] = run_one_step.GetScores()[j];
        }
      } else {
        final_responses.GetMutableResponseTexts()[j] += result_text;
        if (is_custom_sampling) {
          accumulated_scores[j] += run_one_step.GetScores()[j];
          num_decoded_tokens[j]++;
        }
      }
    }

    if (is_streaming && any_updates && !*all_done) {
      observer.value()->OnNext(step_responses);
    }

    if (ShouldStop(*all_done, benchmark_decode_token_count, num_decode_steps,
                   executor.GetCurrentStep().value(), max_num_tokens,
                   observer.value_or(nullptr))) {
      break;
    }
  }

  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(benchmark_info->TimeDecodeTurnEnd(num_decode_steps *
                                                      num_output_candidates));
  }

  if (is_streaming) {
    observer.value()->OnDone();
    return Responses(0);  // Return empty response for streaming.
  }

  // Finalize scores for non-streaming custom sampling.
  if (is_custom_sampling) {
    std::vector<float>& final_scores = final_responses.GetMutableScores();
    for (int j = 0; j < num_output_candidates; ++j) {
      if (num_decoded_tokens[j] > 0) {
        final_scores[j] = accumulated_scores[j] / num_decoded_tokens[j];
      } else {
        final_scores[j] = -std::numeric_limits<float>::infinity();
      }
    }
  }
  return final_responses;
}

}  // namespace

absl::StatusOr<std::vector<float>> ScoreCustomSampling(
    LlmExecutor& executor, Tokenizer& tokenizer,
    const StopTokenDetector& stop_token_detector,
    const std::vector<absl::string_view>& target_texts, Sampler& sampler,
    litert::TensorBuffer& decoded_ids) {
  TopPSampler* top_p_sampler = dynamic_cast<TopPSampler*>(&sampler);
  if (top_p_sampler == nullptr) {
    return absl::InternalError("ScoreCustomSampling requires a TopPSampler.");
  } else if (top_p_sampler->getBatchSize() != target_texts.size()) {
    return absl::InvalidArgumentError(
        "ScoreLoop batch size must match the target text size.");
  }
  int num_output_candidates = top_p_sampler->getBatchSize();
  const int max_num_tokens = TryGetMaxNumTokens(executor);
  std::optional<BenchmarkInfo> benchmark_info = std::nullopt;
  DecodeOneStep run_one_step(&executor, &tokenizer,
                             /*num_output_candidates=*/num_output_candidates,
                             stop_token_detector, benchmark_info, &sampler);
  std::vector<std::vector<int>> ids_for_each_target_in_batch;
  int max_num_tokens_for_each_target_in_batch = 0;
  for (const auto& target : target_texts) {
    ASSIGN_OR_RETURN(std::vector<int> ids, tokenizer.TextToTokenIds(target));
    max_num_tokens_for_each_target_in_batch =
        std::max(max_num_tokens_for_each_target_in_batch, (int)ids.size());
    ids_for_each_target_in_batch.push_back(ids);
  }
  if (max_num_tokens_for_each_target_in_batch >= max_num_tokens) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Input token ids are too long. "
        "Exceeding the maximum number of tokens allowed: ",
        max_num_tokens_for_each_target_in_batch, " >= ", max_num_tokens));
  }
  std::vector<float> step_neg_log_likelihood_sum =
      std::vector<float>(num_output_candidates, 0.0f);
  // TODO(ritviksharma): This is a temporary solution that only supports one
  // output candidate. We need to refactor this to support multiple targets
  // which will require padding the targets with a null token which does not
  // exist in the vocabulary and thus does not contribute to the perplexity.
  std::vector<int> decoded_ids_for_each_target_in_batch =
      std::vector<int>(ids_for_each_target_in_batch.size(), 0);
  for (int i = 0; i < max_num_tokens_for_each_target_in_batch; ++i) {
    for (int j = 0; j < ids_for_each_target_in_batch.size(); ++j) {
      if (i < ids_for_each_target_in_batch[j].size()) {
        decoded_ids_for_each_target_in_batch[j] =
            ids_for_each_target_in_batch[j][i];
      } else {
        decoded_ids_for_each_target_in_batch[j] = -1;  // Padding token.
      }
    }
    auto step_neg_log_likelihood = run_one_step.computeConfidencesForBatch(
        decoded_ids_for_each_target_in_batch, &decoded_ids);
    if (!step_neg_log_likelihood.ok()) {
      return step_neg_log_likelihood.status();
    }
    for (int j = 0; j < num_output_candidates; ++j) {
      step_neg_log_likelihood_sum[j] += step_neg_log_likelihood.value()[j];
    }
  }
  return step_neg_log_likelihood_sum;
}

absl::StatusOr<int> Prefill(LlmExecutor& executor, ExecutorInputs& inputs,
                            bool wait_for_completion,
                            std::optional<BenchmarkInfo>& benchmark_info) {
  const int max_num_tokens = TryGetMaxNumTokens(executor);
  ASSIGN_OR_RETURN(auto text_data, inputs.GetTextDataPtr());
  RET_CHECK(text_data != nullptr) << "text_data must not be null.";
  LITERT_ASSIGN_OR_RETURN_ABSL(auto token_id_tensor_type,
                               text_data->GetTokenIds().TensorType());
  auto num_tokens = token_id_tensor_type.Layout().Dimensions().back();
  if (num_tokens >= max_num_tokens) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Input token ids are too long. Exceeding the maximum number of tokens "
        "allowed: ",
        num_tokens, " >= ", max_num_tokens));
  }
  LITERT_ASSIGN_OR_RETURN_ABSL(
      auto ids_buffer_span,
      ReferTensorBufferAsSpan<int>(text_data->GetTokenIds()));
  if (ids_buffer_span.empty()) {
    return absl::InternalError("Input token ids are empty.");
  }
  const int last_token_id = ids_buffer_span.back();
  ExecutorPrefillParams params;
  params.SetWaitForCompletion(wait_for_completion);
  RETURN_IF_ERROR(executor.Prefill(inputs, params));
  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(benchmark_info->TimePrefillTurnEnd(ids_buffer_span.size()));
  }
  return last_token_id;
}

absl::StatusOr<Responses> Decode(LlmExecutor& executor, Tokenizer& tokenizer,
                                 const StopTokenDetector& stop_token_detector,
                                 std::optional<BenchmarkInfo>& benchmark_info) {
  const int num_output_candidates = 1;
  return DecodeLoop(executor, tokenizer, stop_token_detector,
                    num_output_candidates, benchmark_info, std::nullopt,
                    std::nullopt, std::nullopt);
}

absl::Status DecodeStreaming(LlmExecutor& executor, Tokenizer& tokenizer,
                             const StopTokenDetector& stop_token_detector,
                             std::optional<BenchmarkInfo>& benchmark_info,
                             InferenceObservable* observer) {
  if (observer == nullptr) {
    return absl::InvalidArgumentError(
        "Observer must not be null for streaming.");
  }
  const int num_output_candidates = 1;
  return DecodeLoop(executor, tokenizer, stop_token_detector,
                    num_output_candidates, benchmark_info, std::nullopt,
                    std::nullopt, observer)
      .status();
}

absl::StatusOr<Responses> DecodeCustomSampling(
    LlmExecutor& executor, Tokenizer& tokenizer,
    const StopTokenDetector& stop_token_detector, int num_output_candidates,
    Sampler& sampler, litert::TensorBuffer& decoded_ids,
    std::optional<BenchmarkInfo>& benchmark_info) {
  return DecodeLoop(executor, tokenizer, stop_token_detector,
                    num_output_candidates, benchmark_info, &sampler,
                    &decoded_ids, std::nullopt);
}

absl::Status DecodeCustomSamplingStreaming(
    LlmExecutor& executor, Tokenizer& tokenizer,
    const StopTokenDetector& stop_token_detector, int num_output_candidates,
    Sampler& sampler, litert::TensorBuffer& decoded_ids,
    std::optional<BenchmarkInfo>& benchmark_info,
    InferenceObservable* observer) {
  if (observer == nullptr) {
    return absl::InvalidArgumentError(
        "Observer must not be null for streaming.");
  }
  return DecodeLoop(executor, tokenizer, stop_token_detector,
                    num_output_candidates, benchmark_info, &sampler,
                    &decoded_ids, observer)
      .status();
}

}  // namespace litert::lm
