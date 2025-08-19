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

#include "runtime/core/session_basic.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/components/tokenizer.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/fake_llm_executor.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/framework/threadpool.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

constexpr char kTestdataDir[] =
    "litert_lm/runtime/components/testdata/";

class SessionBasicTest : public testing::Test {
 protected:
  void SetUp() override {
    auto tokenizer = SentencePieceTokenizer::CreateFromFile(
        (std::filesystem::path(::testing::SrcDir()) / kTestdataDir /
         "sentencepiece.model")
            .string());
    ASSERT_OK(tokenizer);
    tokenizer_ = std::move(*tokenizer);
    // The prefill tokens are the expected tokens that will be passed in at each
    // time the Prefill function is called. The values are the token ids of the
    // input prompt "Hello World!".
    std::vector<std::vector<int>> prefill_tokens = {
        {2, 90, 547, 58, 735, 210, 466, 2294}};
    // The decode tokens are the expected tokens that will be returned by the
    // Decode function. The values are the token ids of the output response
    // "How's it going?" followed by the stop token id (2294).
    std::vector<std::vector<int>> decode_tokens = {{224}, {24}, {8},    {66},
                                                   {246}, {18}, {2295}, {2294}};
    executor_ =
        std::make_unique<FakeLlmExecutor>(2560, prefill_tokens, decode_tokens);

    sampler_params_.set_type(proto::SamplerParameters::TYPE_UNSPECIFIED);

    // Creating the thread pool of a single thread to execute the works.
    worker_thread_pool_ = std::make_unique<ThreadPool>(/*name_prefix=*/"engine",
                                                       /*max_num_threads=*/1);
  }

  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<LlmExecutor> executor_;
  proto::SamplerParameters sampler_params_;
  std::unique_ptr<ThreadPool> worker_thread_pool_;
};

TEST_F(SessionBasicTest, RunPrefill) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  auto session = SessionBasic::Create(
      executor_.get(), tokenizer_.get(),
      /*image_preprocessor=*/nullptr,
      /*vision_executor=*/nullptr, session_config,
      /*benchmark_info=*/std::nullopt, worker_thread_pool_.get());
  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));
  EXPECT_OK((*session)->RunPrefill(inputs));
}

TEST_F(SessionBasicTest, RunDecode) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  auto session =
      SessionBasic::Create(executor_.get(), tokenizer_.get(),
                           /*image_preprocessor=*/nullptr,
                           /*vision_executor=*/nullptr, session_config,
                           std::nullopt, worker_thread_pool_.get());
  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));
  EXPECT_OK((*session)->RunPrefill(inputs));
  auto responses = (*session)->RunDecode();
  EXPECT_OK(responses);
  EXPECT_EQ(responses->GetNumOutputCandidates(), 1);
  // The response is " How's it going?" since "!" is the stop token which is
  // not included in the response.
  EXPECT_EQ(*(responses->GetResponseTextAt(0)), " How's it going?");
}

class TestObserver : public InferenceObservable {
 public:
  void OnDone() override { done_ = true; }

  bool IsDone() { return done_; }

 private:
  bool done_ = false;
};

TEST_F(SessionBasicTest, RunPrefillAsync) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.SetStartTokenId(2);
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetSamplerBackend(Backend::CPU);
  auto session =
      SessionBasic::Create(executor_.get(), tokenizer_.get(),
                           /*image_preprocessor=*/nullptr,
                           /*vision_executor=*/nullptr, session_config,
                           std::nullopt, worker_thread_pool_.get());

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));
  TestObserver observer;
  EXPECT_OK((*session)->RunPrefillAsync(inputs, &observer));
  // Wait for the async call to finish.
  EXPECT_OK(worker_thread_pool_->WaitUntilDone(absl::Seconds(100)));
  EXPECT_TRUE(observer.IsDone());
}

TEST_F(SessionBasicTest, RunDecodeAsync) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.SetStartTokenId(2);
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetSamplerBackend(Backend::CPU);
  auto session =
      SessionBasic::Create(executor_.get(), tokenizer_.get(),
                           /*image_preprocessor=*/nullptr,
                           /*vision_executor=*/nullptr, session_config,
                           std::nullopt, worker_thread_pool_.get());

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));
  TestObserver observer;
  EXPECT_OK((*session)->RunPrefillAsync(inputs, &observer));
  EXPECT_OK((*session)->RunDecodeAsync(&observer));
  EXPECT_OK(worker_thread_pool_->WaitUntilDone(absl::Seconds(100)));
  EXPECT_TRUE(observer.IsDone());
}

class StreamingTestObserver : public InferenceObservable {
 public:
  void OnNext(const Responses& responses) override {
    ASSERT_EQ(responses.GetNumOutputCandidates(), 1);
    texts_.push_back(std::string(*responses.GetResponseTextAt(0)));
  }

  void OnError(const absl::Status& status) override {
    status_ = status;
    done_.Notify();
  }

  void OnDone() override { done_.Notify(); }

  absl::Status WaitUntilDone() {
    done_.WaitForNotificationWithTimeout(absl::Seconds(10));
    return status_;
  }

  std::vector<std::string> GetTexts() { return texts_; }

 private:
  absl::Notification done_;
  absl::Status status_;
  std::vector<std::string> texts_;
};

TEST_F(SessionBasicTest, GenerateContentStream) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  auto session =
      SessionBasic::Create(executor_.get(), tokenizer_.get(),
                           /*image_preprocessor=*/nullptr,
                           /*vision_executor=*/nullptr,
                           session_config,
                           std::nullopt, worker_thread_pool_.get());

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));
  StreamingTestObserver observer;
  EXPECT_OK((*session)->GenerateContentStream(inputs, &observer));

  EXPECT_OK(observer.WaitUntilDone());
  EXPECT_THAT(observer.GetTexts(),
              testing::ElementsAre(" How", "'", "s", " it", " go", "ing", "?"));
}

TEST_F(SessionBasicTest, GenerateContentStreamEmptyInput) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  auto session =
      SessionBasic::Create(executor_.get(), tokenizer_.get(),
                           /*image_preprocessor=*/nullptr,
                           /*vision_executor=*/nullptr,
                           session_config,
                           std::nullopt, worker_thread_pool_.get());

  std::vector<InputData> inputs;
  StreamingTestObserver observer;
  EXPECT_THAT((*session)->GenerateContentStream(inputs, &observer),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument,
                                        "Input is empty."));
}

TEST_F(SessionBasicTest, GenerateContentStreamPrefillError) {
  // Configure the executor to fail at prefill.
  auto* fake_executor = static_cast<FakeLlmExecutor*>(executor_.get());
  fake_executor->SetPrefillStatus(absl::InternalError("Prefill failed"));

  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  auto session =
      SessionBasic::Create(executor_.get(), tokenizer_.get(),
                           /*image_preprocessor=*/nullptr,
                           /*vision_executor=*/nullptr,
                           session_config,
                           std::nullopt, worker_thread_pool_.get());

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));
  StreamingTestObserver observer;
  EXPECT_OK((*session)->GenerateContentStream(inputs, &observer));

  absl::Status status = observer.WaitUntilDone();
  EXPECT_THAT(status, testing::status::StatusIs(absl::StatusCode::kInternal,
                                                "Prefill failed"));
}

TEST_F(SessionBasicTest, GenerateContentStreamDecodeError) {
  // Configure the executor to fail at decode.
  auto* fake_executor = static_cast<FakeLlmExecutor*>(executor_.get());
  fake_executor->SetDecodeStatus(absl::InternalError("Decode failed"));

  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  auto session =
      SessionBasic::Create(executor_.get(), tokenizer_.get(),
                           /*image_preprocessor=*/nullptr,
                           /*vision_executor=*/nullptr,
                           session_config,
                           std::nullopt, worker_thread_pool_.get());

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));
  StreamingTestObserver observer;
  EXPECT_OK((*session)->GenerateContentStream(inputs, &observer));

  absl::Status status = observer.WaitUntilDone();
  EXPECT_THAT(status, testing::status::StatusIs(absl::StatusCode::kInternal,
                                                "Decode failed"));
}

TEST_F(SessionBasicTest, ApplyPromptTemplates) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  session_config.GetMutablePromptTemplates().mutable_user()->set_prefix(
      "<test>User\n");
  session_config.GetMutablePromptTemplates().mutable_user()->set_suffix(
      "<end>\n");
  session_config.GetMutablePromptTemplates().mutable_model()->set_prefix(
      "<test>Model\n");
  auto session_or =
      SessionBasic::Create(executor_.get(), tokenizer_.get(),
                           /*image_preprocessor=*/nullptr,
                           /*vision_executor=*/nullptr, session_config,
                           std::nullopt, worker_thread_pool_.get());
  ASSERT_OK(session_or);
  auto session = std::move(*session_or);
  // First chunk of the first turn will have the BOS token.
  EXPECT_THAT(
      session->ApplyPromptTemplates("Hello World!", /*is_first_chunk=*/true,
                                    /*is_last_chunk=*/false),
      testing::status::IsOkAndHolds("</s><test>User\nHello World!"));
  // Both prefix and suffix will be added.
  EXPECT_THAT(
      session->ApplyPromptTemplates("Hello World!", /*is_first_chunk=*/true,
                                    /*is_last_chunk=*/true),
      testing::status::IsOkAndHolds(
          "\n<test>User\nHello World!<end>\n<test>Model\n"));
  // Only suffix will be added.
  EXPECT_THAT(
      session->ApplyPromptTemplates("Hello World!", /*is_first_chunk=*/false,
                                    /*is_last_chunk=*/true),
      testing::status::IsOkAndHolds("Hello World!<end>\n<test>Model\n"));
}

}  // namespace
}  // namespace litert::lm
