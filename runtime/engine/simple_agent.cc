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

// Simplified LLM agent that reads a prompt from prompt.txt, queries the model,
// and outputs the response to response.txt.

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "litert/c/litert_logging.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"

namespace {

using ::litert::lm::Backend;
using ::litert::lm::Engine;
using ::litert::lm::EngineSettings;
using ::litert::lm::InputData;
using ::litert::lm::InputText;
using ::litert::lm::ModelAssets;
using ::litert::lm::SessionConfig;

const absl::Duration kWaitUntilDoneTimeout = absl::Minutes(10);

// Reads the prompt from prompt.txt file.
std::string ReadPromptFromFile(const std::string& file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    ABSL_LOG(ERROR) << "Could not open prompt file: " << file_path;
    return "";
  }
  std::string prompt((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
  return prompt;
}

// Custom observer for streaming output
class StreamingObserver : public litert::lm::InferenceObservable {
 public:
  StreamingObserver(std::ofstream& file) : file_(file) {}

  // Override OnNext to write tokens incrementally
  void OnNext(const litert::lm::Responses& responses) override {
    if (file_.is_open()) {
      // Get clean text without metadata
      auto text_or = responses.GetResponseTextAt(0);
      if (text_or.ok()) {
        file_ << *text_or;
      }
      file_.flush();  // Flush to write immediately
    }
  }

  // Override OnCompleted to add newline at the end
  void OnCompleted() {
    if (file_.is_open()) {
      file_ << std::endl;
      file_.flush();
    }
  }

  // Override OnError for error handling
  void OnError(const absl::Status& status) override {
    ABSL_LOG(ERROR) << "Streaming error: " << status;
  }

 private:
  std::ofstream& file_;
};

absl::Status RunInference(Engine* llm, Engine::Session* session,
                         const std::string& input_prompt, bool async = false) {
  std::vector<InputData> inputs;
  inputs.emplace_back(InputText(input_prompt));

  std::ofstream out_file("response.txt");
  if (!out_file) {
    ABSL_LOG(ERROR) << "Could not open response file: response.txt";
    return absl::InternalError("Failed to open response file");
  }

  if (async) {
    // Use streaming for incremental output
    StreamingObserver observer(out_file);

    absl::Status status = session->GenerateContentStream(inputs, &observer);
    ABSL_CHECK_OK(status);
    ABSL_CHECK_OK(llm->WaitUntilDone(kWaitUntilDoneTimeout));
  } else {
    auto responses = session->GenerateContent(inputs);
    ABSL_CHECK_OK(responses);

    out_file << *responses << std::endl;
  }

  out_file.close();

  return absl::OkStatus();
}

absl::Status MainHelper(int argc, char** argv) {
  // Set minimal log level to INFO
  LiteRtSetMinLoggerSeverity(LiteRtGetDefaultLogger(), LITERT_INFO);

  const std::string default_model_path = "../models/gemma3-1b-it-int4.litertlm";
  std::string model_path = (argc >= 2) ? argv[1] : default_model_path;

  const std::string prompt_file = "./prompt.txt";
  std::string input_prompt = ReadPromptFromFile(prompt_file);
  if (input_prompt.empty()) {
    return absl::InvalidArgumentError("Prompt file is empty or not found.");
  }

  ABSL_LOG(INFO) << "Model path: " << model_path;

  // Check if model file exists
  std::ifstream model_file(model_path);
  if (!model_file.good()) {
    ABSL_LOG(ERROR) << "Model file does not exist or is not accessible: " << model_path;
    return absl::InvalidArgumentError("Model file not found");
  }
  model_file.close();

  ASSIGN_OR_RETURN(ModelAssets model_assets,
                   ModelAssets::Create(model_path));
  ABSL_LOG(INFO) << "Model loaded successfully from: " << model_path;

  // Use CPU backend as default
  Backend backend = Backend::CPU;
  ABSL_LOG(INFO) << "Using backend: CPU";

  ASSIGN_OR_RETURN(EngineSettings engine_settings,
                   EngineSettings::CreateDefault(std::move(model_assets), backend));

  ABSL_LOG(INFO) << "Creating engine";
  absl::StatusOr<std::unique_ptr<Engine>> llm =
      Engine::CreateEngine(std::move(engine_settings));
  ABSL_CHECK_OK(llm) << "Failed to create engine";
  ABSL_LOG(INFO) << "Engine created successfully";

  ABSL_LOG(INFO) << "Creating session";
  SessionConfig session_config = SessionConfig::CreateDefault();
  absl::StatusOr<std::unique_ptr<Engine::Session>> session =
      (*llm)->CreateSession(session_config);
  ABSL_CHECK_OK(session) << "Failed to create session";
  ABSL_LOG(INFO) << "Session created successfully";

  // Run inference with streaming
  RETURN_IF_ERROR(RunInference(llm->get(), session->get(), input_prompt, /*async=*/true));

  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  ABSL_CHECK_OK(MainHelper(argc, argv));
  return 0;
}