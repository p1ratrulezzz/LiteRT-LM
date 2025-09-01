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
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>


namespace {

using ::litert::lm::Backend;
using ::litert::lm::Engine;
using ::litert::lm::EngineSettings;
using ::litert::lm::InputData;
using ::litert::lm::InputText;
using ::litert::lm::ModelAssets;
using ::litert::lm::SessionConfig;

const absl::Duration kWaitUntilDoneTimeout = absl::Minutes(5);

// Custom observer for streaming output
class StreamingObserver : public litert::lm::InferenceObservable {
 public:
  StreamingObserver(int sock) : sock_(sock), tokens_sent_(0), total_bytes_sent_(0),
                               repetition_count_(0), max_repetitions_(3) {
    recent_tokens_.reserve(10);
  }

  // Override OnNext to write tokens incrementally
  void OnNext(const litert::lm::Responses& responses) override {
    // Get clean text without metadata
    auto text_or = responses.GetResponseTextAt(0);
    if (text_or.ok()) {
      // Check for repetition to prevent infinite loops
      if (is_repeating(std::string(*text_or))) {
        repetition_count_++;
        ABSL_LOG(WARNING) << "Repetition detected (" << repetition_count_ << "/" << max_repetitions_ << "): " << *text_or;

        if (repetition_count_ >= max_repetitions_) {
          ABSL_LOG(ERROR) << "Too many repetitions detected, stopping generation";
          const char* stop_msg = " [Generation stopped due to repetition]";
          send(sock_, stop_msg, strlen(stop_msg), 0);
          // Force stop by not processing more tokens
          return;
        }
      } else {
        repetition_count_ = 0;  // Reset counter on new content
      }

      // Add token to recent history
      add_to_recent_tokens(std::string(*text_or));

      tokens_sent_++;
      ABSL_LOG(INFO) << "Token #" << tokens_sent_ << " received: " << *text_or;
      ssize_t sent = send(sock_, (*text_or).data(), (*text_or).length(), 0);
      if (sent >= 0) {
        total_bytes_sent_ += sent;
        ABSL_LOG(INFO) << "Sent " << sent << " bytes to client (total: " << total_bytes_sent_ << " bytes)";
      } else {
        ABSL_LOG(ERROR) << "Failed to send token to client: " << strerror(errno);
      }
    } else {
      ABSL_LOG(ERROR) << "Failed to get response text: " << text_or.status();
    }
  }

  // Override OnCompleted to add EOF marker and signal completion
  void OnCompleted() {
    ABSL_LOG(INFO) << "Streaming completed after " << tokens_sent_ << " tokens (" << total_bytes_sent_ << " bytes total)";
    ABSL_LOG(INFO) << "Sending EOF marker";
    const char* eof_marker = "\n<END_OF_RESPONSE>\n";
    ssize_t sent = send(sock_, eof_marker, strlen(eof_marker), 0);
    if (sent >= 0) {
      ABSL_LOG(INFO) << "EOF marker sent successfully (" << sent << " bytes)";
    } else {
      ABSL_LOG(ERROR) << "Failed to send EOF marker: " << strerror(errno);
    }
  }

  // Override OnError for error handling
  void OnError(const absl::Status& status) override {
    ABSL_LOG(ERROR) << "Streaming error after " << tokens_sent_ << " tokens: " << status;
    // Just close connection on streaming errors without sending error messages
    close(sock_);
  }

private:
 bool is_repeating(const std::string& token) {
   // Check if token is the same as the last few tokens
   for (size_t i = 0; i < recent_tokens_.size() && i < 3; ++i) {
     if (recent_tokens_[recent_tokens_.size() - 1 - i] == token) {
       return true;
     }
   }
   return false;
 }

 void add_to_recent_tokens(const std::string& token) {
   recent_tokens_.push_back(token);
   // Keep only last 5 tokens
   if (recent_tokens_.size() > 5) {
     recent_tokens_.erase(recent_tokens_.begin());
   }
 }

 int sock_;
 int tokens_sent_;
 size_t total_bytes_sent_;
 std::vector<std::string> recent_tokens_;
 int repetition_count_;
 const int max_repetitions_;
};

absl::Status RunInference(Engine* llm, std::unique_ptr<Engine::Session> session,
                         const std::string& input_prompt, int client_sock) {
  std::vector<InputData> inputs;
  inputs.emplace_back(InputText(input_prompt));

  // Use streaming for incremental output
  StreamingObserver observer(client_sock);

  absl::Status status = session->GenerateContentStream(inputs, &observer);
  ABSL_CHECK_OK(status);
  status = llm->WaitUntilDone(kWaitUntilDoneTimeout);
  if (!status.ok()) {
    // Just close connection on timeout without sending error messages
    close(client_sock);
    return absl::OkStatus();
  }

  // Ensure client socket is closed after inference
  close(client_sock);

  return absl::OkStatus();
}

absl::Status MainHelper(int argc, char** argv) {
  // Set minimal log level to INFO
  LiteRtSetMinLoggerSeverity(LiteRtGetDefaultLogger(), LITERT_INFO);

  const std::string default_model_path = "../models/gemma3-1b-it-int4.litertlm";
  std::string model_path = (argc >= 2) ? argv[1] : default_model_path;

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

  // Set up socket server
  const int PORT = 5188;
  int server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd < 0) {
    ABSL_LOG(ERROR) << "Socket creation failed";
    return absl::InternalError("Socket creation failed");
  }

  struct sockaddr_in address;
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(PORT);

  if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
    ABSL_LOG(ERROR) << "Bind failed";
    close(server_fd);
    return absl::InternalError("Bind failed");
  }

  if (listen(server_fd, 1) < 0) {
    ABSL_LOG(ERROR) << "Listen failed";
    close(server_fd);
    return absl::InternalError("Listen failed");
  }

  ABSL_LOG(INFO) << "Listening on port 5188";

  // Loop to handle multiple clients
  while (true) {
    socklen_t addrlen = sizeof(address);
    int client_sock = accept(server_fd, (struct sockaddr*)&address, &addrlen);
    if (client_sock < 0) {
      ABSL_LOG(ERROR) << "Accept failed";
      continue;
    }

    ABSL_LOG(INFO) << "Client connected";

  // Read prompt from client
  std::string input_prompt;
  char buffer[1024];
  int bytes_read;
  const size_t MAX_PROMPT_LENGTH = 1024 * 10;  // 10KB limit
  size_t total_bytes_received = 0;
  int recv_count = 0;

  ABSL_LOG(INFO) << "Starting to read prompt from client";

  while ((bytes_read = recv(client_sock, buffer, sizeof(buffer) - 1, 0)) > 0) {
    buffer[bytes_read] = '\0';
    input_prompt += buffer;
    total_bytes_received += bytes_read;
    recv_count++;

    ABSL_LOG(INFO) << "Received chunk #" << recv_count << " (" << bytes_read << " bytes): " << std::string(buffer, bytes_read);

    // Prevent infinite reading
    if (input_prompt.length() > MAX_PROMPT_LENGTH) {
      ABSL_LOG(WARNING) << "Prompt too long (" << input_prompt.length() << " chars), truncating";
      input_prompt = input_prompt.substr(0, MAX_PROMPT_LENGTH);
      break;
    }
  }

  ABSL_LOG(INFO) << "Finished reading prompt. Total chunks: " << recv_count << ", Total bytes: " << total_bytes_received;

  // Remove #### marker if present
  size_t marker_pos = input_prompt.find("####");
  if (marker_pos != std::string::npos) {
    ABSL_LOG(INFO) << "#### marker found at position " << marker_pos << ", removing it";
    input_prompt = input_prompt.substr(0, marker_pos);
  } else {
    ABSL_LOG(INFO) << "No #### marker found";
  }

  if (bytes_read < 0) {
    ABSL_LOG(ERROR) << "Recv failed with error: " << strerror(errno);
    close(client_sock);
    continue;  // Continue listening for next client
  }

  if (input_prompt.empty()) {
    ABSL_LOG(ERROR) << "Empty prompt received";
    close(client_sock);
    return;
  }

  ABSL_LOG(INFO) << "Final processed prompt (length: " << input_prompt.length() << "): " << input_prompt.substr(0, 500) << (input_prompt.length() > 500 ? "..." : "");

  // Check for suspicious patterns that might cause crashes
  if (input_prompt.find('\0') != std::string::npos) {
    ABSL_LOG(WARNING) << "Null character found in prompt at position: " << input_prompt.find('\0');
  }
  if (input_prompt.length() > 10000) {
    ABSL_LOG(WARNING) << "Very long prompt: " << input_prompt.length() << " characters";
  }

  ABSL_LOG(INFO) << "Creating new session for request";
  SessionConfig session_config = SessionConfig::CreateDefault();
  absl::StatusOr<std::unique_ptr<Engine::Session>> session_or =
      (*llm)->CreateSession(session_config);
  ABSL_CHECK_OK(session_or) << "Failed to create session";
  std::unique_ptr<Engine::Session> session = std::move(*session_or);
  ABSL_LOG(INFO) << "Session created successfully";

  // Run inference
  RETURN_IF_ERROR(RunInference(llm->get(), std::move(session), input_prompt, client_sock));

  // Client socket is closed in StreamingObserver::OnCompleted or OnError
  // Continue to next client
  }
}

}  // namespace

int main(int argc, char** argv) {
  ABSL_CHECK_OK(MainHelper(argc, argv));
  return 0;
}
