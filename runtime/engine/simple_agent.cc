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
#include <thread>
#include <nlohmann/json.hpp>


namespace {

    using ::litert::lm::Backend;
    using ::litert::lm::Engine;
    using ::litert::lm::EngineSettings;
    using ::litert::lm::InputData;
    using ::litert::lm::InputText;
    using ::litert::lm::ModelAssets;
    using ::litert::lm::SessionConfig;
    using json = nlohmann::json;

    const absl::Duration kWaitUntilDoneTimeout = absl::Minutes(1);

    // Custom observer for streaming output
    class StreamingObserver : public litert::lm::InferenceObservable {
     public:
      StreamingObserver(int sock, std::atomic<bool>& cancelled) : sock_(sock), cancelled_(&cancelled), tokens_sent_(0), total_bytes_sent_(0),
                                                                  repetition_count_(0), max_repetitions_(3), stop_sending_(false), closed_(false) {
        recent_chars_.reserve(100);
      }



      // Override OnNext to write tokens incrementally
      void OnNext(const litert::lm::Responses& responses) override {
        if (stop_sending_) return;  // Skip sending tokens after repetition limit

        // Get clean text without metadata
        auto text_or = responses.GetResponseTextAt(0);
        if (text_or.ok()) {
          // Check for repetition in character patterns to prevent infinite loops
          std::string response_chunk = std::string(*text_or);
          if (has_repeating_pattern(response_chunk)) {
            repetition_count_++;
            ABSL_LOG(WARNING) << "Repeating pattern detected (" << repetition_count_ << "/" << max_repetitions_ << "): " << response_chunk;

            if (repetition_count_ >= max_repetitions_) {
              ABSL_LOG(ERROR) << "Too many repeating patterns detected, cancelling inference remotely";
              const char* stop_msg = " [Generation stopped due to repetition]";
              send(sock_, stop_msg, strlen(stop_msg), 0);
              // Send EOF marker immediately to signal client completion
              const char* eof_marker = "\n<END_OF_RESPONSE>\n";
              ssize_t sent = send(sock_, eof_marker, strlen(eof_marker), 0);
              if (sent >= 0) {
                ABSL_LOG(INFO) << "Early EOF marker sent due to repetition (" << sent << " bytes)";
              } else {
                ABSL_LOG(ERROR) << "Failed to send early EOF marker: " << strerror(errno);
              }
              closed_ = true;
              close(sock_);
              *cancelled_ = true;
              return;  // Stop sending tokens
            }
          } else {
            repetition_count_ = 0;  // Reset counter on new content
          }

          // Add characters to recent history
          add_recent_chars(response_chunk);

          tokens_sent_++;
          ABSL_LOG(INFO) << "Token #" << tokens_sent_ << " received: " << *text_or;
          ssize_t sent = send(sock_, (*text_or).data(), (*text_or).length(), 0);
          if (sent >= 0) {
            total_bytes_sent_ += sent;
            ABSL_LOG(INFO) << "Sent " << sent << " bytes to client (total: " << total_bytes_sent_ << " bytes)";
          } else {
            ABSL_LOG(ERROR) << "Failed to send token to client: " << strerror(errno);
            stop_sending_ = true;
          }
        } else {
          ABSL_LOG(ERROR) << "Failed to get response text: " << text_or.status();
        }
      }

      // Override OnCompleted to add EOF marker and signal completion
      void OnCompleted() {
        if (closed_) return;  // Already closed
        ABSL_LOG(INFO) << "Streaming completed after " << tokens_sent_ << " tokens (" << total_bytes_sent_ << " bytes total)";
        ABSL_LOG(INFO) << "Sending EOF marker";
        const char* eof_marker = "\n<END_OF_RESPONSE>\n";
        ssize_t sent = send(sock_, eof_marker, strlen(eof_marker), 0);
        if (sent >= 0) {
          ABSL_LOG(INFO) << "EOF marker sent successfully (" << sent << " bytes)";
        } else {
          ABSL_LOG(ERROR) << "Failed to send EOF marker: " << strerror(errno);
        }
        close(sock_);
        closed_ = true;
      }

      // Override OnError for error handling
      void OnError(const absl::Status& status) override {
        ABSL_LOG(ERROR) << "Streaming error after " << tokens_sent_ << " tokens: " << status;
        // Just close connection on streaming errors without sending error messages
        if (!closed_) {
          close(sock_);
          closed_ = true;
        }
      }

    private:
     bool has_repeating_pattern(const std::string& new_chars) {
       // Add new characters to recent history first
       recent_chars_ += new_chars;

       // Keep only last 200 characters for better detection
       const size_t max_history = 200;
       if (recent_chars_.size() > max_history) {
         recent_chars_ = recent_chars_.substr(recent_chars_.size() - max_history);
       }

       // Only check for patterns if we have enough characters
       if (recent_chars_.size() < 40) {
         return false;
       }

       // Check for repeating character sequences of increasing lengths
       // Start from smaller patterns (sentences) and go up to larger ones
       for (size_t len = 15; len <= 80 && len <= recent_chars_.size() / 2; len += 5) {
         std::string last_sequence = recent_chars_.substr(recent_chars_.size() - len);
         std::string prev_sequence = recent_chars_.substr(recent_chars_.size() - 2 * len, len);

         if (last_sequence == prev_sequence && !last_sequence.empty()) {
           // Additional check - ensure it's not just punctuation or whitespace
           bool has_meaningful_content = false;
           for (char c : last_sequence) {
             if (isalnum(c)) {
               has_meaningful_content = true;
               break;
             }
           }

           if (has_meaningful_content) {
             ABSL_LOG(WARNING) << "Found repeating phrase of length " << len << ": '" << last_sequence.substr(0, 30) << "...'";
             return true;
           }
         }
       }

       return false;
     }

     void add_recent_chars(const std::string& chars) {
       // This is now handled in has_repeating_pattern for better control
     }

     private:
      int sock_;
      std::atomic<bool>* cancelled_;
      int tokens_sent_;
      size_t total_bytes_sent_;
      std::string recent_chars_;
      int repetition_count_;
      const int max_repetitions_;
      bool stop_sending_;
      bool closed_;

     };

    void RunInference(Engine* llm, std::unique_ptr<Engine::Session> session,
                      const std::vector<std::string>& formatted_inputs, int client_sock) {
      std::atomic<bool> cancelled = false;

      std::vector<InputData> inputs;
      for (const auto& formatted_input : formatted_inputs) {
        inputs.emplace_back(InputText(formatted_input));
      }

      // Use streaming for incremental output
      StreamingObserver observer(client_sock, cancelled);

      absl::Status status = session->GenerateContentStream(inputs, &observer);
      if (!status.ok()) {
        ABSL_LOG(ERROR) << "Failed to start content stream: " << status;
        close(client_sock);
        return;
      }

      // Check if cancelled remotely before waiting
      if (cancelled) {
        ABSL_LOG(INFO) << "Inference cancelled remotely, killing thread early";
        // Client socket is closed in StreamingObserver
        return;
      }

      status = llm->WaitUntilDone(kWaitUntilDoneTimeout);
      if (!status.ok()) {
        // Just close connection on timeout without sending error messages
        close(client_sock);
        return;
      }

      // Client socket is closed in StreamingObserver::OnCompleted or OnError

      return;
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

      ABSL_LOG(INFO) << "Starting to read JSON from client";

      while ((bytes_read = recv(client_sock, buffer, sizeof(buffer) - 1, 0)) > 0) {
        buffer[bytes_read] = '\0';
        input_prompt += buffer;
        total_bytes_received += bytes_read;
        recv_count++;

        ABSL_LOG(INFO) << "Received chunk #" << recv_count << " (" << bytes_read << " bytes): " << std::string(buffer, bytes_read);

        // Prevent infinite reading
        if (input_prompt.length() > MAX_PROMPT_LENGTH) {
          ABSL_LOG(WARNING) << "JSON too long (" << input_prompt.length() << " chars), truncating";
          input_prompt = input_prompt.substr(0, MAX_PROMPT_LENGTH);
          break;
        }
      }

      ABSL_LOG(INFO) << "Finished reading JSON. Total chunks: " << recv_count << ", Total bytes: " << total_bytes_received;

      if (bytes_read < 0) {
        ABSL_LOG(ERROR) << "Recv failed with error: " << strerror(errno);
        close(client_sock);
        continue;  // Continue listening for next client
      }

      if (input_prompt.empty()) {
        ABSL_LOG(ERROR) << "Empty JSON received";
        close(client_sock);
        return absl::OkStatus();
      }

      ABSL_LOG(INFO) << "Final processed JSON (length: " << input_prompt.length() << "): " << input_prompt.substr(0, 500) << (input_prompt.length() > 500 ? "..." : "");

      // Parse JSON
      std::vector<std::string> formatted_inputs;
      try {
        json j = json::parse(input_prompt);
        auto messages = j["messages"];
        for (const auto& msg : messages) {
          std::string role = msg["role"];
          std::string content = msg["content"];
          std::string full_input;
          if (role == "user") {
            full_input = "<start_of_turn>user\n" + content + "<end_of_turn>\n";
          } else if (role == "assistant") {
            full_input = "<start_of_turn>model\n" + content + "<end_of_turn>\n";
          } else {
            continue; // Ignore other roles
          }
          formatted_inputs.push_back(full_input);
        }
        if (formatted_inputs.empty()) {
          ABSL_LOG(ERROR) << "No valid messages found in JSON";
          close(client_sock);
          return absl::OkStatus();
        }
      } catch (const std::exception& e) {
        ABSL_LOG(ERROR) << "Error parsing JSON: " << e.what();
        close(client_sock);
        return absl::OkStatus();
      }

      ABSL_LOG(INFO) << "Creating new session for request";
      SessionConfig session_config = SessionConfig::CreateDefault();
      session_config.GetPromptTemplates();
      absl::StatusOr<std::unique_ptr<Engine::Session>> session_or =
          (*llm)->CreateSession(session_config);
      ABSL_CHECK_OK(session_or) << "Failed to create session";
      std::unique_ptr<Engine::Session> session = std::move(*session_or);
      ABSL_LOG(INFO) << "Session created successfully";

      // Run inference in separate thread; will be killed on repetition exception
      std::thread inference_thread(RunInference, llm->get(), std::move(session), formatted_inputs, client_sock);
      inference_thread.detach();

      // Continue to next client immediately, without waiting for inference to complete
      }
    }

}  // namespace

int main(int argc, char** argv) {
  ABSL_CHECK_OK(MainHelper(argc, argv));
  return 0;
}
