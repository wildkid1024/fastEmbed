#include <crow.h>
#include "sentence_transformers.h"
#include "tokenizer.h"
#include <nlohmann/json.hpp>
#include <cstring>
#include <string>
#include <stdexcept>

// Add CORS headers helper function
void add_cors_headers(crow::response& res) {
    res.add_header("Access-Control-Allow-Origin", "*");
    res.add_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.add_header("Access-Control-Allow-Headers", "*");
}

int main(int argc, char* argv[]) {
    std::string model_path;
    std::string serve_model_name = "default-model"; // Default model name
    uint16_t server_port = 8080; // Default port
    std::string api_prefix = ""; // Default API prefix

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model_path") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--serve_model_name") == 0 && i + 1 < argc) {
            serve_model_name = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            server_port = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if (strcmp(argv[i], "--api_prefix") == 0 && i + 1 < argc) {
            api_prefix = argv[++i];
            // Ensure API prefix starts with /
            if (!api_prefix.empty() && api_prefix[0] != '/') {
                api_prefix = "/" + api_prefix;
            }
            // Ensure API prefix doesn't end with /
            if (!api_prefix.empty() && api_prefix.back() == '/') {
                api_prefix.pop_back();
            }
        } else {
            std::cerr << "Usage: " << argv[0] \
                      << " --model_path <path> [--serve_model_name <name>]" \
                      << " [--port <port_number>] [--api_prefix <prefix>]" << std::endl;
            return 1;
        }
    }

    if (model_path.empty()) {
        std::cerr << "Error: --model_path is required" << std::endl;
        return 1;
    }

    try {
        SentenceTransformer model(model_path);
        crow::SimpleApp app;

        // Build route paths
        std::string embed_route = api_prefix + "/embed";
        std::string openai_route = api_prefix + "/v1/embeddings";
        std::string health_route = api_prefix + "/health";

        // Health check endpoint using dynamic routing
        auto& health_endpoint = app.route_dynamic(health_route);
        health_endpoint.methods(crow::HTTPMethod::GET)
        ([&](const crow::request& req, crow::response& res) {
            add_cors_headers(res);
            res.body = R"({"status": "ok", "model": """ + serve_model_name + """, "version": "1.0"})";
            res.code = 200;
            res.end();
        });

        // Embedding endpoint using dynamic routing
        auto& embed_endpoint = app.route_dynamic(embed_route);
        embed_endpoint.methods(crow::HTTPMethod::POST)
        ([&](const crow::request& req, crow::response& res) {
            add_cors_headers(res);
            try {
                nlohmann::json req_json = nlohmann::json::parse(req.body);
                if (!req_json.contains("text") || !req_json["text"].is_string()) {
                    res.code = 400;
                    res.body = R"({"error": "Missing or invalid 'text' field"})";
                    res.end();
                    return;
                }
                
                std::string text = req_json["text"];
                std::vector<std::string> texts = {text};
                std::vector<std::vector<float>> embeddings = model.encode_batch(texts);

                // Return result
                nlohmann::json res_json;
                res_json["embedding"] = embeddings[0];
                res_json["model"] = serve_model_name;
                res.body = res_json.dump();
                res.code = 200;
            } catch (const std::exception& e) {
                res.code = 500;
                res.body = R"({"error": ")" + std::string(e.what()) + R"("})";
            }
            res.end();
        });

        // OpenAI compatible embedding endpoint using dynamic routing
        auto& openai_endpoint = app.route_dynamic(openai_route);
        openai_endpoint.methods(crow::HTTPMethod::POST)
        // CROW_ROUTE(app, "/v1/embeddings").methods(crow::HTTPMethod::POST)   
        ([&](const crow::request& req, crow::response& res) {
            add_cors_headers(res);
            try {
                nlohmann::json req_json = nlohmann::json::parse(req.body);

                // Validate required parameters
                if (!req_json.contains("input") || (!req_json["input"].is_string() && !req_json["input"].is_array())) {
                    res.code = 400;
                    res.body = R"({
                        "error": {
                            "message": "Invalid input parameter. Must provide a string or array 'input' field.",
                            "type": "invalid_request_error",
                            "param": "input",
                            "code": "missing_required_parameter"
                        }
                    })";
                    res.end();
                    return;
                }

                // Process input (support both single string and array)
                std::vector<std::string> texts;
                if (req_json["input"].is_string()) {
                    texts.push_back(req_json["input"]);
                } else {
                    texts = req_json["input"].get<std::vector<std::string>>();
                }

                // Generate embeddings
                std::vector<std::vector<float>> embeddings = model.encode_batch(texts);

                // Format response according to OpenAI spec
                nlohmann::json response_json;
                response_json["object"] = "list";
                response_json["data"] = nlohmann::json::array();
                response_json["model"] = serve_model_name;
                response_json["usage"] = {
                    {"prompt_tokens", 0},
                    {"total_tokens", 0}
                };

                for (size_t i = 0; i < embeddings.size(); ++i) {
                    response_json["data"].push_back({
                        {"object", "embedding"},
                        {"embedding", embeddings[i]},
                        {"index", i}
                    });
                }

                res.body = response_json.dump();
                res.code = 200;
            } catch (const std::exception& e) {
                res.code = 500;
                res.body = R"({
                    "error": {
                        "message": "Internal server error: )" + std::string(e.what()) + R"(",
                        "type": "server_error",
                        "code": "internal_error"
                    }
                })";
            }
            res.end();
        });

        // Start the server
        std::cout << "Starting server on port " << server_port << "..." << std::endl;
        std::cout << "Available endpoints:" << std::endl;
        std::cout << "  - " << health_route << " (GET, Health Check)" << std::endl;
        std::cout << "  - " << embed_route << " (POST, Simple Embedding)" << std::endl;
        std::cout << "  - " << openai_route << " (POST, OpenAI Compatible Embedding)" << std::endl;
        app.port(server_port).multithreaded().run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}