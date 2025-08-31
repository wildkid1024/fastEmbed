#include <crow.h>
#include "sentence_transformers.h"
#include "tokenizer.h"
#include <nlohmann/json.hpp>
#include <cstring>


int main(int argc, char* argv[]) {
    std::string model_path;
    std::string serve_model_name = "default-model"; // 默认模型名称

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model_path") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--serve_model_name") == 0 && i + 1 < argc) {
            serve_model_name = argv[++i];
        } else {
            std::cerr << "Usage: " << argv[0] << " --model_path <path> [--serve_model_name <name>]" << std::endl;
            return 1;
        }
    }

    if (model_path.empty()) {
        std::cerr << "Error: --model_path is required" << std::endl;
        return 1;
    }

    SentenceTransformer model(model_path);
    crow::SimpleApp app;  // Fixed app declaration

    // crow::App<crow::middleware::CORSHandler> app;
    // auto& cors = app.get_middleware<crow::CORSHandler>();
    // cors.global().allow_origin("*");
    // cors.global().allow_headers("*");
    // cors.global().allow_methods("GET, POST, OPTIONS");

    // 嵌入接口
    // Fixed route with double quotes and explicit JSON namespace
    CROW_ROUTE(app, "/embed")
        .methods(crow::HTTPMethod::POST)
    ([&](const crow::request& req) {
        nlohmann::json req_json = nlohmann::json::parse(req.body);
        std::string text = req_json["text"];

        std::vector<std::string> texts = {text};
        std::vector<std::vector<float>> embeddings = model.encode_batch(texts);

        // 返回结果
        nlohmann::json res_json;
        res_json["embedding"] = embeddings[0];
        return crow::response(200, res_json.dump());
    });

     // OpenAI compatible embedding endpoint
    CROW_ROUTE(app, "/v1/embeddings")
        .methods(crow::HTTPMethod::POST)
    ([&](const crow::request& req) {
        try {
            nlohmann::json req_json = nlohmann::json::parse(req.body);

            // Validate required parameters
            if (!req_json.contains("input") || (!req_json["input"].is_string() && !req_json["input"].is_array())) {
                return crow::response(400, R"({
                    "error": {
                        "message": "Invalid input parameter. Must provide a string or array 'input' field.",
                        "type": "invalid_request_error",
                        "param": "input",
                        "code": "missing_required_parameter"
                    }
                })"_json.dump());
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

            return crow::response(200, response_json.dump());
        } catch (const std::exception& e) {
            return crow::response(500, R"({
                "error": {
                    "message": "Internal server error: )" + std::string(e.what()) + R"(",
                    "type": "server_error",
                    "code": "internal_error"
                }
            })"_json.dump());
        }
    });


    // 启动服务器
    app.port(8080).multithreaded().run();
    return 0;
}