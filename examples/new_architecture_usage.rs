// 新架构使用示例
//
// 展示如何使用 CALF 的新分层架构：
// 1. CalfInferenceEngine - 核心推理引擎（底层）
// 2. CalfApiService - OpenAI兼容的API服务（高层）

use calf::{
    CalfApiService,
    ChatRequest, ChatMessage, ChatResponse,
    CompletionRequest, CompletionResponse,
    EmbeddingRequest, EmbeddingResponse, EmbeddingInput,
    CreateSessionRequest, SessionConfig,
    InferenceRequest, InferenceResponse,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建API服务实例
    let api_service = CalfApiService::new("models.yaml", false).await?;

    println!("=== CALF 新架构使用示例 ===\n");

    // 示例 1: OpenAI兼容的聊天完成
    println!("1. 聊天完成 (Chat Completion)");
    let chat_request = ChatRequest {
        model: "qwen2.5-0.5b-instruct".to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant.".to_string(),
                name: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hello! What can you help me with?".to_string(),
                name: None,
            },
        ],
        max_tokens: 100,
        temperature: 0.8,
        top_p: None,
        top_k: None,
        repeat_penalty: 1.1,
        stream: false,
        stop: None,
        user: None,
    };

    match api_service.chat_completion(chat_request).await {
        Ok(response) => {
            println!("Response ID: {}", response.id);
            if let Some(choice) = response.choices.first() {
                println!("Assistant: {}", choice.message.content);
            }
            println!("Usage: {} tokens\n", response.usage.total_tokens);
        }
        Err(e) => println!("Chat completion error: {}\n", e),
    }

    // 示例 2: 文本完成
    println!("2. 文本完成 (Text Completion)");
    let completion_request = CompletionRequest {
        model: "qwen2.5-0.5b-instruct".to_string(),
        prompt: "The future of artificial intelligence is".to_string(),
        max_tokens: 50,
        temperature: 0.7,
        top_p: None,
        top_k: None,
        repeat_penalty: Some(1.1),
        seed: None,
        stream: false,
        stop: None,
        user: None,
    };

    match api_service.completion(completion_request).await {
        Ok(response) => {
            if let Some(choice) = response.choices.first() {
                println!("Completion: {}", choice.text);
            }
            println!("Usage: {} tokens\n", response.usage.total_tokens);
        }
        Err(e) => println!("Completion error: {}\n", e),
    }

    // 示例 3: 嵌入向量计算
    println!("3. 嵌入向量 (Embeddings)");
    let embedding_request = EmbeddingRequest {
        model: "qwen2.5-0.5b-instruct".to_string(),
        input: EmbeddingInput::Single("Hello, world!".to_string()),
        encoding_format: "float".to_string(),
        dimensions: None,
        user: None,
    };

    match api_service.embeddings(embedding_request).await {
        Ok(response) => {
            if let Some(embedding_data) = response.data.first() {
                println!("Embedding dimensions: {}", embedding_data.embedding.len());
                println!("First 5 values: {:?}", &embedding_data.embedding[..5.min(embedding_data.embedding.len())]);
            }
            println!("Usage: {} tokens\n", response.usage.total_tokens);
        }
        Err(e) => println!("Embedding error: {}\n", e),
    }

    // 示例 4: 批量嵌入向量
    println!("4. 批量嵌入向量 (Batch Embeddings)");
    let batch_embedding_request = EmbeddingRequest {
        model: "qwen2.5-0.5b-instruct".to_string(),
        input: EmbeddingInput::Batch(vec![
            "First document".to_string(),
            "Second document".to_string(),
            "Third document".to_string(),
        ]),
        encoding_format: "float".to_string(),
        dimensions: None,
        user: None,
    };

    match api_service.embeddings(batch_embedding_request).await {
        Ok(response) => {
            println!("Processed {} documents", response.data.len());
            for (i, embedding_data) in response.data.iter().enumerate() {
                println!("Document {}: {} dimensions", i + 1, embedding_data.embedding.len());
            }
            println!("Total usage: {} tokens\n", response.usage.total_tokens);
        }
        Err(e) => println!("Batch embedding error: {}\n", e),
    }

    // 示例 5: 会话管理
    println!("5. 会话管理 (Session Management)");
    let create_session_request = CreateSessionRequest {
        model: "qwen2.5-0.5b-instruct".to_string(),
        system_message: Some("You are a helpful coding assistant.".to_string()),
        config: Some(SessionConfig {
            max_history: 10,
            temperature: 0.7,
            max_tokens: 100,
        }),
    };

    match api_service.create_session(create_session_request).await {
        Ok(session_info) => {
            println!("Created session: {}", session_info.id);

            // 在会话中聊天
            match api_service.chat_with_session(&session_info.id, "What's the best programming language?".to_string()).await {
                Ok(response) => {
                    if let Some(choice) = response.choices.first() {
                        println!("Session response: {}", choice.message.content);
                    }
                }
                Err(e) => println!("Session chat error: {}", e),
            }

            // 删除会话
            let _ = api_service.delete_session(&session_info.id);
            println!("Session deleted\n");
        }
        Err(e) => println!("Session creation error: {}\n", e),
    }

    // 示例 6: 低层次推理接口（通过引擎直接访问）
    println!("6. 低层次推理 (Low-level Inference)");
    let inference_request = InferenceRequest {
        model: "qwen2.5-0.5b-instruct".to_string(),
        prompt: "Explain quantum computing in simple terms:".to_string(),
        max_tokens: 80,
        temperature: 0.8,
        top_p: None,
        top_k: None,
        repeat_penalty: 1.1,
        repeat_last_n: 64,
        seed: None,
        split_prompt: false,
        stream: false,
        stop: None,
    };

    match api_service.engine().inference(inference_request).await {
        Ok(response) => {
            println!("Generated text: {}", response.text);
            println!("Processing time: {}ms", response.processing_time_ms);
            println!("Tokens generated: {}", response.tokens_generated);
        }
        Err(e) => println!("Low-level inference error: {}", e),
    }

    // 示例 7: 批量推理
    println!("\n7. 批量推理 (Batch Inference)");
    let batch_requests = vec![
        InferenceRequest {
            model: "qwen2.5-0.5b-instruct".to_string(),
            prompt: "The benefits of exercise are".to_string(),
            max_tokens: 30,
            temperature: 0.7,
            top_p: None,
            top_k: None,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: None,
            split_prompt: false,
            stream: false,
            stop: None,
        },
        InferenceRequest {
            model: "qwen2.5-0.5b-instruct".to_string(),
            prompt: "The importance of education is".to_string(),
            max_tokens: 30,
            temperature: 0.7,
            top_p: None,
            top_k: None,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: None,
            split_prompt: false,
            stream: false,
            stop: None,
        },
    ];

    match api_service.batch_inference(batch_requests).await {
        Ok(responses) => {
            for (i, response) in responses.iter().enumerate() {
                println!("Batch result {}: {}", i + 1, response.text.lines().next().unwrap_or(&response.text));
            }
        }
        Err(e) => println!("Batch inference error: {}", e),
    }

    // 示例 8: 模型列表
    println!("\n8. 可用模型列表 (Available Models)");
    let models = api_service.list_models();
    println!("Available models:");
    for model in models.data {
        println!("  - {} ({})", model.id, model.name);
    }

    println!("\n=== 示例完成 ===");
    Ok(())
}