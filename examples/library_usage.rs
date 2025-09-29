// CALF库使用示例
// 展示如何使用CALF库进行LLM推理

use calf::{
    CalfInferenceEngine, InferenceRequest, ChatRequest, ChatMessage,
    CreateSessionRequest, SessionConfig, InferenceResult
};

#[tokio::main]
async fn main() -> InferenceResult<()> {
    println!("CALF库使用示例");
    println!("================");

    // 1. 创建推理引擎
    println!("\n1. 创建推理引擎...");
    let engine = CalfInferenceEngine::new("models.yaml", false).await?;

    // 2. 列出可用模型
    println!("\n2. 可用模型列表:");
    let models = engine.list_models();
    for model in models.data {
        println!("  - {} ({})", model.id, model.name);
    }

    // 3. 单次推理示例
    println!("\n3. 单次推理示例:");
    let inference_request = InferenceRequest {
        model: "qwen3-0.6b".to_string(),
        prompt: "什么是人工智能？".to_string(),
        max_tokens: 100,
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

    match engine.inference(inference_request).await {
        Ok(response) => {
            println!("生成的回答: {}", response.text);
            println!("生成token数: {}", response.tokens_generated);
            println!("处理时间: {}ms", response.processing_time_ms);
        }
        Err(e) => {
            println!("推理失败: {}", e);
        }
    }

    // 4. 聊天完成示例（OpenAI兼容格式）
    println!("\n4. 聊天完成示例:");
    let chat_request = ChatRequest {
        model: "qwen3-0.6b".to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: "你是一个有帮助的AI助手".to_string(),
                name: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "请简单介绍一下Rust编程语言".to_string(),
                name: None,
            },
        ],
        max_tokens: 150,
        temperature: 0.7,
        top_p: None,
        top_k: None,
        repeat_penalty: 1.1,
        stream: false,
        stop: None,
        user: None,
    };

    match engine.chat_completion(chat_request).await {
        Ok(response) => {
            println!("聊天回复: {}", response.choices[0].message.content);
            println!("使用统计: {:?}", response.usage);
        }
        Err(e) => {
            println!("聊天失败: {}", e);
        }
    }

    // 5. 会话管理示例
    println!("\n5. 会话管理示例:");
    let session_request = CreateSessionRequest {
        model: "qwen3-0.6b".to_string(),
        system_message: Some("你是一个编程助手，专门帮助用户解决编程问题。".to_string()),
        config: Some(SessionConfig {
            max_history: 10,
            temperature: 0.8,
            max_tokens: 200,
        }),
    };

    match engine.create_session(session_request).await {
        Ok(session_info) => {
            println!("创建会话成功: {}", session_info.id);

            // 在会话中进行对话
            match engine.chat_with_session(&session_info.id, "如何在Rust中处理错误？".to_string()).await {
                Ok(response) => {
                    println!("会话回复: {}", response.choices[0].message.content);
                }
                Err(e) => {
                    println!("会话对话失败: {}", e);
                }
            }

            // 继续对话
            match engine.chat_with_session(&session_info.id, "能给个具体例子吗？".to_string()).await {
                Ok(response) => {
                    println!("后续回复: {}", response.choices[0].message.content);
                }
                Err(e) => {
                    println!("后续对话失败: {}", e);
                }
            }

            // 清理会话
            match engine.delete_session(&session_info.id) {
                Ok(_) => println!("会话删除成功"),
                Err(e) => println!("会话删除失败: {}", e),
            }
        }
        Err(e) => {
            println!("创建会话失败: {}", e);
        }
    }

    // 6. 批量操作示例
    println!("\n6. 批量操作示例:");
    let prompts = vec![
        "1+1等于几？",
        "太阳系有几颗行星？",
        "Rust的优势是什么？",
    ];

    for (i, prompt) in prompts.iter().enumerate() {
        let request = InferenceRequest {
            model: "qwen3-0.6b".to_string(),
            prompt: prompt.to_string(),
            max_tokens: 50,
            temperature: 0.7,
            top_p: None,
            top_k: None,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: Some(42 + i as u64), // 不同的种子
            split_prompt: false,
            stream: false,
            stop: None,
        };

        match engine.inference(request).await {
            Ok(response) => {
                println!("问题 {}: {}", i + 1, prompt);
                println!("回答 {}: {}", i + 1, response.text);
                println!();
            }
            Err(e) => {
                println!("批量推理 {} 失败: {}", i + 1, e);
            }
        }
    }

    println!("示例演示完成！");
    Ok(())
}