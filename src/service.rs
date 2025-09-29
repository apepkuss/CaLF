use crate::engine::CalfInferenceEngine;
use crate::session::SessionManager;
use crate::types::{
    InferenceRequest, InferenceResponse, ChatRequest, ChatResponse, ChatMessage,
    ChatChoice, Usage, ModelListResponse, CreateSessionRequest,
    SessionInfo, InferenceError, InferenceResult, EmbeddingRequest, EmbeddingResponse,
    CompletionRequest, CompletionResponse,
};

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH, Instant};
use uuid::Uuid;

/// CALF API服务 - 提供OpenAI兼容的高层接口
pub struct CalfApiService {
    /// 核心推理引擎
    engine: Arc<CalfInferenceEngine>,
    /// 会话管理器
    session_manager: Arc<SessionManager>,
}

impl CalfApiService {
    /// 创建新的API服务实例
    pub async fn new(config_path: &str, force_cpu: bool) -> InferenceResult<Self> {
        let engine = Arc::new(CalfInferenceEngine::new(config_path, force_cpu).await?);
        let session_manager = Arc::new(SessionManager::default());

        Ok(Self {
            engine,
            session_manager,
        })
    }

    /// 从已有的推理引擎创建API服务
    pub fn from_engine(engine: CalfInferenceEngine) -> Self {
        Self {
            engine: Arc::new(engine),
            session_manager: Arc::new(SessionManager::default()),
        }
    }

    /// 获取推理引擎的引用（用于低层次操作）
    pub fn engine(&self) -> &Arc<CalfInferenceEngine> {
        &self.engine
    }

    /// OpenAI兼容的聊天完成接口
    pub async fn chat_completion(&self, request: ChatRequest) -> InferenceResult<ChatResponse> {
        let start_time = Instant::now();

        // 确保模型已加载
        self.engine.ensure_model_loaded(&request.model).await?;

        // 构建完整的对话上下文
        let conversation_text = self.build_conversation_context(&request.messages, &request.model)?;

        // 创建推理请求
        let inference_request = InferenceRequest {
            model: request.model.clone(),
            prompt: conversation_text,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            repeat_penalty: request.repeat_penalty,
            repeat_last_n: 64,
            seed: None,
            split_prompt: false,
            stream: false,
            stop: request.stop,
        };

        // 执行推理
        let response = self.engine.inference(inference_request).await?;

        // 计算token使用量
        let prompt_tokens = self.engine.count_tokens(&request.model, &response.text).await?;

        let _processing_time = start_time.elapsed().as_millis();

        // 构建OpenAI格式的响应
        let choice = ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: response.text,
                name: None,
            },
            finish_reason: if response.finished_by_stop {
                response.stop_reason
            } else {
                Some("length".to_string())
            },
        };

        let usage = Usage {
            prompt_tokens,
            completion_tokens: response.tokens_generated,
            total_tokens: prompt_tokens + response.tokens_generated,
        };

        Ok(ChatResponse {
            id: format!("chatcmpl-{}", Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            model: request.model,
            choices: vec![choice],
            usage,
        })
    }

    /// OpenAI兼容的文本完成接口
    pub async fn completion(&self, request: CompletionRequest) -> InferenceResult<CompletionResponse> {
        // 创建推理请求
        let inference_request = InferenceRequest {
            model: request.model.clone(),
            prompt: request.prompt.clone(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            repeat_penalty: request.repeat_penalty.unwrap_or(1.1),
            repeat_last_n: 64,
            seed: request.seed,
            split_prompt: false,
            stream: request.stream,
            stop: request.stop,
        };

        // 执行推理
        let response = self.engine.inference(inference_request).await?;

        // 构建OpenAI格式的响应
        Ok(CompletionResponse {
            id: format!("cmpl-{}", Uuid::new_v4()),
            object: "text_completion".to_string(),
            created: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            model: request.model,
            choices: vec![crate::types::CompletionChoice {
                text: response.text,
                index: 0,
                logprobs: None,
                finish_reason: if response.finished_by_stop {
                    response.stop_reason
                } else {
                    Some("length".to_string())
                },
            }],
            usage: crate::types::CompletionUsage {
                prompt_tokens: 0, // 需要实际计算
                completion_tokens: response.tokens_generated,
                total_tokens: response.tokens_generated,
            },
        })
    }

    /// OpenAI兼容的嵌入向量接口
    pub async fn embeddings(&self, request: EmbeddingRequest) -> InferenceResult<EmbeddingResponse> {
        // 确保模型已加载
        self.engine.ensure_model_loaded(&request.model).await?;

        match request.input {
            crate::types::EmbeddingInput::Single(text) => {
                let embedding = self.engine.compute_embeddings(&request.model, &text).await?;

                let token_count = self.engine.count_tokens(&request.model, &text).await?;
                Ok(EmbeddingResponse {
                    object: "list".to_string(),
                    data: vec![crate::types::EmbeddingData {
                        object: "embedding".to_string(),
                        index: 0,
                        embedding,
                    }],
                    model: request.model,
                    usage: crate::types::EmbeddingUsage {
                        prompt_tokens: token_count,
                        total_tokens: token_count,
                    },
                })
            }
            crate::types::EmbeddingInput::Batch(texts) => {
                let mut embeddings_data = Vec::new();
                let mut total_tokens = 0;

                for (index, text) in texts.iter().enumerate() {
                    let embedding = self.engine.compute_embeddings(&request.model, text).await?;
                    let tokens = self.engine.count_tokens(&request.model, text).await?;

                    embeddings_data.push(crate::types::EmbeddingData {
                        object: "embedding".to_string(),
                        index,
                        embedding,
                    });

                    total_tokens += tokens;
                }

                Ok(EmbeddingResponse {
                    object: "list".to_string(),
                    data: embeddings_data,
                    model: request.model,
                    usage: crate::types::EmbeddingUsage {
                        prompt_tokens: total_tokens,
                        total_tokens,
                    },
                })
            }
        }
    }

    /// 列出可用模型
    pub fn list_models(&self) -> ModelListResponse {
        self.engine.list_models()
    }

    /// 创建聊天会话
    pub async fn create_session(&self, request: CreateSessionRequest) -> InferenceResult<SessionInfo> {
        // 验证模型存在
        if !self.engine.has_model(&request.model) {
            return Err(InferenceError::ModelNotFound(request.model));
        }

        let session = self.session_manager.create_session(request)?;
        Ok(session.get_info())
    }

    /// 会话内聊天
    pub async fn chat_with_session(&self, session_id: &str, message: String) -> InferenceResult<ChatResponse> {
        // 获取会话
        let mut session = self.session_manager.get_session(session_id)?;

        // 添加用户消息
        session.add_message(ChatMessage {
            role: "user".to_string(),
            content: message,
            name: None,
        });

        // 构建聊天请求
        let chat_request = ChatRequest {
            model: session.model.clone(),
            messages: session.get_context_messages(),
            max_tokens: session.config.max_tokens,
            temperature: session.config.temperature,
            top_p: None,
            top_k: None,
            repeat_penalty: 1.1,
            stream: false,
            stop: None,
            user: None,
        };

        // 执行推理
        let response = self.chat_completion(chat_request).await?;

        // 添加助手回复到会话
        if let Some(choice) = response.choices.first() {
            session.add_message(choice.message.clone());
        }

        // 更新会话
        self.session_manager.update_session(session)?;

        Ok(response)
    }

    /// 获取会话列表
    pub fn list_sessions(&self) -> InferenceResult<Vec<SessionInfo>> {
        self.session_manager.list_sessions()
    }

    /// 删除会话
    pub fn delete_session(&self, session_id: &str) -> InferenceResult<()> {
        self.session_manager.delete_session(session_id)
    }

    /// 清理过期会话
    pub fn cleanup_expired_sessions(&self) -> InferenceResult<usize> {
        self.session_manager.cleanup_expired_sessions()
    }

    /// 批量推理接口
    pub async fn batch_inference(&self, requests: Vec<InferenceRequest>) -> InferenceResult<Vec<InferenceResponse>> {
        let mut responses = Vec::with_capacity(requests.len());

        for request in requests {
            let response = self.engine.inference(request).await?;
            responses.push(response);
        }

        Ok(responses)
    }

    // 私有辅助方法

    /// 构建对话上下文
    fn build_conversation_context(&self, messages: &[ChatMessage], _model_name: &str) -> InferenceResult<String> {
        // 根据模型类型构建不同的对话格式
        let mut context = String::new();

        for message in messages {
            match message.role.as_str() {
                "system" => {
                    context.push_str(&format!("System: {}\n", message.content));
                }
                "user" => {
                    context.push_str(&format!("User: {}\n", message.content));
                }
                "assistant" => {
                    context.push_str(&format!("Assistant: {}\n", message.content));
                }
                _ => {
                    return Err(InferenceError::InvalidRequest(
                        format!("Invalid message role: {}", message.role)
                    ));
                }
            }
        }

        // 添加助手提示
        context.push_str("Assistant: ");

        Ok(context)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_service_creation() {
        // 测试服务创建
        // 需要有效的配置文件来完整测试
        assert!(true);
    }
}