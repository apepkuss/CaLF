use serde::{Deserialize, Serialize};


/// 推理请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// 模型名称
    pub model: String,
    /// 输入提示词
    pub prompt: String,
    /// 最大生成tokens数量
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// 温度参数 (0.0-2.0)
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Top-p采样参数
    #[serde(default)]
    pub top_p: Option<f64>,
    /// Top-k采样参数
    #[serde(default)]
    pub top_k: Option<usize>,
    /// 重复惩罚
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,
    /// 重复惩罚上下文长度
    #[serde(default = "default_repeat_last_n")]
    pub repeat_last_n: usize,
    /// 随机种子
    #[serde(default)]
    pub seed: Option<u64>,
    /// 是否分割提示词处理
    #[serde(default)]
    pub split_prompt: bool,
    /// 是否流式输出
    #[serde(default)]
    pub stream: bool,
    /// 停止词列表
    #[serde(default)]
    pub stop: Option<Vec<String>>,
}

/// 推理响应结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// 生成的文本
    pub text: String,
    /// 生成的token数量
    pub tokens_generated: usize,
    /// 处理时间（毫秒）
    pub processing_time_ms: u128,
    /// 使用的模型
    pub model: String,
    /// 是否被停止词截断
    pub finished_by_stop: bool,
    /// 使用的停止词（如果有）
    pub stop_reason: Option<String>,
}

/// 聊天消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// 消息角色: "system", "user", "assistant"
    pub role: String,
    /// 消息内容
    pub content: String,
    /// 消息名称（可选）
    #[serde(default)]
    pub name: Option<String>,
}

/// 聊天请求（兼容OpenAI格式）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    /// 模型名称
    pub model: String,
    /// 消息历史
    pub messages: Vec<ChatMessage>,
    /// 最大生成tokens数量
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// 温度参数
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Top-p采样参数
    #[serde(default)]
    pub top_p: Option<f64>,
    /// Top-k采样参数
    #[serde(default)]
    pub top_k: Option<usize>,
    /// 重复惩罚
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,
    /// 是否流式输出
    #[serde(default)]
    pub stream: bool,
    /// 停止词
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    /// 用户ID（用于会话管理）
    #[serde(default)]
    pub user: Option<String>,
}

/// 聊天选择项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    /// 索引
    pub index: usize,
    /// 消息
    pub message: ChatMessage,
    /// 完成原因
    pub finish_reason: Option<String>,
}

/// 使用统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// 提示词tokens
    pub prompt_tokens: usize,
    /// 完成tokens
    pub completion_tokens: usize,
    /// 总tokens
    pub total_tokens: usize,
}

/// 聊天响应（兼容OpenAI格式）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// 响应ID
    pub id: String,
    /// 对象类型
    pub object: String,
    /// 创建时间戳
    pub created: u64,
    /// 使用的模型
    pub model: String,
    /// 选择项列表
    pub choices: Vec<ChatChoice>,
    /// 使用统计
    pub usage: Usage,
}

/// 流式聊天响应块
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatStreamChunk {
    /// 响应ID
    pub id: String,
    /// 对象类型
    pub object: String,
    /// 创建时间戳
    pub created: u64,
    /// 使用的模型
    pub model: String,
    /// 选择项增量
    pub choices: Vec<ChatStreamChoice>,
}

/// 流式聊天选择项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatStreamChoice {
    /// 索引
    pub index: usize,
    /// 消息增量
    pub delta: ChatMessageDelta,
    /// 完成原因
    pub finish_reason: Option<String>,
}

/// 消息增量
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessageDelta {
    /// 角色（通常只在第一个chunk中出现）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// 内容增量
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// 模型信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// 模型ID
    pub id: String,
    /// 显示名称
    pub name: String,
    /// 模型类型
    pub model_type: String,
    /// 最大上下文长度
    pub max_context_length: usize,
    /// 默认温度
    pub default_temperature: f64,
    /// 是否可用
    pub available: bool,
}

/// 模型列表响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListResponse {
    /// 对象类型
    pub object: String,
    /// 模型列表
    pub data: Vec<ModelInfo>,
}

/// 错误响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// 错误信息
    pub error: ErrorDetail,
}

/// 错误详情
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    /// 错误类型
    #[serde(rename = "type")]
    pub error_type: String,
    /// 错误消息
    pub message: String,
    /// 错误代码
    #[serde(default)]
    pub code: Option<String>,
}

/// 会话信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    /// 会话ID
    pub id: String,
    /// 创建时间戳
    pub created: u64,
    /// 最后更新时间戳
    pub last_updated: u64,
    /// 使用的模型
    pub model: String,
    /// 消息数量
    pub message_count: usize,
    /// 会话状态
    pub status: String,
}

/// 会话创建请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSessionRequest {
    /// 模型名称
    pub model: String,
    /// 系统消息（可选）
    #[serde(default)]
    pub system_message: Option<String>,
    /// 会话配置
    #[serde(default)]
    pub config: Option<SessionConfig>,
}

/// 会话配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// 最大历史消息数
    #[serde(default = "default_max_history")]
    pub max_history: usize,
    /// 温度参数
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// 最大tokens
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

// 默认值函数
fn default_max_tokens() -> usize {
    512
}

fn default_temperature() -> f64 {
    0.8
}

fn default_repeat_penalty() -> f32 {
    1.1
}

fn default_repeat_last_n() -> usize {
    64
}

fn default_max_history() -> usize {
    20
}

fn default_encoding_format() -> String {
    "float".to_string()
}

/// 文本完成请求（兼容OpenAI格式）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// 模型名称
    pub model: String,
    /// 提示词
    pub prompt: String,
    /// 最大生成tokens数量
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// 温度参数
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Top-p采样参数
    #[serde(default)]
    pub top_p: Option<f64>,
    /// Top-k采样参数
    #[serde(default)]
    pub top_k: Option<usize>,
    /// 重复惩罚
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    /// 随机种子
    #[serde(default)]
    pub seed: Option<u64>,
    /// 是否流式输出
    #[serde(default)]
    pub stream: bool,
    /// 停止词
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    /// 用户ID
    #[serde(default)]
    pub user: Option<String>,
}

/// 文本完成选择项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    /// 生成的文本
    pub text: String,
    /// 索引
    pub index: usize,
    /// 对数概率信息
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
    /// 完成原因
    pub finish_reason: Option<String>,
}

/// 文本完成使用统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionUsage {
    /// 提示词tokens
    pub prompt_tokens: usize,
    /// 完成tokens
    pub completion_tokens: usize,
    /// 总tokens
    pub total_tokens: usize,
}

/// 文本完成响应（兼容OpenAI格式）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// 响应ID
    pub id: String,
    /// 对象类型
    pub object: String,
    /// 创建时间戳
    pub created: u64,
    /// 使用的模型
    pub model: String,
    /// 选择项列表
    pub choices: Vec<CompletionChoice>,
    /// 使用统计
    pub usage: CompletionUsage,
}

/// 嵌入向量输入
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// 单个文本
    Single(String),
    /// 文本数组
    Batch(Vec<String>),
}

/// 嵌入向量请求（兼容OpenAI格式）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// 模型名称
    pub model: String,
    /// 输入文本
    pub input: EmbeddingInput,
    /// 编码格式："float" 或 "base64"
    #[serde(default = "default_encoding_format")]
    pub encoding_format: String,
    /// 输出维度（可选）
    #[serde(default)]
    pub dimensions: Option<usize>,
    /// 用户ID
    #[serde(default)]
    pub user: Option<String>,
}

/// 嵌入向量数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    /// 对象类型
    pub object: String,
    /// 索引
    pub index: usize,
    /// 嵌入向量
    pub embedding: Vec<f32>,
}

/// 嵌入向量使用统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    /// 提示词tokens
    pub prompt_tokens: usize,
    /// 总tokens
    pub total_tokens: usize,
}

/// 嵌入向量响应（兼容OpenAI格式）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// 对象类型
    pub object: String,
    /// 嵌入向量数据列表
    pub data: Vec<EmbeddingData>,
    /// 使用的模型
    pub model: String,
    /// 使用统计
    pub usage: EmbeddingUsage,
}

/// 推理错误类型
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model loading failed: {0}")]
    ModelLoadError(String),

    #[error("Tokenization failed: {0}")]
    TokenizationError(String),

    #[error("Inference failed: {0}")]
    InferenceFailure(String),

    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

pub type InferenceResult<T> = Result<T, InferenceError>;