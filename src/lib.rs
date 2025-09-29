pub mod config;
pub mod model;
pub mod types;
pub mod session;
pub mod engine;
pub mod service;

// 核心配置和模型类型
pub use config::{Config, ModelConfig, ModelType, DefaultSettings};
pub use model::{UnifiedModel, ModelLoader, ModelWrapper, UnifiedModelInstance};

// API数据类型
pub use types::{
    // 推理相关
    InferenceRequest, InferenceResponse, InferenceError, InferenceResult,

    // 聊天相关
    ChatRequest, ChatResponse, ChatMessage, ChatChoice, Usage,
    ChatStreamChunk, ChatStreamChoice, ChatMessageDelta,

    // 文本完成相关
    CompletionRequest, CompletionResponse, CompletionChoice, CompletionUsage,

    // 嵌入向量相关
    EmbeddingRequest, EmbeddingResponse, EmbeddingInput, EmbeddingData, EmbeddingUsage,

    // 模型相关
    ModelInfo, ModelListResponse,

    // 会话相关
    SessionInfo, CreateSessionRequest, SessionConfig,

    // 错误处理
    ErrorResponse, ErrorDetail,
};

// 会话管理
pub use session::{SessionManager, ChatSession, SessionStatus};

// 核心推理引擎
pub use engine::{CalfInferenceEngine, TokenOutputStream};

// 高层API服务 (推荐使用)
pub use service::CalfApiService;