use crate::types::{
    ChatMessage, SessionInfo, CreateSessionRequest, SessionConfig, InferenceError, InferenceResult
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// 聊天会话
#[derive(Debug, Clone)]
pub struct ChatSession {
    /// 会话ID
    pub id: String,
    /// 使用的模型
    pub model: String,
    /// 消息历史
    pub messages: Vec<ChatMessage>,
    /// 创建时间戳
    pub created: u64,
    /// 最后更新时间戳
    pub last_updated: u64,
    /// 会话配置
    pub config: SessionConfig,
    /// 会话状态
    pub status: SessionStatus,
}

/// 会话状态
#[derive(Debug, Clone, PartialEq)]
pub enum SessionStatus {
    Active,
    Inactive,
    Expired,
}

impl std::fmt::Display for SessionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionStatus::Active => write!(f, "active"),
            SessionStatus::Inactive => write!(f, "inactive"),
            SessionStatus::Expired => write!(f, "expired"),
        }
    }
}

impl ChatSession {
    /// 创建新会话
    pub fn new(model: String, config: SessionConfig) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id: Uuid::new_v4().to_string(),
            model,
            messages: Vec::new(),
            created: now,
            last_updated: now,
            config,
            status: SessionStatus::Active,
        }
    }

    /// 添加消息到会话
    pub fn add_message(&mut self, message: ChatMessage) {
        self.messages.push(message);
        self.last_updated = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // 限制历史消息数量
        if self.messages.len() > self.config.max_history {
            // 保留系统消息（如果存在）
            let system_messages: Vec<_> = self.messages
                .iter()
                .filter(|m| m.role == "system")
                .cloned()
                .collect();

            let mut other_messages: Vec<_> = self.messages
                .iter()
                .filter(|m| m.role != "system")
                .cloned()
                .collect();

            // 只保留最近的消息
            let keep_count = self.config.max_history.saturating_sub(system_messages.len());
            if other_messages.len() > keep_count {
                other_messages.drain(0..other_messages.len() - keep_count);
            }

            self.messages = system_messages;
            self.messages.extend(other_messages);
        }
    }

    /// 获取用于推理的消息历史
    pub fn get_context_messages(&self) -> Vec<ChatMessage> {
        self.messages.clone()
    }

    /// 获取会话信息
    pub fn get_info(&self) -> SessionInfo {
        SessionInfo {
            id: self.id.clone(),
            created: self.created,
            last_updated: self.last_updated,
            model: self.model.clone(),
            message_count: self.messages.len(),
            status: self.status.to_string(),
        }
    }

    /// 更新会话状态
    pub fn update_status(&mut self, status: SessionStatus) {
        self.status = status;
        self.last_updated = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }

    /// 清空会话历史（保留系统消息）
    pub fn clear_history(&mut self) {
        let system_messages: Vec<_> = self.messages
            .iter()
            .filter(|m| m.role == "system")
            .cloned()
            .collect();

        self.messages = system_messages;
        self.last_updated = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }
}

/// 会话管理器
pub struct SessionManager {
    /// 会话存储
    sessions: Arc<RwLock<HashMap<String, ChatSession>>>,
    /// 会话超时时间（秒）
    session_timeout: u64,
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new(3600) // 默认1小时超时
    }
}

impl SessionManager {
    /// 创建新的会话管理器
    pub fn new(session_timeout: u64) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            session_timeout,
        }
    }

    /// 创建新会话
    pub fn create_session(&self, request: CreateSessionRequest) -> InferenceResult<ChatSession> {
        let mut session = ChatSession::new(request.model.clone(),
            request.config.unwrap_or_default());

        // 如果有系统消息，添加到会话中
        if let Some(system_message) = request.system_message {
            session.add_message(ChatMessage {
                role: "system".to_string(),
                content: system_message,
                name: None,
            });
        }

        let session_id = session.id.clone();

        // 存储会话
        {
            let mut sessions = self.sessions.write()
                .map_err(|_| InferenceError::Other(anyhow::anyhow!("Failed to acquire write lock")))?;
            sessions.insert(session_id.clone(), session.clone());
        }

        Ok(session)
    }

    /// 获取会话
    pub fn get_session(&self, session_id: &str) -> InferenceResult<ChatSession> {
        let sessions = self.sessions.read()
            .map_err(|_| InferenceError::Other(anyhow::anyhow!("Failed to acquire read lock")))?;

        sessions.get(session_id)
            .cloned()
            .ok_or_else(|| InferenceError::SessionNotFound(session_id.to_string()))
    }

    /// 更新会话
    pub fn update_session(&self, session: ChatSession) -> InferenceResult<()> {
        let mut sessions = self.sessions.write()
            .map_err(|_| InferenceError::Other(anyhow::anyhow!("Failed to acquire write lock")))?;

        sessions.insert(session.id.clone(), session);
        Ok(())
    }

    /// 删除会话
    pub fn delete_session(&self, session_id: &str) -> InferenceResult<()> {
        let mut sessions = self.sessions.write()
            .map_err(|_| InferenceError::Other(anyhow::anyhow!("Failed to acquire write lock")))?;

        sessions.remove(session_id)
            .ok_or_else(|| InferenceError::SessionNotFound(session_id.to_string()))?;

        Ok(())
    }

    /// 列出所有会话
    pub fn list_sessions(&self) -> InferenceResult<Vec<SessionInfo>> {
        let sessions = self.sessions.read()
            .map_err(|_| InferenceError::Other(anyhow::anyhow!("Failed to acquire read lock")))?;

        Ok(sessions.values().map(|s| s.get_info()).collect())
    }

    /// 清理过期会话
    pub fn cleanup_expired_sessions(&self) -> InferenceResult<usize> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut sessions = self.sessions.write()
            .map_err(|_| InferenceError::Other(anyhow::anyhow!("Failed to acquire write lock")))?;

        let initial_count = sessions.len();

        // 移除超时的会话
        sessions.retain(|_, session| {
            now - session.last_updated < self.session_timeout
        });

        Ok(initial_count - sessions.len())
    }

    /// 获取活跃会话数量
    pub fn get_active_session_count(&self) -> InferenceResult<usize> {
        let sessions = self.sessions.read()
            .map_err(|_| InferenceError::Other(anyhow::anyhow!("Failed to acquire read lock")))?;

        Ok(sessions.len())
    }

    /// 添加消息到指定会话
    pub fn add_message_to_session(&self, session_id: &str, message: ChatMessage) -> InferenceResult<()> {
        let mut sessions = self.sessions.write()
            .map_err(|_| InferenceError::Other(anyhow::anyhow!("Failed to acquire write lock")))?;

        if let Some(session) = sessions.get_mut(session_id) {
            session.add_message(message);
            Ok(())
        } else {
            Err(InferenceError::SessionNotFound(session_id.to_string()))
        }
    }

    /// 获取会话的消息历史
    pub fn get_session_messages(&self, session_id: &str) -> InferenceResult<Vec<ChatMessage>> {
        let sessions = self.sessions.read()
            .map_err(|_| InferenceError::Other(anyhow::anyhow!("Failed to acquire read lock")))?;

        if let Some(session) = sessions.get(session_id) {
            Ok(session.get_context_messages())
        } else {
            Err(InferenceError::SessionNotFound(session_id.to_string()))
        }
    }

    /// 清空会话历史
    pub fn clear_session_history(&self, session_id: &str) -> InferenceResult<()> {
        let mut sessions = self.sessions.write()
            .map_err(|_| InferenceError::Other(anyhow::anyhow!("Failed to acquire write lock")))?;

        if let Some(session) = sessions.get_mut(session_id) {
            session.clear_history();
            Ok(())
        } else {
            Err(InferenceError::SessionNotFound(session_id.to_string()))
        }
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_history: 20,
            temperature: 0.8,
            max_tokens: 512,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_creation() {
        let config = SessionConfig::default();
        let session = ChatSession::new("test-model".to_string(), config);

        assert!(!session.id.is_empty());
        assert_eq!(session.model, "test-model");
        assert!(session.messages.is_empty());
        assert_eq!(session.status, SessionStatus::Active);
    }

    #[test]
    fn test_message_history_limit() {
        let mut session = ChatSession::new("test-model".to_string(), SessionConfig {
            max_history: 3,
            temperature: 0.8,
            max_tokens: 512,
        });

        // 添加系统消息
        session.add_message(ChatMessage {
            role: "system".to_string(),
            content: "You are a helpful assistant".to_string(),
            name: None,
        });

        // 添加多个用户消息
        for i in 0..5 {
            session.add_message(ChatMessage {
                role: "user".to_string(),
                content: format!("Message {}", i),
                name: None,
            });
        }

        // 应该保留系统消息和最近的2个用户消息
        assert_eq!(session.messages.len(), 3);
        assert_eq!(session.messages[0].role, "system");
        assert_eq!(session.messages[1].content, "Message 3");
        assert_eq!(session.messages[2].content, "Message 4");
    }

    #[test]
    fn test_session_manager() {
        let manager = SessionManager::default();

        let request = CreateSessionRequest {
            model: "test-model".to_string(),
            system_message: Some("Test system message".to_string()),
            config: None,
        };

        let session = manager.create_session(request).unwrap();
        let session_id = session.id.clone();

        // 测试获取会话
        let retrieved = manager.get_session(&session_id).unwrap();
        assert_eq!(retrieved.id, session_id);
        assert_eq!(retrieved.messages.len(), 1);
        assert_eq!(retrieved.messages[0].role, "system");

        // 测试列出会话
        let sessions = manager.list_sessions().unwrap();
        assert_eq!(sessions.len(), 1);

        // 测试删除会话
        manager.delete_session(&session_id).unwrap();
        assert!(manager.get_session(&session_id).is_err());
    }
}