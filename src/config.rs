use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    pub display_name: String,
    pub model_type: ModelType,
    pub repo: String,
    pub filename: String,
    pub revision: String,
    pub tokenizer_repo: String,
    pub prompt_template: String,
    pub eos_token: String,
    pub default_temperature: f64,
    pub max_context_length: usize,
    #[serde(default)]
    pub features: ModelFeatures,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelFeatures {
    #[serde(default)]
    pub flash_attention: bool,
}

impl Default for ModelFeatures {
    fn default() -> Self {
        Self {
            flash_attention: false,
        }
    }
}

impl Default for DefaultSettings {
    fn default() -> Self {
        Self {
            sample_length: 512,
            temperature: 0.8,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: 42,
            split_prompt: false,
            tracing: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    Qwen3,
    Phi2,
    Phi3,
    Phi3b,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptTemplate {
    pub format: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DefaultSettings {
    pub sample_length: usize,
    pub temperature: f64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
    pub split_prompt: bool,
    pub tracing: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    /// 服务器监听地址
    #[serde(default = "default_host")]
    pub host: String,
    /// 服务器监听端口
    #[serde(default = "default_port")]
    pub port: u16,
    /// 最大并发请求数
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_requests: usize,
    /// 请求超时时间（秒）
    #[serde(default = "default_request_timeout")]
    pub request_timeout: u64,
    /// 是否启用CORS
    #[serde(default = "default_cors")]
    pub enable_cors: bool,
    /// 是否启用日志
    #[serde(default = "default_logging")]
    pub enable_logging: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SessionSettings {
    /// 默认会话超时时间（秒）
    #[serde(default = "default_session_timeout")]
    pub default_timeout: u64,
    /// 最大会话数量
    #[serde(default = "default_max_sessions")]
    pub max_sessions: usize,
    /// 自动清理间隔（秒）
    #[serde(default = "default_cleanup_interval")]
    pub cleanup_interval: u64,
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_max_concurrent() -> usize {
    100
}

fn default_request_timeout() -> u64 {
    300
}

fn default_cors() -> bool {
    true
}

fn default_logging() -> bool {
    true
}

fn default_session_timeout() -> u64 {
    3600
}

fn default_max_sessions() -> usize {
    1000
}

fn default_cleanup_interval() -> u64 {
    300
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            max_concurrent_requests: default_max_concurrent(),
            request_timeout: default_request_timeout(),
            enable_cors: default_cors(),
            enable_logging: default_logging(),
        }
    }
}

impl Default for SessionSettings {
    fn default() -> Self {
        Self {
            default_timeout: default_session_timeout(),
            max_sessions: default_max_sessions(),
            cleanup_interval: default_cleanup_interval(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub models: HashMap<String, ModelConfig>,
    pub prompt_templates: HashMap<String, PromptTemplate>,
    pub default_settings: DefaultSettings,
    /// 服务器配置（可选）
    #[serde(default)]
    pub server: Option<ServerConfig>,
    /// 会话设置（可选）
    #[serde(default)]
    pub session: Option<SessionSettings>,
}

impl Config {
    pub fn load_from_file(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    pub fn get_model(&self, name: &str) -> Option<&ModelConfig> {
        self.models.get(name)
    }

    pub fn list_models(&self) -> Vec<&String> {
        self.models.keys().collect()
    }

    pub fn format_prompt(&self, template_name: &str, prompt: &str) -> anyhow::Result<String> {
        let template = self.prompt_templates.get(template_name)
            .ok_or_else(|| anyhow::anyhow!("Unknown prompt template: {}", template_name))?;

        let formatted = template.format.replace("{prompt}", prompt);
        Ok(formatted)
    }

    pub fn get_server_config(&self) -> ServerConfig {
        self.server.clone().unwrap_or_default()
    }

    pub fn get_session_settings(&self) -> SessionSettings {
        self.session.clone().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_loading() {
        let config_yaml = r#"
models:
  test_model:
    display_name: "Test Model"
    model_type: "qwen3"
    repo: "test/repo"
    filename: "test.gguf"
    revision: "main"
    tokenizer_repo: "test/tokenizer"
    prompt_template: "raw"
    eos_token: "<|endoftext|>"
    default_temperature: 0.8
    max_context_length: 2048

prompt_templates:
  raw:
    format: "{prompt}"

default_settings:
  sample_length: 1000
  temperature: 0.8
  repeat_penalty: 1.1
  repeat_last_n: 64
  seed: 299792458
  split_prompt: false
  tracing: false
"#;
        let config: Config = serde_yaml::from_str(config_yaml).unwrap();
        assert_eq!(config.models.len(), 1);
        assert!(config.models.contains_key("test_model"));
    }
}