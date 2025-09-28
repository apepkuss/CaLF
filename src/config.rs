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
pub struct Config {
    pub models: HashMap<String, ModelConfig>,
    pub prompt_templates: HashMap<String, PromptTemplate>,
    pub default_settings: DefaultSettings,
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