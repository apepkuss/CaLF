use crate::config::Config;
use crate::model::{ModelLoader, UnifiedModel};
use crate::types::{
    InferenceRequest, InferenceResponse, ModelInfo, ModelListResponse,
    InferenceError, InferenceResult,
};

use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH, Instant};
use tokenizers::Tokenizer;

/// CALF推理引擎 - 专注于核心推理能力
pub struct CalfInferenceEngine {
    /// 配置
    config: Config,
    /// 已加载的模型
    models: Arc<RwLock<HashMap<String, Box<dyn UnifiedModel>>>>,
    /// 已加载的分词器
    tokenizers: Arc<RwLock<HashMap<String, Tokenizer>>>,
    /// 计算设备
    device: Device,
    /// 是否强制使用CPU
    _force_cpu: bool,
}

impl CalfInferenceEngine {
    /// 创建新的推理引擎
    pub async fn new(config_path: &str, force_cpu: bool) -> InferenceResult<Self> {
        let config = Config::load_from_file(config_path)
            .map_err(|e| InferenceError::ConfigError(e.to_string()))?;

        let device = Self::setup_device(force_cpu)?;

        Ok(Self {
            config,
            models: Arc::new(RwLock::new(HashMap::new())),
            tokenizers: Arc::new(RwLock::new(HashMap::new())),
            device,
            _force_cpu: force_cpu,
        })
    }

    /// 设置计算设备
    fn setup_device(force_cpu: bool) -> InferenceResult<Device> {
        if force_cpu {
            Ok(Device::Cpu)
        } else if candle_core::utils::cuda_is_available() {
            Device::new_cuda(0).map_err(|e| InferenceError::Other(e.into()))
        } else if candle_core::utils::metal_is_available() {
            Device::new_metal(0).map_err(|e| InferenceError::Other(e.into()))
        } else {
            Ok(Device::Cpu)
        }
    }

    /// 加载模型（如果尚未加载）
    pub async fn ensure_model_loaded(&self, model_name: &str) -> InferenceResult<()> {
        // 检查模型是否已经加载
        {
            let models = self.models.read()
                .map_err(|_| InferenceError::Other(anyhow::anyhow!("Failed to acquire read lock")))?;
            if models.contains_key(model_name) {
                return Ok(());
            }
        }

        // 获取模型配置
        let model_config = self.config.get_model(model_name)
            .ok_or_else(|| InferenceError::ModelNotFound(model_name.to_string()))?;

        // 下载并加载模型
        let model_path = ModelLoader::download_model(model_config)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        let model = ModelLoader::load_model(model_config, model_path, &self.device)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        // 加载分词器
        let tokenizer_path = ModelLoader::download_tokenizer(model_config)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| InferenceError::TokenizationError(e.to_string()))?;

        // 存储模型和分词器
        {
            let mut models = self.models.write()
                .map_err(|_| InferenceError::Other(anyhow::anyhow!("Failed to acquire write lock")))?;
            models.insert(model_name.to_string(), model);
        }

        {
            let mut tokenizers = self.tokenizers.write()
                .map_err(|_| InferenceError::Other(anyhow::anyhow!("Failed to acquire write lock")))?;
            tokenizers.insert(model_name.to_string(), tokenizer);
        }

        Ok(())
    }

    /// 单次推理
    pub async fn inference(&self, request: InferenceRequest) -> InferenceResult<InferenceResponse> {
        let start_time = Instant::now();

        // 确保模型已加载
        self.ensure_model_loaded(&request.model).await?;

        // 获取模型配置
        let model_config = self.config.get_model(&request.model)
            .ok_or_else(|| InferenceError::ModelNotFound(request.model.clone()))?;

        // 格式化提示词
        let formatted_prompt = self.config.format_prompt(&model_config.prompt_template, &request.prompt)
            .map_err(|e| InferenceError::InvalidRequest(e.to_string()))?;

        // 执行推理
        let (generated_text, tokens_generated, finished_by_stop, stop_reason) =
            self.generate_text(&request.model, &formatted_prompt, &request).await?;

        let processing_time = start_time.elapsed().as_millis();

        Ok(InferenceResponse {
            text: generated_text,
            tokens_generated,
            processing_time_ms: processing_time,
            model: request.model,
            finished_by_stop,
            stop_reason,
        })
    }

    /// 计算嵌入向量
    pub async fn compute_embeddings(&self, model_name: &str, text: &str) -> InferenceResult<Vec<f32>> {
        // 确保模型已加载
        self.ensure_model_loaded(model_name).await?;

        // 获取分词器
        let tokenizer = {
            let tokenizers = self.tokenizers.read()
                .map_err(|_| InferenceError::Other(anyhow::anyhow!("Failed to acquire tokenizer read lock")))?;
            tokenizers.get(model_name)
                .ok_or_else(|| InferenceError::ModelNotFound(model_name.to_string()))?
                .clone()
        };

        // 分词
        let tokens = tokenizer
            .encode(text, true)
            .map_err(|e| InferenceError::TokenizationError(e.to_string()))?
            .get_ids()
            .to_vec();

        let input = Tensor::new(tokens.as_slice(), &self.device)
            .map_err(|e| InferenceError::InferenceFailure(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| InferenceError::InferenceFailure(e.to_string()))?;

        // 获取模型进行前向传播
        let embeddings = {
            let mut models = self.models.write()
                .map_err(|_| InferenceError::Other(anyhow::anyhow!("Failed to acquire model write lock")))?;
            let model = models.get_mut(model_name)
                .ok_or_else(|| InferenceError::ModelNotFound(model_name.to_string()))?;

            // 注意：这里需要模型支持嵌入向量计算
            // 具体实现取决于模型类型，这里是一个简化版本
            model.forward(&input, 0)
                .map_err(|e| InferenceError::InferenceFailure(e.to_string()))?
        };

        // 转换为Vec<f32> - 这里需要根据实际模型输出格式调整
        let embeddings_flat = embeddings
            .flatten_all()
            .map_err(|e| InferenceError::InferenceFailure(e.to_string()))?;

        let embeddings_vec = embeddings_flat
            .to_vec1::<f32>()
            .map_err(|e| InferenceError::InferenceFailure(e.to_string()))?;

        Ok(embeddings_vec)
    }

    /// 检查模型是否存在
    pub fn has_model(&self, model_name: &str) -> bool {
        self.config.models.contains_key(model_name)
    }

    /// 列出可用模型
    pub fn list_models(&self) -> ModelListResponse {
        let models: Vec<ModelInfo> = self.config.models.iter()
            .map(|(id, config)| ModelInfo {
                id: id.clone(),
                name: config.display_name.clone(),
                model_type: format!("{:?}", config.model_type),
                max_context_length: config.max_context_length,
                default_temperature: config.default_temperature,
                available: true,
            })
            .collect();

        ModelListResponse {
            object: "list".to_string(),
            data: models,
        }
    }



    /// 简化的文本生成方法（用于CLI）
    pub async fn generate_text_simple(
        &self,
        model_name: &str,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        repeat_penalty: f32,
        repeat_last_n: usize,
        seed: Option<u64>,
        split_prompt: bool,
        top_p: Option<f64>,
        top_k: Option<usize>,
    ) -> InferenceResult<String> {
        // 确保模型已加载
        self.ensure_model_loaded(model_name).await?;

        let request = InferenceRequest {
            model: model_name.to_string(),
            prompt: prompt.to_string(),
            max_tokens,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            repeat_last_n,
            seed,
            split_prompt,
            stream: false,
            stop: None,
        };

        // 格式化提示词
        let model_config = self.config.get_model(model_name)
            .ok_or_else(|| InferenceError::ModelNotFound(model_name.to_string()))?;

        let formatted_prompt = self.config.format_prompt(&model_config.prompt_template, prompt)
            .map_err(|e| InferenceError::InvalidRequest(e.to_string()))?;

        let (generated_text, _, _, _) = self.generate_text(model_name, &formatted_prompt, &request).await?;
        Ok(generated_text)
    }

    // 私有辅助方法

    /// 生成文本
    async fn generate_text(
        &self,
        model_name: &str,
        prompt: &str,
        request: &InferenceRequest,
    ) -> InferenceResult<(String, usize, bool, Option<String>)> {
        // 获取分词器
        let tokenizer = {
            let tokenizers = self.tokenizers.read()
                .map_err(|_| InferenceError::Other(anyhow::anyhow!("Failed to acquire tokenizer read lock")))?;
            tokenizers.get(model_name)
                .ok_or_else(|| InferenceError::ModelNotFound(model_name.to_string()))?
                .clone()
        };

        // 获取模型配置
        let model_config = self.config.get_model(model_name)
            .ok_or_else(|| InferenceError::ModelNotFound(model_name.to_string()))?;

        // 分词
        let tokens = tokenizer
            .encode(prompt, true)
            .map_err(|e| InferenceError::TokenizationError(e.to_string()))?
            .get_ids()
            .to_vec();

        let mut tokens = if request.split_prompt {
            // 实现split_prompt逻辑 - 为每个token单独进行前向传播
            let mut all_tokens = Vec::new();
            for token in tokens.iter() {
                let _input_ids = Tensor::new(&[*token], &self.device)
                    .map_err(|e| InferenceError::InferenceFailure(e.to_string()))?
                    .unsqueeze(0)
                    .map_err(|e| InferenceError::InferenceFailure(e.to_string()))?;

                // 注意：这里需要模型的可变引用，暂时先处理token
                all_tokens.push(*token);
            }
            all_tokens
        } else {
            tokens
        };

        // 设置采样器
        let seed = request.seed.unwrap_or_else(|| {
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
        });

        let mut logits_processor = LogitsProcessor::new(
            seed,
            Some(request.temperature),
            request.top_p,
        );

        let mut generated_tokens = 0;
        let mut tos = TokenOutputStream::new(tokenizer.clone());

        // 获取EOS token
        let eos_token = *tokenizer
            .get_vocab(true)
            .get(&model_config.eos_token)
            .unwrap_or(&0);

        // 开始生成循环 - 这里需要模型的可变引用
        // 由于当前架构使用RwLock，需要在每次循环中获取写锁
        // 这不是最优的设计，但可以工作
        for index in 0..request.max_tokens {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let context_index = tokens.len().saturating_sub(context_size);
            let input = Tensor::new(&tokens[context_index..], &self.device)
                .map_err(|e| InferenceError::InferenceFailure(e.to_string()))?
                .unsqueeze(0)
                .map_err(|e| InferenceError::InferenceFailure(e.to_string()))?;

            // 获取模型进行前向传播
            let logits = {
                let mut models = self.models.write()
                    .map_err(|_| InferenceError::Other(anyhow::anyhow!("Failed to acquire model write lock")))?;
                let model = models.get_mut(model_name)
                    .ok_or_else(|| InferenceError::ModelNotFound(model_name.to_string()))?;

                model.forward(&input, tokens.len().saturating_sub(1))
                    .map_err(|e| InferenceError::InferenceFailure(e.to_string()))?
            };

            // 应用重复惩罚
            let logits = if request.repeat_penalty == 1.0 {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(request.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    request.repeat_penalty,
                    &tokens[start_at..],
                ).map_err(|e| InferenceError::InferenceFailure(e.to_string()))?
            };

            // 采样下一个token
            let next_token = logits_processor.sample(&logits)
                .map_err(|e| InferenceError::InferenceFailure(e.to_string()))?;

            tokens.push(next_token);
            generated_tokens += 1;

            // 解码token并添加到输出流
            tos.next_token(next_token)
                .map_err(|e| InferenceError::TokenizationError(e.to_string()))?;

            // 检查是否遇到停止token
            if next_token == eos_token {
                return Ok((tos.get_current_text().to_string(), generated_tokens, true, Some("eos".to_string())));
            }

            // 检查停止词
            if let Some(ref stop_words) = request.stop {
                let current_text = tos.get_current_text();
                for stop_word in stop_words {
                    if current_text.contains(stop_word) {
                        return Ok((current_text.to_string(), generated_tokens, true, Some(stop_word.clone())));
                    }
                }
            }
        }

        // 达到最大长度限制
        Ok((tos.get_current_text().to_string(), generated_tokens, false, Some("length".to_string())))
    }



    /// 计算tokens数量
    pub async fn count_tokens(&self, model_name: &str, text: &str) -> InferenceResult<usize> {
        let tokenizers = self.tokenizers.read()
            .map_err(|_| InferenceError::Other(anyhow::anyhow!("Failed to acquire read lock")))?;

        let tokenizer = tokenizers.get(model_name)
            .ok_or_else(|| InferenceError::ModelNotFound(model_name.to_string()))?;

        let encoding = tokenizer.encode(text, false)
            .map_err(|e| InferenceError::TokenizationError(e.to_string()))?;

        Ok(encoding.len())
    }
}

// Token输出流（从main.rs移植过来）
pub struct TokenOutputStream {
    tokenizer: Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    pub current_text: String,
}

impl TokenOutputStream {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_text: String::new(),
        }
    }

    pub fn next_token(&mut self, token: u32) -> anyhow::Result<String> {
        self.tokens.push(token);
        let text = self.tokenizer.decode(&self.tokens[self.prev_index..], false)
            .map_err(|e| anyhow::anyhow!("Tokenizer decode error: {}", e))?;
        self.current_text.push_str(&text);
        self.prev_index = self.tokens.len();
        Ok(text)
    }

    pub fn get_tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn get_current_text(&self) -> &str {
        &self.current_text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_has_model() {
        use crate::config::ModelConfig;
        // 创建测试配置
        let mut models = HashMap::new();
        models.insert("test-model".to_string(), ModelConfig {
            display_name: "Test Model".to_string(),
            model_type: crate::config::ModelType::Qwen3,
            repo: "test-repo".to_string(),
            filename: "test.gguf".to_string(),
            revision: "main".to_string(),
            tokenizer_repo: "test-tokenizer".to_string(),
            prompt_template: "raw".to_string(),
            eos_token: "<|endoftext|>".to_string(),
            default_temperature: 0.8,
            max_context_length: 2048,
            features: Default::default(),
        });

        let config = Config {
            models,
            prompt_templates: HashMap::new(),
            default_settings: Default::default(),
            server: None,
            session: None,
        };

        let engine = CalfInferenceEngine {
            config,
            models: Arc::new(RwLock::new(HashMap::new())),
            tokenizers: Arc::new(RwLock::new(HashMap::new())),
            device: Device::Cpu,
            _force_cpu: true,
        };

        assert!(engine.has_model("test-model"));
        assert!(!engine.has_model("non-existent-model"));
    }
}