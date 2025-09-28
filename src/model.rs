use candle_core::{Device, Result, Tensor};
use crate::config::{ModelConfig, ModelType};

// 统一模型抽象trait
pub trait UnifiedModel: Send + Sync {
    fn forward(&mut self, input: &Tensor, pos: usize) -> Result<Tensor>;
    fn get_config(&self) -> &ModelConfig;
}

// 模型包装器枚举
pub enum ModelWrapper {
    Qwen3(Box<candle_transformers::models::quantized_qwen3::ModelWeights>),
    Phi2(Box<candle_transformers::models::quantized_phi::ModelWeights>),
    Phi3(Box<candle_transformers::models::quantized_phi3::ModelWeights>),
    Phi3b(Box<candle_transformers::models::quantized_llama::ModelWeights>),
}

pub struct UnifiedModelInstance {
    model: ModelWrapper,
    config: ModelConfig,
}

impl UnifiedModelInstance {
    pub fn new(model: ModelWrapper, config: ModelConfig) -> Self {
        Self { model, config }
    }
}

impl UnifiedModel for UnifiedModelInstance {
    fn forward(&mut self, input: &Tensor, pos: usize) -> Result<Tensor> {
        match &mut self.model {
            ModelWrapper::Qwen3(model) => model.forward(input, pos),
            ModelWrapper::Phi2(model) => model.forward(input, pos),
            ModelWrapper::Phi3(model) => model.forward(input, pos),
            ModelWrapper::Phi3b(model) => model.forward(input, pos),
        }
    }

    fn get_config(&self) -> &ModelConfig {
        &self.config
    }
}

// 模型加载器
pub struct ModelLoader;

impl ModelLoader {
    pub fn load_model(
        config: &ModelConfig,
        model_path: std::path::PathBuf,
        device: &Device,
    ) -> anyhow::Result<Box<dyn UnifiedModel>> {
        let mut file = std::fs::File::open(&model_path)?;
        let gguf_content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| e.with_path(&model_path))?;

        let model_wrapper = match config.model_type {
            ModelType::Qwen3 => {
                let model = candle_transformers::models::quantized_qwen3::ModelWeights::from_gguf(
                    gguf_content, &mut file, device
                )?;
                ModelWrapper::Qwen3(Box::new(model))
            }
            ModelType::Phi2 => {
                let model = candle_transformers::models::quantized_phi::ModelWeights::from_gguf(
                    gguf_content, &mut file, device
                )?;
                ModelWrapper::Phi2(Box::new(model))
            }
            ModelType::Phi3 => {
                let model = candle_transformers::models::quantized_phi3::ModelWeights::from_gguf(
                    config.features.flash_attention,
                    gguf_content, &mut file, device
                )?;
                ModelWrapper::Phi3(Box::new(model))
            }
            ModelType::Phi3b => {
                let model = candle_transformers::models::quantized_llama::ModelWeights::from_gguf(
                    gguf_content, &mut file, device
                )?;
                ModelWrapper::Phi3b(Box::new(model))
            }
        };

        let unified_model = UnifiedModelInstance::new(model_wrapper, config.clone());
        Ok(Box::new(unified_model))
    }

    pub fn download_model(config: &ModelConfig) -> anyhow::Result<std::path::PathBuf> {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.repo(hf_hub::Repo::with_revision(
            config.repo.clone(),
            hf_hub::RepoType::Model,
            config.revision.clone(),
        ));
        let model_path = repo.get(&config.filename)?;
        Ok(model_path)
    }

    pub fn download_tokenizer(config: &ModelConfig) -> anyhow::Result<std::path::PathBuf> {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(config.tokenizer_repo.clone());
        let tokenizer_path = repo.get("tokenizer.json")?;
        Ok(tokenizer_path)
    }
}

pub fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{size_in_bytes}B")
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
}

pub fn print_model_info(
    tensor_count: usize,
    total_size: usize,
    load_time: f32,
    model_name: &str,
) {
    println!(
        "Loaded {} ({}) tensors ({}) for {} in {:.2}s",
        tensor_count,
        model_name,
        format_size(total_size),
        model_name,
        load_time,
    );
}