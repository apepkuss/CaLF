#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::Parser;
use std::io::{stdout, Write};
use tokenizers::Tokenizer;

use candle_core::{Device, Tensor, utils};
use candle_transformers::generation::{LogitsProcessor, Sampling};

// Simple token output handler
struct TokenOutputStream {
    tokenizer: Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
}

impl TokenOutputStream {
    fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
        }
    }

    fn next_token(&mut self, token: u32) -> anyhow::Result<()> {
        self.tokens.push(token);
        let text = self.tokenizer.decode(&self.tokens[self.prev_index..], false)
            .map_err(|e| anyhow::anyhow!("Tokenizer decode error: {}", e))?;
        print!("{}", text);
        stdout().flush()?;
        self.prev_index = self.tokens.len();
        Ok(())
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn get_tokens(&self) -> &[u32] {
        &self.tokens
    }
}

fn device(cpu: bool) -> anyhow::Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle_core::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle_core::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

mod config;
mod model;

use config::Config;
use model::{ModelLoader, UnifiedModel};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model name from configuration file
    #[arg(short, long)]
    model: Option<String>,

    /// Configuration file path
    #[arg(short, long, default_value = "models.yaml")]
    config: String,

    /// Local GGUF file to load (overrides config)
    #[arg(long)]
    local_model: Option<String>,

    /// Local tokenizer file (overrides config)
    #[arg(long)]
    local_tokenizer: Option<String>,

    /// The initial prompt
    #[arg(long)]
    prompt: Option<String>,

    /// The length of the sample to generate (in tokens)
    #[arg(short = 'n', long)]
    sample_len: Option<usize>,

    /// The temperature used to generate samples
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples
    #[arg(long)]
    seed: Option<u64>,

    /// Enable tracing (generates a trace-timestamp.json file)
    #[arg(long)]
    tracing: bool,

    /// Process prompt elements separately
    #[arg(long)]
    split_prompt: bool,

    /// Run on CPU rather than GPU even if a GPU is available
    #[arg(long)]
    cpu: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty
    #[arg(long)]
    repeat_penalty: Option<f32>,

    /// The context size to consider for the repeat penalty
    #[arg(long)]
    repeat_last_n: Option<usize>,

    /// List available models
    #[arg(long)]
    list: bool,

    /// Interactive mode
    #[arg(long)]
    interactive: bool,
}

fn main() -> anyhow::Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();

    // Load configuration
    let config = Config::load_from_file(&args.config)?;

    // Handle list models command
    if args.list {
        println!("Available models:");
        for (model_name, model_config) in &config.models {
            println!("  {}: {} ({:?})", model_name, model_config.display_name, model_config.model_type);
        }
        return Ok(());
    }

    let model_name = match args.model {
        Some(ref name) => name,
        None => return Err(anyhow::anyhow!("Model name is required. Use --model <model_name> or --list to see available models")),
    };

    // Get model configuration
    let model_config = config
        .models
        .get(model_name)
        .ok_or_else(|| anyhow::anyhow!("Model '{}' not found in configuration", model_name))?;

    println!("Loading model: {}", model_config.display_name);

    // Setup tracing
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    // Print system info
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        utils::with_avx(),
        utils::with_neon(),
        utils::with_simd128(),
        utils::with_f16c()
    );

    // Setup device and model
    let device = device(args.cpu)?;
    // First need to download model
    let model_path = ModelLoader::download_model(model_config)?;
    let mut model = ModelLoader::load_model(model_config, model_path, &device)?;

    // Load tokenizer
    let tokenizer_path = ModelLoader::download_tokenizer(model_config)?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Setup parameters with config defaults
    let temperature = args.temperature
        .or(Some(model_config.default_temperature))
        .unwrap_or(config.default_settings.temperature);
    let sample_len = args.sample_len
        .unwrap_or(config.default_settings.sample_length);
    let repeat_penalty = args.repeat_penalty
        .unwrap_or(config.default_settings.repeat_penalty);
    let repeat_last_n = args.repeat_last_n
        .unwrap_or(config.default_settings.repeat_last_n);
    let seed = args.seed
        .unwrap_or(config.default_settings.seed);
    let split_prompt = args.split_prompt;

    if args.interactive {
        run_interactive_mode(
            &mut model,
            &tokenizer,
            &config,
            model_config,
            temperature,
            seed,
            repeat_penalty,
            repeat_last_n,
            sample_len,
            split_prompt,
            &device,
        )?;
    } else {
        let prompt = args
            .prompt
            .ok_or_else(|| anyhow::anyhow!("Prompt is required for non-interactive mode"))?;

        run_single_inference(
            &mut model,
            &tokenizer,
            &config,
            model_config,
            &prompt,
            temperature,
            seed,
            repeat_penalty,
            repeat_last_n,
            sample_len,
            split_prompt,
            &device,
        )?;
    }

    Ok(())
}

fn run_single_inference(
    model: &mut Box<dyn UnifiedModel>,
    tokenizer: &Tokenizer,
    config: &Config,
    model_config: &config::ModelConfig,
    prompt: &str,
    temperature: f64,
    seed: u64,
    repeat_penalty: f32,
    repeat_last_n: usize,
    sample_len: usize,
    split_prompt: bool,
    device: &Device,
) -> anyhow::Result<()> {
    // Format prompt using template
    let formatted_prompt = config.format_prompt(&model_config.prompt_template, prompt)?;
    print!("Prompt: {}", formatted_prompt);

    // Tokenize
    let mut tos = TokenOutputStream::new(tokenizer.clone());
    let tokens = tokenizer
        .encode(formatted_prompt, true)
        .map_err(|e| anyhow::anyhow!("Tokenizer encode error: {}", e))?
        .get_ids()
        .to_vec();

    let mut tokens = if split_prompt {
        let mut all_tokens = Vec::new();
        for token in tokens.iter() {
            let input_ids = Tensor::new(&[*token], device)?.unsqueeze(0)?;
            let _logits = model.forward(&input_ids, 0)?;
            all_tokens.push(*token);
        }
        all_tokens
    } else {
        tokens
    };

    // Setup sampling
    let mut logits_processor = LogitsProcessor::new(
        seed,
        Some(temperature),
        None, // top_p
    );

    print!(" -> ");
    for index in 0..sample_len {
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let context_index = tokens.len().saturating_sub(context_size);
        let input = Tensor::new(&tokens[context_index..], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, tokens.len().saturating_sub(1))?;

        let logits = if repeat_penalty == 1.0 {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                repeat_penalty,
                &tokens[start_at..],
            )?
        };

        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        tos.next_token(next_token)?;

        let eos_token = *tokenizer
            .get_vocab(true)
            .get(&model_config.eos_token)
            .unwrap_or(&0);

        if next_token == eos_token {
            break;
        }
    }

    println!();
    println!("Generated {} tokens", tokens.len() - 1);

    Ok(())
}

fn run_interactive_mode(
    model: &mut Box<dyn UnifiedModel>,
    tokenizer: &Tokenizer,
    config: &Config,
    model_config: &config::ModelConfig,
    temperature: f64,
    seed: u64,
    repeat_penalty: f32,
    repeat_last_n: usize,
    sample_len: usize,
    split_prompt: bool,
    device: &Device,
) -> anyhow::Result<()> {
    println!("Interactive mode. Type 'quit' to exit.");

    loop {
        print!("\n> ");
        std::io::stdout().flush()?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input == "quit" || input == "exit" {
            break;
        }

        if input.is_empty() {
            continue;
        }

        if let Err(e) = run_single_inference(
            model,
            tokenizer,
            config,
            model_config,
            input,
            temperature,
            seed,
            repeat_penalty,
            repeat_last_n,
            sample_len,
            split_prompt,
            device,
        ) {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}