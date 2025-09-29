#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::Parser;
use std::io::Write;

use candle_core::utils;

// 使用新的库接口
use calf::{CalfApiService, Config};

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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();

    // Load configuration first to handle list command
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

    // Get model configuration for display
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

    // 直接创建高层API服务
    let api_service = CalfApiService::new(&args.config, args.cpu).await?;

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
    let seed = args.seed;
    let split_prompt = args.split_prompt;
    let top_p = args.top_p;
    let top_k = args.top_k;

    if args.interactive {
        run_interactive_mode(
            &api_service,
            model_name,
            temperature,
            sample_len,
            repeat_penalty,
            repeat_last_n,
            seed,
            split_prompt,
            top_p,
            top_k,
        ).await?;
    } else {
        let prompt = args
            .prompt
            .ok_or_else(|| anyhow::anyhow!("Prompt is required for non-interactive mode"))?;

        run_single_inference(
            &api_service,
            model_name,
            &prompt,
            temperature,
            sample_len,
            repeat_penalty,
            repeat_last_n,
            seed,
            split_prompt,
            top_p,
            top_k,
        ).await?;
    }

    Ok(())
}

async fn run_single_inference(
    api_service: &CalfApiService,
    model_name: &str,
    prompt: &str,
    temperature: f64,
    max_tokens: usize,
    repeat_penalty: f32,
    repeat_last_n: usize,
    seed: Option<u64>,
    split_prompt: bool,
    top_p: Option<f64>,
    top_k: Option<usize>,
) -> anyhow::Result<()> {
    print!("Prompt: {}", prompt);
    print!(" -> ");
    std::io::stdout().flush()?;

    // Use the new engine interface through api service
    let result = api_service.engine().generate_text_simple(
        model_name,
        prompt,
        max_tokens,
        temperature,
        repeat_penalty,
        repeat_last_n,
        seed,
        split_prompt,
        top_p,
        top_k,
    ).await;

    match result {
        Ok(generated_text) => {
            println!("{}", generated_text);
            println!("Generated text successfully");
        }
        Err(e) => {
            eprintln!("Error generating text: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}

async fn run_interactive_mode(
    api_service: &CalfApiService,
    model_name: &str,
    temperature: f64,
    max_tokens: usize,
    repeat_penalty: f32,
    repeat_last_n: usize,
    seed: Option<u64>,
    split_prompt: bool,
    top_p: Option<f64>,
    top_k: Option<usize>,
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
            api_service,
            model_name,
            input,
            temperature,
            max_tokens,
            repeat_penalty,
            repeat_last_n,
            seed,
            split_prompt,
            top_p,
            top_k,
        ).await {
            eprintln!("Error: {}", e);
        }
    }

    Ok(())
}