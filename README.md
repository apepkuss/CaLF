<p align="center">
<img src="/docs/calf-logo.png" alt="CALF Logo" style="width:60%;"/>
</p>

# CALF: Candle-based LLM Inference Framework

## Usage

### List available models
```bash
cargo run --example unified-quantized -- --list-models
```

### Run single inference
```bash
# Use Qwen3 0.6B model
cargo run --example unified-quantized -- -m qwen3-0.6b --prompt "Explain quantum computing"

# Use Phi-2 model
cargo run --example unified-quantized -- -m phi-2 --prompt "Write a sorting algorithm"

# Use Phi-3 model
cargo run --example unified-quantized -- -m phi-3 --prompt "What is machine learning?"
```

### Interactive mode
```bash
cargo run --example unified-quantized -- -m qwen3-0.6b --interactive
```

### Custom configuration
```bash
cargo run --example unified-quantized -- -m qwen3-4b --config custom_models.yaml
```

### Override settings
```bash
cargo run --example unified-quantized -- -m phi-3 \
    --prompt "Explain Rust ownership" \
    --temperature 0.7 \
    --sample-len 500 \
    --top-p 0.9
```

## Configuration File Structure

The `models.yaml` file defines:

- **Models**: Model configurations including repos, files, templates
- **Prompt Templates**: Different formatting styles for various models
- **Default Settings**: Fallback values for inference parameters

### Adding a New Model

```yaml
models:
  my-new-model:
    display_name: "My New Model"
    model_type: "qwen3"  # or phi2, phi3, phi3b
    repo: "user/model-repo"
    filename: "model.gguf"
    revision: "main"
    tokenizer_repo: "user/tokenizer-repo"
    prompt_template: "raw"  # or any template defined below
    eos_token: "<|endoftext|>"
    default_temperature: 0.8
    max_context_length: 4096
```

### Adding a New Prompt Template

```yaml
prompt_templates:
  my-template:
    format: "Human: {prompt}\n\nAssistant: "
```

## Command Line Options

- `-m, --model`: Model name from configuration
- `-c, --config`: Configuration file path (default: models.yaml)
- `--local-model`: Use local GGUF file instead of downloading
- `--local-tokenizer`: Use local tokenizer file
- `--prompt`: Input prompt text
- `-n, --sample-len`: Maximum tokens to generate
- `--temperature`: Sampling temperature
- `--top-p`: Nucleus sampling parameter
- `--top-k`: Top-k sampling parameter
- `--seed`: Random seed
- `--interactive`: Enter interactive mode
- `--cpu`: Force CPU usage
- `--list-models`: Show available models

## Architecture

The example demonstrates several design patterns:

1. **Configuration-driven design**: All model-specific parameters externalized
2. **Trait-based abstraction**: `UnifiedModel` trait for consistent interface
3. **Strategy pattern**: Different prompt templates for different models
4. **Factory pattern**: `ModelLoader` for creating model instances
5. **Template method**: Shared inference logic with customizable parts

This design makes it easy to:
- Add new models without code changes
- Experiment with different prompt formats
- Share common inference logic
- Maintain model-specific configurations centrally

## Benefits Over Individual Examples

1. **DRY principle**: No code duplication across model examples
2. **Consistency**: Same interface and behavior for all models
3. **Maintainability**: Single place to fix bugs or add features
4. **Extensibility**: Easy to add new models, templates, or features
5. **Configuration management**: Centralized model parameters
6. **User experience**: Consistent CLI across all models