#!/bin/bash

# Simple demo showing how to use the unified quantized model runner
echo "=== Unified Quantized Model Runner - Simple Demo ==="

# Check if cargo is installed
if ! command -v cargo &> /dev/null; then
    echo "Error: Cargo is not installed or not in PATH"
    exit 1
fi

# Set the working directory to the example location
cd "$(dirname "$0")"

echo "Current directory: $(pwd)"
echo "Available models in simple_models.yaml:"
echo

# Show available models
cargo run -- --config simple_models.yaml --list 2>/dev/null | tail -n +2

echo
echo "Example 1: Single inference with qwen3-small"
echo "Command: cargo run -- --config simple_models.yaml --model qwen3-small --prompt \"What is machine learning?\""
echo

cargo run -- --config simple_models.yaml --model qwen3-small --prompt "What is machine learning?" 2>/dev/null

echo
echo "Example 2: Simple conversation with phi2-basic"
echo "Command: cargo run -- --config simple_models.yaml --model phi2-basic --prompt \"Explain quantum computing briefly\""
echo

cargo run -- --config simple_models.yaml --model phi2-basic --prompt "Explain quantum computing briefly" 2>/dev/null

echo
echo "=== Demo Complete ==="
echo "To start interactive mode with qwen3-small: cargo run -- --config simple_models.yaml --model qwen3-small"
echo "To start interactive mode with phi2-basic: cargo run -- --config simple_models.yaml --model phi2-basic"