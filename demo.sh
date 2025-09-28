#!/bin/bash

# Example usage script for unified-quantized

echo "=== Unified Quantized Model Runner Demo ==="
echo

# Check if in correct directory
if [ ! -f "models.yaml" ]; then
    echo "Error: Please run this script from the unified-quantized directory"
    echo "Expected to find models.yaml in current directory"
    exit 1
fi

echo "1. Listing available models..."
cargo run --release -- --list-models
echo

echo "2. Running Qwen3-0.6B with a simple prompt..."
cargo run --release -- -m qwen3-0.6b --prompt "Write a hello world program in Rust" --sample-len 200
echo

echo "3. Running Phi-2 with different temperature..."
cargo run --release -- -m phi-2 --prompt "Explain the concept of ownership in Rust" --temperature 0.5 --sample-len 150
echo

echo "4. Interactive mode example (Qwen3-4B)..."
echo "Note: Type 'quit' to exit interactive mode"
cargo run --release -- -m qwen3-4b --interactive

echo "Demo completed!"