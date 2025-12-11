#!/bin/bash

# Set debug mode
set -e
# set -x

# Set the OLLAMA_HOST environment variable for local execution
export OLLAMA_HOST=${OLLAMA_HOST:-"http://localhost:11434"}

log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local message="$timestamp - $1"
    local len=${#message}
    # Using printf for border creation
    local border=$(printf '=%.0s' $(seq 1 $len))
    
    echo "$border"
    echo "$message"
    echo "$border"
}

# Ensure the base predictions directory exists
mkdir -p ./predictions

run_results() {
    local language=$1

    log "[INFO] Running inference for language: ${language}"
    python ./My_RAG/main.py \
        --query_path ./dragonball_dataset/queries_show/queries_${language}.jsonl \
        --docs_path ./dragonball_dataset/dragonball_docs.jsonl \
        --language ${language} \
        --output ./predictions/predictions_${language}.jsonl

    log "[INFO] Checking output format for language: ${language}"
    python ./check_output_format.py \
        --query_file ./dragonball_dataset/queries_show/queries_${language}.jsonl \
        --processed_file ./predictions/predictions_${language}.jsonl

    if [ $? -eq 0 ]; then
        echo "Format check passed for ${language}."
    fi
}

run_results "en"
run_results "zh"
log "[INFO] All inference tasks completed."