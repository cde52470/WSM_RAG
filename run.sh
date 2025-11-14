#!/bin/bash

# Try to get network time
RAW_NETWORK_TIME=$(curl -s http://worldtimeapi.org/api/timezone/Etc/UTC | jq -r '.utc_datetime')
NETWORK_TIMESTAMP=$(echo "${RAW_NETWORK_TIME}" | sed -e 's/[-T:]//g' -e 's/\..*//' | sed 's/\(........\)/\1_/')

# Fallback to system time if network time fails
if [ -z "$NETWORK_TIMESTAMP" ]; then
    echo "Warning: Failed to fetch network time. Falling back to system time."
    TIMESTAMP_BASE=$(date +"%Y%m%d_%H%M%S")
    TIMESTAMP="${TIMESTAMP_BASE}_localTime"
else
    TIMESTAMP="${NETWORK_TIMESTAMP}"
fi

PREDICTIONS_DIR="./predictions/${TIMESTAMP}"
RESULTS_DIR="./result/${TIMESTAMP}"

mkdir -p "${PREDICTIONS_DIR}"
mkdir -p "${RESULTS_DIR}"

python3 My_RAG/main.py --query_path ./dragonball_dataset/test_queries_zh.jsonl --docs_path ./dragonball_dataset/dragonball_docs.jsonl --language zh --output "${PREDICTIONS_DIR}/predictions_zh.jsonl"
python3 My_RAG/main.py --query_path ./dragonball_dataset/test_queries_en.jsonl --docs_path ./dragonball_dataset/dragonball_docs.jsonl --language en --output "${PREDICTIONS_DIR}/predictions_en.jsonl"

python3 rageval/evaluation/main.py --input_file "${PREDICTIONS_DIR}/predictions_zh.jsonl" --output_file "score_zh.jsonl" --language zh --output_dir "${RESULTS_DIR}"
python3 rageval/evaluation/main.py --input_file "${PREDICTIONS_DIR}/predictions_en.jsonl" --output_file "score_en.jsonl" --language en --output_dir "${RESULTS_DIR}"

python3 rageval/evaluation/process_intermediate.py --input_dir "${RESULTS_DIR}" --output_dir "${RESULTS_DIR}"