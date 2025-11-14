#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PREDICTIONS_DIR="./predictions/${TIMESTAMP}"
RESULTS_DIR="./result/${TIMESTAMP}"

mkdir -p "${PREDICTIONS_DIR}"
mkdir -p "${RESULTS_DIR}"

python3 My_RAG/main.py --query_path ./dragonball_dataset/test_queries_zh.jsonl --docs_path ./dragonball_dataset/dragonball_docs.jsonl --language zh --output "${PREDICTIONS_DIR}/predictions_zh.jsonl"
python3 My_RAG/main.py --query_path ./dragonball_dataset/test_queries_en.jsonl --docs_path ./dragonball_dataset/dragonball_docs.jsonl --language en --output "${PREDICTIONS_DIR}/predictions_en.jsonl"

python3 rageval/evaluation/main.py --input_file "${PREDICTIONS_DIR}/predictions_zh.jsonl" --output_file "score_zh.jsonl" --language zh --output_dir "${RESULTS_DIR}"
python3 rageval/evaluation/main.py --input_file "${PREDICTIONS_DIR}/predictions_en.jsonl" --output_file "score_en.jsonl" --language en --output_dir "${RESULTS_DIR}"

python3 rageval/evaluation/process_intermediate.py --input_dir "${RESULTS_DIR}" --output_dir "${RESULTS_DIR}"