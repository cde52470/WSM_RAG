import json
import os
import argparse
from typing import Dict, List, Any
from pathlib import Path

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def calculate_averages(data: List[Dict[str, Any]], metric_list: List[str]) -> Dict[str, float]:
    metric_sums = {metric: 0 for metric in metric_list}
    metric_counts = {metric: 0 for metric in metric_list}
    
    for item in data:
        for metric in metric_list:
            if metric in item:
                metric_sums[metric] += item[metric]
                metric_counts[metric] += 1
    #only return the average of the metrics that the metric_counts is not 0
    return {metric: metric_sums[metric] / metric_counts[metric] for metric in metric_list if metric_counts[metric] != 0}


def process_folder(folder_path: str, output_file: str, metric_list: List[str]):
    results = {}
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(folder_path, filename)
            data = load_jsonl(file_path)
            averages = calculate_averages(data, metric_list)
            results[filename] = averages
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process intermediate evaluation results.")
    parser.add_argument("--input_dir", type=str, default='./result', help="Directory containing intermediate JSONL files.")
    parser.add_argument("--output_dir", type=str, default='./result', help="Directory to save the final aggregated results.")
    
    args = parser.parse_args()

    folder_path = args.input_dir
    output_file_name = 'final_result.json'
    output_file = Path(args.output_dir) / output_file_name
    
    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    metric_list = ['EIR', 'Precision', 'Recall', 'ROUGELScore', "completeness", "hallucination", "irrelevance"]
    
    process_folder(folder_path, output_file, metric_list)
    print(f"Results saved to {output_file}")