import argparse
import json
import os
import sys

def main():
    """
    A script to check if the processed file has the same number of entries
    as the query file.
    """
    parser = argparse.ArgumentParser(description="Check output format.")
    parser.add_argument("--query_file", type=str, required=True, help="Path to the query file.")
    parser.add_argument("--processed_file", type=str, required=True, help="Path to the processed prediction file.")
    args = parser.parse_args()

    print(f"Checking format for {args.processed_file} against {args.query_file}...")

    if not os.path.exists(args.query_file):
        print(f"Error: Query file not found at {args.query_file}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.processed_file):
        print(f"Error: Processed file not found at {args.processed_file}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.query_file, 'r', encoding='utf-8') as f:
            query_lines = f.readlines()

        with open(args.processed_file, 'r', encoding='utf-8') as f:
            processed_lines = f.readlines()

        if len(query_lines) == len(processed_lines):
            print(f"Check successful: Number of lines match ({len(query_lines)}).")
            sys.exit(0)
        else:
            print(
                f"Error: Mismatch in number of lines. "
                f"Query file has {len(query_lines)} lines, but "
                f"processed file has {len(processed_lines)} lines.",
                file=sys.stderr
            )
            sys.exit(1)

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from a file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()