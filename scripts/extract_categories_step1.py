#!/usr/bin/env python3
"""
Script to extract unique categories from all step1 dictionaries.
"""

import json
import glob
import os

def main():
    input_dir = 'data/dictionaries/combined_step1'
    pattern = os.path.join(input_dir, '*.json')
    files = glob.glob(pattern)
    categories = set()
    for fn in files:
        try:
            with open(fn, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue

        # Extract categories from keys
        if isinstance(data.get('categories'), dict):
            categories.update(data['categories'].keys())
        elif isinstance(data.get('word_sets'), dict):
            categories.update(data['word_sets'].keys())

    output_path = os.path.join(input_dir, 'unique_categories.txt')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for cat in sorted(categories):
            f.write(f"{cat}\n")

    print(f"Unique categories compiled into {output_path}")

if __name__ == '__main__':
    main()