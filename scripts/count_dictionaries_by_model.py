#!/usr/bin/env python3
"""
Count dictionaries by model and step to identify missing ones.
"""

import json
from pathlib import Path
from collections import defaultdict

# Load metadata
data_dir = Path("/home5/chuenho/gpr-zero/data/gpr_correlations")
with open(data_dir / "dictionary_metadata.json", 'r') as f:
    metadata = json.load(f)

# Count by model and step
counts = defaultdict(lambda: defaultdict(int))
dict_lists = defaultdict(lambda: defaultdict(list))

for name, meta in metadata.items():
    model = meta['model']
    step = meta['step']
    counts[model][step] += 1
    dict_lists[model][step].append(name)

print("=== Dictionary Count by Model and Step ===")
print("Expected: Step1=10 (5 roles × 2 runs), Step2=5 (5 runs)")
print()

total_by_model = {}
for model in sorted(counts.keys()):
    step1_count = counts[model]['step1']
    step2_count = counts[model]['step2']
    total = step1_count + step2_count
    total_by_model[model] = total
    
    print(f"{model}:")
    print(f"  Step1: {step1_count}")
    print(f"  Step2: {step2_count}")
    print(f"  Total: {total}")
    
    if total != 15:
        print(f"  ⚠️  MISSING: Expected 15, found {total}")
    print()

print("=== GPT-4.1 Dictionary Details ===")
if 'gpt-4.1' in dict_lists:
    print("Step1 dictionaries:")
    for d in sorted(dict_lists['gpt-4.1']['step1']):
        print(f"  {d}")
    print("\nStep2 dictionaries:")
    for d in sorted(dict_lists['gpt-4.1']['step2']):
        print(f"  {d}")

print("\n=== Missing Dictionary Analysis ===")
expected_roles = ['journalist', 'president', 'poli_professor', 'defense_minister', 'nsb_director']
expected_runs = ['run1', 'run2']

for model in ['gpt-4.1', 'gpt-4o']:
    print(f"\n{model} missing analysis:")
    
    # Check Step1 (should have 5 roles × 2 runs = 10)
    step1_dicts = dict_lists[model]['step1']
    step1_combos = set()
    for d in step1_dicts:
        parts = d.split('_')
        if len(parts) >= 4:
            role = '_'.join(parts[2:-1])  # Handle multi-word roles
            run = parts[-1]
            step1_combos.add((role, run))
    
    print(f"  Step1 found combinations: {len(step1_combos)}/10")
    expected_step1 = {(role, run) for role in expected_roles for run in expected_runs}
    missing_step1 = expected_step1 - step1_combos
    if missing_step1:
        print(f"  Missing Step1: {sorted(missing_step1)}")
    
    # Check Step2 (should have 5 runs)
    step2_dicts = dict_lists[model]['step2']
    print(f"  Step2 found: {len(step2_dicts)}/5")
    if len(step2_dicts) < 5:
        print(f"  Step2 dictionaries: {sorted(step2_dicts)}")

# Check if taiwan dictionaries are misclassified
print("\n=== Taiwan Dictionaries (might be misclassified GPT-4.1) ===")
taiwan_dicts = [name for name, meta in metadata.items() if meta['model'] == 'taiwan']
for d in taiwan_dicts:
    print(f"  {d}: {metadata[d]}")