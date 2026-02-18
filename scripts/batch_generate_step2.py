#!/usr/bin/env python3
"""
Batch-generate GPR dictionaries (step 2):
Run the stable 8-category prompt across multiple LLM providers.
Generates 4 models × 5 runs = 20 dictionaries total.
"""

import json
from pathlib import Path
import time
import random

import os
from dotenv import load_dotenv
load_dotenv()

from gpr_zero.llm.dictionary_generator import DictionaryGenerator
from gpr_zero.llm.prompt_manager import PromptManager

# Configuration: providers and run counts
PROVIDERS = {
    "gpt-4.1": {"model": "openai/gpt-4.1", "runs": 5},
    "gpt-4o": {"model": "openai/gpt-4o-2024-11-20", "runs": 5},
    "gemini-2.5-pro": {"model": "google/gemini-2.5-pro", "runs": 5},
    "claude-opus-4": {"model": "anthropic/claude-opus-4", "runs": 5},
}

# Path to the stable 8-category prompt template
PROMPT_FILE = Path("prompts/generate_taiwan_gpr_dictionary_stable_8categories.json")

# Output directory for generated dictionaries
OUTPUT_DIR = Path("data/dictionaries/step2_batch")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def validate_dictionary(dictionary: dict, provider_key: str, run_num: int) -> bool:
    """Validate that generated dictionary meets requirements."""
    try:
        # Check required fields
        if 'categories' not in dictionary or 'word_sets' not in dictionary:
            print(f"  ✗ Missing required fields for {provider_key} run {run_num}")
            return False
        
        # Check category count
        categories = dictionary['categories']
        if len(categories) != 8:
            print(f"  ✗ Expected 8 categories, got {len(categories)} for {provider_key} run {run_num}")
            return False
        
        # Check word sets
        word_sets = dictionary['word_sets']
        for word_set_name, words in word_sets.items():
            if len(words) < 15:
                print(f"  ⚠ Word set '{word_set_name}' has only {len(words)} words (min 15)")
            if len(words) > 25:
                print(f"  ⚠ Word set '{word_set_name}' has {len(words)} words (max 25)")
        
        # Check that all word sets referenced in categories exist
        for cat_name, cat_logic in categories.items():
            # Extract word set names from category logic
            parts = cat_logic.split(' AND ')
            for part in parts:
                part = part.strip()
                if part and part not in word_sets:
                    print(f"  ✗ Category '{cat_name}' references non-existent word set '{part}'")
                    return False
        
        return True
    except Exception as e:
        print(f"  ✗ Validation error: {e}")
        return False

def main():
    # Initialize PromptManager and DictionaryGenerator
    pm = PromptManager()
    gen = DictionaryGenerator(prompt_manager=pm, output_dir=str(OUTPUT_DIR))

    # Track statistics
    total_runs = sum(cfg["runs"] for cfg in PROVIDERS.values())
    successful = 0
    failed = 0
    
    print(f"Starting batch generation of {total_runs} dictionaries")
    print(f"Output directory: {OUTPUT_DIR}\n")

    for provider_key, cfg in PROVIDERS.items():
        model_name = cfg["model"]
        run_count = cfg["runs"]
        
        print(f"\n{'='*60}")
        print(f"Provider: {provider_key} (Model: {model_name})")
        print(f"{'='*60}")

        for run_num in range(1, run_count + 1):
            output_name = f"{provider_key}_stable8cat_run{run_num}"
            print(f"\n[{provider_key} Run {run_num}/{run_count}] Generating: {output_name}")
            
            try:
                # Generate dictionary
                dictionary = gen.generate_dictionary(
                    strategy="prompt_file",
                    prompt_file_path=PROMPT_FILE,
                    template_args_list=[],  # No template args needed
                    model=model_name,
                    temperature=None,  # Use temperature from prompt file (0.2)
                    name=output_name
                )
                
                if dictionary and validate_dictionary(dictionary, provider_key, run_num):
                    print(f"  ✓ Successfully generated and validated")
                    successful += 1
                    
                    # Save metadata summary
                    summary_path = OUTPUT_DIR / f"{output_name}_meta.json"
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(dictionary.get("metadata", {}), f, ensure_ascii=False, indent=2)
                else:
                    print(f"  ✗ Failed validation")
                    failed += 1
                    
            except Exception as e:
                print(f"  ✗ Error generating dictionary: {e}")
                failed += 1
            
            # Add delay between API calls to respect rate limits
            if run_num < run_count or provider_key != list(PROVIDERS.keys())[-1]:
                delay = random.uniform(2, 5)
                print(f"  ⏳ Waiting {delay:.1f}s before next generation...")
                time.sleep(delay)
    
    # Summary
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total attempts: {total_runs}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # List generated files
    generated_files = list(OUTPUT_DIR.glob("*_stable8cat_run*.json"))
    if generated_files:
        print(f"\nGenerated {len(generated_files)} dictionary files:")
        for f in sorted(generated_files)[:10]:
            print(f"  - {f.name}")
        if len(generated_files) > 10:
            print(f"  ... and {len(generated_files) - 10} more")

if __name__ == "__main__":
    main()