#!/usr/bin/env python3
"""
Batch-generate GPR dictionaries (step 1):
Run the current role-based prompt across multiple LLM providers & runs.
"""

import json
from pathlib import Path

from gpr_zero.llm.dictionary_generator import DictionaryGenerator
from gpr_zero.project import get_project

# Configuration: providers and run counts
PROVIDERS = {
    "gpt-4.1": {"model": "openai/gpt-4.1", "runs": 1},
    "gpt-4o": {"model": "openai/gpt-4o-2024-11-20", "runs": 2},
    "gemini-2.5-pro": {"model": "google/gemini-2.5-pro", "runs": 2},
    "claude-opus-4": {"model": "anthropic/claude-opus-4", "runs": 2},
}

# Mapping of role filenames to prompting roles
ROLE_MAP = {
    "president": "Taiwanese President",
    "defense_minister": "Taiwanese Minister of Defense",
    "journalist": "Senior Taiwanese Journalist specializing in Cross-Strait Relations",
    "nsb_director": "Director of Taiwan's National Security Bureau",
    "poli_professor": "Taiwanese Politics Professor",
}

# Path to the role-based prompt template
PROMPT_FILE = Path("prompts/generate_taiwan_gpr_dictionary_role_based.json")


def main(project=None):
    project = project or get_project()
    output_dir = project.data_path("dictionaries", "step1_batch")
    prompt_file = project.root / PROMPT_FILE
    pm = project.prompt_manager()
    generator = DictionaryGenerator(prompt_manager=pm, output_dir=str(output_dir))

    for provider_key, cfg in PROVIDERS.items():
        model_name = cfg["model"]
        run_count = cfg["runs"]

        for abbr, full_role in ROLE_MAP.items():
            for i in range(1, run_count + 1):
                output_name = f"{provider_key}_{abbr}_run{i}"
                print(f"Generating dictionary: {output_name} using model {model_name}")

                dictionary = generator.generate_dictionary(
                    strategy="prompt_file",
                    prompt_file_path=prompt_file,
                    template_args_list=[{"role": full_role}],
                    model=model_name,
                    temperature=None,
                    name=output_name,
                )
                output_path = output_dir / f"{output_name}.json"

                # Save metadata summary
                summary_path = output_dir / f"{output_name}_meta.json"
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(
                        dictionary.get("metadata", {}), f, ensure_ascii=False, indent=2
                    )

                print(f"  Saved: {output_path}")
