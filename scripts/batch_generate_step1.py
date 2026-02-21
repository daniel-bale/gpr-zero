#!/usr/bin/env python3
"""
Batch-generate GPR dictionaries (step 1):
Run the current role-based prompt across multiple LLM providers & runs.
"""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from gpr_zero.llm.prompt_manager import PromptManager  # noqa: E402

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

# Output directory for generated dictionaries
OUTPUT_DIR = Path("data/dictionaries/step1_batch")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    pm = PromptManager(provider="openrouter")

    for provider_key, cfg in PROVIDERS.items():
        model_name = cfg["model"]
        run_count = cfg["runs"]

        for abbr, full_role in ROLE_MAP.items():
            for i in range(1, run_count + 1):
                output_name = f"{provider_key}_{abbr}_run{i}"
                print(f"Generating dictionary: {output_name} using model {model_name}")

                response = pm.call_llm_with_prompt_file(
                    prompt_template_file=PROMPT_FILE,
                    template_args_list=[{"role": full_role}],
                    model_override=model_name,
                )

                dictionary = response.get("structured_output") or json.loads(
                    response["text"]
                )

                # Save dictionary
                output_path = OUTPUT_DIR / f"{output_name}.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(dictionary, f, ensure_ascii=False, indent=2)

                # Save metadata summary
                summary_path = OUTPUT_DIR / f"{output_name}_meta.json"
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(
                        response.get("metadata", {}), f, ensure_ascii=False, indent=2
                    )

                print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
