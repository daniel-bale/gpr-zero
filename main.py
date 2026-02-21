#!/usr/bin/env python3
"""Main entry point for GPR-Zero."""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from gpr_zero.llm.prompt_manager import PromptManager


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str = "llm_config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        sys.exit(1)


def generate_dictionary_command(args) -> None:
    """Generate a dictionary using the specified prompt file."""
    config = load_config(args.config)
    setup_logging(args.log_level)
    model_config = config.get("models", {})
    default_model = model_config.get("default", "gpt-4o-2024-08-06")
    available_models = model_config.get("options", [default_model])

    if args.model and args.model not in available_models:
        logging.error(f"Error: Model '{args.model}' not found in available models.")
        logging.info(f"Available models: {', '.join(available_models)}")
        sys.exit(1)
    model_name = args.model or default_model

    gen_params = config.get("generation", {})
    temperature = gen_params.get("temperature", 0.0)

    pm = PromptManager(
        model_name=model_name, temperature=temperature, log_dir=args.log_dir
    )

    # Resolve prompt file path
    prompt_file_path = Path(args.prompts_dir) / f"{args.prompt_file}.json"
    if not prompt_file_path.exists():
        logging.error(f"Prompt file not found: {prompt_file_path}")
        sys.exit(1)

    # Prepare template_args_list if role is provided
    template_args = None
    if args.role:
        template_args = [{"role": args.role}, {}]

    # Call LLM and save result
    try:
        response = pm.call_llm_with_prompt_file(
            prompt_template_file=prompt_file_path,
            template_args_list=template_args,
            model_override=args.model,
            temperature_override=temperature,
        )

        dictionary = response.get("structured_output") or json.loads(response["text"])

        # Save dictionary
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.output_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, ensure_ascii=False, indent=2)

        logging.info(f"Dictionary saved to {output_path}")
    except Exception as e:
        logging.error(f"Dictionary generation failed: {e}", exc_info=True)
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GPR-Zero: GPR Dictionary Generation")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument(
        "--config", default="llm_config.yaml", help="Path to configuration file"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    gen_parser = subparsers.add_parser(
        "generate-dictionary", help="Generate a dictionary"
    )
    gen_parser.add_argument(
        "--prompt-file",
        required=True,
        help="Prompt file name (without .json extension)",
    )
    gen_parser.add_argument(
        "--output-name",
        required=True,
        help="Output dictionary name (without .json extension)",
    )
    gen_parser.add_argument(
        "--role",
        help="Role for template rendering (e.g., 'Taiwanese Minister of Defense')",
    )
    gen_parser.add_argument(
        "--model", help="Override the model specified in the config"
    )
    gen_parser.add_argument(
        "--prompts-dir", default="prompts", help="Directory containing prompt templates"
    )
    gen_parser.add_argument(
        "--output-dir",
        default="data/dictionaries",
        help="Directory to save generated dictionaries",
    )
    gen_parser.add_argument(
        "--log-dir", default="logs/llm", help="Directory to save LLM logs"
    )

    args = parser.parse_args()

    if args.command == "generate-dictionary":
        generate_dictionary_command(args)
    elif args.command is None:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
