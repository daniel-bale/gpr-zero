#!/usr/bin/env python3
"""Main entry point for GPR-Zero."""

import argparse
import logging
import sys
import os
from pathlib import Path

import yaml

from src.gpr_zero.llm.prompt_manager import PromptManager
from src.gpr_zero.llm.dictionary_generator import DictionaryGenerator
from gpr_zero.llm.news_classifier import NewsClassifier
from gpr_zero.gpr_index.calculator import GPRCalculator
from gpr_zero.gpr_index.analyzer import GPRAnalyzer
from gpr_zero.utils.data_loader import NewsDataLoader


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
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        sys.exit(1)


def generate_dictionary_command(args) -> None:
    """Generate a dictionary using the specified strategy."""
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
    
    prompt_manager = PromptManager(
        model_name=model_name,
        temperature=temperature,
        log_dir=args.log_dir
    )
    
    dictionary_generator = DictionaryGenerator(
        prompt_manager=prompt_manager,
        output_dir=args.output_dir,
        reference_dictionary_path=args.reference_dict
    )

    # Construct the full path to the prompt file if a name is provided
    prompt_file_full_path = None
    if args.prompt_file:
         prompt_file_full_path = Path(args.prompts_dir) / f"{args.prompt_file}.json"
         if not prompt_file_full_path.exists():
             logging.error(f"Prompt file not found: {prompt_file_full_path}")
             sys.exit(1)

    # Prepare template_args_list if role is provided
    template_args = None
    if args.role:
        # Assuming the role applies to the first message (system)
        # and the second message (user) doesn't need args.
        template_args = [{"role": args.role}, {}] 

    # Call the main generation method
    try:
        generated_dictionary = dictionary_generator.generate_dictionary(
            strategy=args.strategy,
            name=args.output_name, # Pass name for saving
            model=args.model, # Pass model override if provided
            temperature=temperature, # Pass configured/default temperature
            prompt_file_path=prompt_file_full_path, # Pass the resolved path
            template_args_list=template_args, # Pass the role args
            save=True # Ensure saving is handled
        )
        logging.info(f"Dictionary generation completed using {args.strategy} strategy. Saved to {generated_dictionary.get('metadata', {}).get('filename', 'N/A')}")
    except Exception as e:
        logging.error(f"Dictionary generation failed: {e}", exc_info=True)
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GPR-Zero: GPR Dictionary Generation")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--config", default="llm_config.yaml", help="Path to configuration file")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    gen_parser = subparsers.add_parser("generate-dictionary", help="Generate a dictionary")
    gen_parser.add_argument("--strategy", required=True, 
                          choices=["direct", "translation", "guided", "prompt_file", "role_based_generation"], # Added new strategy
                          help="Strategy for dictionary generation")
    gen_parser.add_argument("--prompt-file", help="Prompt file name (without .json) to use for relevant strategies (e.g., translation, prompt_file, role_based_generation)")
    gen_parser.add_argument("--output-name", required=True, help="Output dictionary name (without .json extension)")
    gen_parser.add_argument("--role", help="Role to use for role_based_generation strategy (e.g., 'Taiwanese Minister of Defense')") # New argument
    gen_parser.add_argument("--reference-dict", help="Path to reference dictionary for translation strategy")
    gen_parser.add_argument("--model", help="Override the model specified in the config")
    gen_parser.add_argument("--prompts-dir", default="prompts", help="Directory containing prompt templates")
    gen_parser.add_argument("--output-dir", default="data/dictionaries", help="Directory to save generated dictionaries")
    gen_parser.add_argument("--log-dir", default="logs/llm", help="Directory to save LLM logs")
    
    args = parser.parse_args()
    
    # Validation for required args based on strategy
    if args.command == "generate-dictionary":
        if args.strategy in ["translation", "prompt_file", "role_based_generation"] and not args.prompt_file:
            parser.error("argument --prompt-file is required for strategy: " + args.strategy)
        if args.strategy == "role_based_generation" and not args.role:
            parser.error("argument --role is required for strategy: role_based_generation")
        
        generate_dictionary_command(args)
    elif args.command is None:
        parser.print_help()
        sys.exit(1)
    

if __name__ == "__main__":
    main()
