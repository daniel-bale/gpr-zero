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
    """Generate a dictionary using the specified strategy.
    
    Args:
        args: Command line arguments
    """
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Get model configuration
    model_config = config.get("models", {})
    default_model = model_config.get("default", "gpt-4o-2024-08-06")
    available_models = model_config.get("options", [default_model])
    
    # Validate the model if specified by user
    if args.model and args.model not in available_models:
        logging.error(f"Error: Model '{args.model}' not found in available models.")
        logging.info(f"Available models: {', '.join(available_models)}")
        sys.exit(1)
    model_name = args.model or default_model
    
    # Setup generation parameters
    gen_params = config.get("generation", {})
    temperature = gen_params.get("temperature", 0.0)
    
    # Initialize prompt manager
    prompt_manager = PromptManager(
        model_name=model_name,
        temperature=temperature,
        log_dir=args.log_dir,
        prompts_dir=args.prompts_dir
    )
    
    # Initialize dictionary generator
    dictionary_generator = DictionaryGenerator(
        prompt_manager=prompt_manager,
        output_dir=args.output_dir,
        reference_dictionary_path=args.reference_dict
    )
    
    # Generate dictionary using the specified strategy
    if args.strategy == "direct":
        dictionary_generator.generate_direct(
            prompt_file=args.prompt_file,
            output_name=args.output_name
        )
    elif args.strategy == "translation":
        dictionary_generator.generate_translation(
            prompt_file=args.prompt_file,
            reference_path=args.reference_dict,
            output_name=args.output_name
        )
    elif args.strategy == "guided":
        dictionary_generator.generate_guided(
            prompt_file=args.prompt_file,
            output_name=args.output_name
        )
    else:
        logging.error(f"Unknown strategy: {args.strategy}")
        sys.exit(1)
    
    logging.info(f"Dictionary generation completed using {args.strategy} strategy")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GPR-Zero: GPR Dictionary Generation")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--config", default="llm_config.yaml", help="Path to configuration file")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Dictionary generation command
    gen_parser = subparsers.add_parser("generate-dictionary", help="Generate a dictionary")
    gen_parser.add_argument("--strategy", required=True, choices=["direct", "translation", "guided"], 
                          help="Strategy for dictionary generation")
    gen_parser.add_argument("--prompt-file", required=True, help="Prompt file to use (without .json extension)")
    gen_parser.add_argument("--output-name", required=True, help="Output dictionary name (without .json extension)")
    gen_parser.add_argument("--reference-dict", help="Path to reference dictionary for translation strategy")
    gen_parser.add_argument("--model", help="Override the model specified in the config")
    gen_parser.add_argument("--prompts-dir", default="prompts", help="Directory containing prompt templates")
    gen_parser.add_argument("--output-dir", default="data/dictionaries", help="Directory to save generated dictionaries")
    gen_parser.add_argument("--log-dir", default="logs/llm", help="Directory to save LLM logs")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "generate-dictionary":
        generate_dictionary_command(args)
    

if __name__ == "__main__":
    main()
