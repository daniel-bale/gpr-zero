#!/usr/bin/env python3
"""Main entry point for the GPR-Zero project."""

import argparse
import logging
import sys

from gpr_zero.llm.prompt_manager import PromptManager
from gpr_zero.llm.dictionary_generator import DictionaryGenerator
from gpr_zero.llm.news_classifier import NewsClassifier
from gpr_zero.gpr_index.calculator import GPRCalculator
from gpr_zero.gpr_index.analyzer import GPRAnalyzer
from gpr_zero.utils.data_loader import NewsDataLoader


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("gpr_zero.log")],
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GPR-Zero: Improving Geopolitical Risk Assessment"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Dictionary generation command
    dict_parser = subparsers.add_parser(
        "generate-dictionary", help="Generate GPR keyword dictionary"
    )
    dict_parser.add_argument(
        "--strategy",
        choices=["direct", "translation", "guided"],
        default="direct",
        help="Strategy for dictionary generation",
    )
    dict_parser.add_argument("--output", help="Output file name")

    # Index calculation command
    index_parser = subparsers.add_parser("calculate-index", help="Calculate GPR index")
    index_parser.add_argument(
        "--dictionary", required=True, help="Path to dictionary file"
    )
    index_parser.add_argument("--source", required=True, help="News source name")
    index_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    index_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    index_parser.add_argument(
        "--normalize", action="store_true", help="Normalize by total articles"
    )
    index_parser.add_argument("--output", help="Output file name")

    # LLM classification command
    classify_parser = subparsers.add_parser(
        "classify", help="Classify articles using LLM"
    )
    classify_parser.add_argument("--source", required=True, help="News source name")
    classify_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    classify_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    classify_parser.add_argument("--limit", type=int, help="Limit number of articles")

    # Analysis command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze and compare indices"
    )
    analyze_parser.add_argument(
        "--indices", required=True, nargs="+", help="List of index files to compare"
    )
    analyze_parser.add_argument("--names", nargs="+", help="Names for the indices")
    analyze_parser.add_argument(
        "--output-dir", default="data/analysis", help="Directory for analysis outputs"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    setup_logging()
    args = parse_args()

    logger = logging.getLogger("main")
    logger.info(f"Running command: {args.command}")

    if args.command == "generate-dictionary":
        prompt_manager = PromptManager()
        generator = DictionaryGenerator(prompt_manager=prompt_manager)

        dictionary = generator.generate_dictionary(
            strategy=args.strategy, name=args.output
        )

        logger.info(
            f"Generated dictionary with {sum(len(v) for v in dictionary.values())} keywords"
        )

    elif args.command == "calculate-index":
        data_loader = NewsDataLoader()
        calculator = GPRCalculator(dictionary_path=args.dictionary)

        # Load articles
        articles_df = data_loader.load_articles(
            source=args.source, start_date=args.start_date, end_date=args.end_date
        )

        # Group by date
        articles_by_date = data_loader.group_by_date(articles_df)

        # Get total articles if normalizing
        total_articles_by_date = None
        if args.normalize:
            total_articles_by_date = data_loader.get_total_articles_by_date(
                source=args.source, start_date=args.start_date, end_date=args.end_date
            )

        # Calculate index
        index_df = calculator.calculate_index_series(
            articles_by_date=articles_by_date,
            normalize=args.normalize,
            total_articles_by_date=total_articles_by_date,
        )

        # Save index
        output_name = args.output or f"gpr_index_{args.source}"
        calculator.save_index(index_df, output_name)

        logger.info(f"Calculated index for {len(articles_by_date)} dates")

    elif args.command == "classify":
        data_loader = NewsDataLoader()
        prompt_manager = PromptManager()
        classifier = NewsClassifier(prompt_manager=prompt_manager)

        # Load articles
        articles_df = data_loader.load_articles(
            source=args.source,
            start_date=args.start_date,
            end_date=args.end_date,
            limit=args.limit,
        )

        # Prepare batch
        articles_batch = {}
        for _, row in articles_df.iterrows():
            articles_batch[str(row["id"])] = row["text"]

        # Classify
        results = classifier.batch_classify(articles_batch)

        logger.info(f"Classified {len(results)} articles")

    elif args.command == "analyze":
        analyzer = GPRAnalyzer(output_dir=args.output_dir)

        # Load indices
        index_paths = {}
        if args.names and len(args.names) == len(args.indices):
            for name, path in zip(args.names, args.indices):
                index_paths[name] = path
        else:
            for i, path in enumerate(args.indices):
                index_paths[f"index_{i + 1}"] = path

        indices = analyzer.load_indices(index_paths)

        # Compare indices
        comparison_df = analyzer.compare_indices(indices)

        # Plot comparison
        analyzer.plot_indices(comparison_df)

        # Calculate and plot correlation matrix
        corr_matrix = analyzer.correlation_analysis(comparison_df)
        analyzer.plot_correlation_matrix(corr_matrix)

        # Find divergence points
        divergence_df = analyzer.find_divergence_points(comparison_df)

        logger.info(f"Analyzed {len(indices)} indices")
        logger.info(f"Found {len(divergence_df)} significant divergence points")

    else:
        logger.error(f"Unknown command: {args.command}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
