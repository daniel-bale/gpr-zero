"""Generator for Chinese GPR keyword dictionaries using LLMs."""

from typing import Dict, List, Optional, Any
import json
import os

from .prompt_manager import PromptManager


class DictionaryGenerator:
    """Generates GPR keyword dictionaries using LLM prompting strategies."""

    def __init__(
        self,
        prompt_manager: Optional[PromptManager] = None,
        output_dir: str = "data/dictionaries",
    ):
        """Initialize dictionary generator.

        Args:
            prompt_manager: PromptManager instance for LLM interactions
            output_dir: Directory to save generated dictionaries
        """
        self.prompt_manager = prompt_manager or PromptManager()
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def generate_dictionary(
        self,
        strategy: str,
        categories: Optional[List[str]] = None,
        save: bool = True,
        name: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """Generate a GPR keyword dictionary using the specified strategy.

        Args:
            strategy: Prompting strategy to use ('direct', 'translation', 'guided', etc.)
            categories: Optional list of categories to include in the dictionary
            save: Whether to save the generated dictionary
            name: Name for the saved dictionary file

        Returns:
            Dictionary mapping categories to lists of keywords
        """
        # Different strategies use different prompts
        if strategy == "direct":
            dictionary = self._direct_generation(categories)
        elif strategy == "translation":
            dictionary = self._translation_based_generation(categories)
        elif strategy == "guided":
            dictionary = self._guided_generation(categories)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Save dictionary if requested
        if save:
            self._save_dictionary(dictionary, name or f"dictionary_{strategy}")

        return dictionary

    def _direct_generation(
        self, categories: Optional[List[str]]
    ) -> Dict[str, List[str]]:
        """Generate dictionary by directly asking LLM for Chinese keywords.

        Args:
            categories: Optional list of categories

        Returns:
            Generated dictionary
        """
        # Build prompt based on categories
        if categories:
            categories_str = ", ".join(categories)
            prompt = f"""Please generate a Chinese language dictionary for detecting geopolitical risk in news articles.
The dictionary should include keywords for the following categories: {categories_str}.
For each category, provide a list of at least 10 relevant Chinese keywords that indicate geopolitical risk.
Format your response as a JSON object where keys are categories and values are lists of keywords."""
        else:
            prompt = """Please generate a Chinese language dictionary for detecting geopolitical risk in news articles.
The dictionary should be organized into relevant categories (such as military, terrorism, war, etc.).
For each category, provide a list of at least 10 relevant Chinese keywords that indicate geopolitical risk.
Format your response as a JSON object where keys are categories and values are lists of keywords."""

        # Call LLM
        system_message = (
            "You are an expert in geopolitical risk assessment and Chinese language."
        )
        response = self.prompt_manager.call_llm(prompt, system_message)

        # Parse response (in a real implementation, you would parse the JSON from the response)
        # This is a placeholder
        dictionary = {"军事": ["战争", "冲突"], "恐怖主义": ["袭击", "爆炸"]}

        return dictionary

    def _translation_based_generation(
        self, categories: Optional[List[str]]
    ) -> Dict[str, List[str]]:
        """Generate dictionary by translating English keywords to Chinese.

        Args:
            categories: Optional list of categories

        Returns:
            Generated dictionary
        """
        # Placeholder implementation
        return {"军事": ["战争", "冲突"], "恐怖主义": ["袭击", "爆炸"]}

    def _guided_generation(
        self, categories: Optional[List[str]]
    ) -> Dict[str, List[str]]:
        """Generate dictionary through guided prompting with specific instructions.

        Args:
            categories: Optional list of categories

        Returns:
            Generated dictionary
        """
        # Placeholder implementation
        return {"军事": ["战争", "冲突"], "恐怖主义": ["袭击", "爆炸"]}

    def _save_dictionary(self, dictionary: Dict[str, List[str]], name: str) -> None:
        """Save dictionary to a JSON file.

        Args:
            dictionary: Dictionary to save
            name: Base name for the file
        """
        filename = os.path.join(self.output_dir, f"{name}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, ensure_ascii=False, indent=2)

    def load_dictionary(self, name: str) -> Dict[str, List[str]]:
        """Load a previously saved dictionary.

        Args:
            name: Name of the dictionary file (without .json extension)

        Returns:
            Loaded dictionary
        """
        filename = os.path.join(self.output_dir, f"{name}.json")
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_dictionary(self, dictionary: Dict[str, List[str]]) -> Dict[str, float]:
        """Evaluate the quality of a generated dictionary.

        Args:
            dictionary: Dictionary to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        # Placeholder for implementation
        return {"coverage": 0.85, "precision": 0.92}
