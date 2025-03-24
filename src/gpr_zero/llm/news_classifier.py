"""Zero-shot classifier for geopolitical risk in news articles using LLMs."""

from typing import Dict, List, Optional, Any, Union
import json
import os

from .prompt_manager import PromptManager


class NewsClassifier:
    """Classifies news articles for geopolitical risk using zero-shot LLM classification."""

    def __init__(
        self,
        prompt_manager: Optional[PromptManager] = None,
        output_dir: str = "data/classifications",
    ):
        """Initialize news classifier.

        Args:
            prompt_manager: PromptManager instance for LLM interactions
            output_dir: Directory to save classification results
        """
        self.prompt_manager = prompt_manager or PromptManager()
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def classify_article(
        self, article_text: str, article_id: Optional[str] = None, save: bool = True
    ) -> Dict[str, Any]:
        """Classify a news article for geopolitical risk.

        Args:
            article_text: Text of the article to classify
            article_id: Optional ID for the article
            save: Whether to save the classification results

        Returns:
            Classification results including risk score and categories
        """
        # Truncate article if it's too long
        max_length = 4000  # Approximate max tokens for typical LLM context
        if len(article_text) > max_length:
            article_text = article_text[:max_length] + "..."

        # Build the prompt
        prompt = f"""Please analyze the following news article for geopolitical risk.
        
Article:
{article_text}

Analyze this article and provide:
1. Is this article about geopolitical risk? (yes/no)
2. Risk level (0-10 scale, where 0 is no risk and 10 is extreme risk)
3. Primary risk categories present (military, terrorism, sanctions, etc.)
4. Key sentences that indicate geopolitical risk
5. A brief summary of the geopolitical risk elements

Format your response as a JSON object with the following keys:
- "is_gpr": boolean
- "risk_level": number (0-10)
- "categories": list of strings
- "key_sentences": list of strings
- "summary": string
"""

        # Call LLM
        system_message = "You are an expert in geopolitical risk assessment."
        response = self.prompt_manager.call_llm(prompt, system_message)

        # Parse response (in a real implementation, you would parse the JSON from the response)
        # This is a placeholder
        classification = {
            "is_gpr": True,
            "risk_level": 7,
            "categories": ["military", "territorial_dispute"],
            "key_sentences": [
                "Tensions escalated as troops mobilized near the border."
            ],
            "summary": "The article describes increasing military tensions that pose significant geopolitical risk.",
        }

        # Save if requested
        if save and article_id:
            self._save_classification(classification, article_id)

        return classification

    def batch_classify(
        self, articles: Dict[str, str], save: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Classify multiple articles.

        Args:
            articles: Dictionary mapping article IDs to article texts
            save: Whether to save classification results

        Returns:
            Dictionary mapping article IDs to classification results
        """
        results = {}
        for article_id, article_text in articles.items():
            results[article_id] = self.classify_article(
                article_text, article_id=article_id, save=save
            )
        return results

    def _save_classification(
        self, classification: Dict[str, Any], article_id: str
    ) -> None:
        """Save classification results to a JSON file.

        Args:
            classification: Classification results to save
            article_id: ID of the article
        """
        filename = os.path.join(self.output_dir, f"classification_{article_id}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(classification, f, ensure_ascii=False, indent=2)
