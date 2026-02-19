"""Generator for Chinese GPR keyword dictionaries using LLMs."""

from typing import Dict, List, Optional, Any
import json
import os
from pathlib import Path
from datetime import datetime
import re
import logging

from gpr_zero.llm.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class DictionaryGenerator:
    """Generates GPR keyword dictionaries using LLM prompting strategies."""

    def __init__(
        self,
        prompt_manager: PromptManager,
        output_dir: str = "data/dictionaries",
        reference_dictionary_path: str | None = None
    ):
        """Initialize dictionary generator.

        Args:
            prompt_manager: PromptManager instance for LLM interactions
            output_dir: Directory to save generated dictionaries
            reference_dictionary_path: Path to reference dictionary for translation
        """
        self.prompt_manager = prompt_manager
        self.output_dir = output_dir
        self.reference_dictionary_path = reference_dictionary_path

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def generate_dictionary(
        self,
        strategy: str,
        categories: Optional[List[str]] = None,
        save: bool = True,
        name: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        prompt_file_path: Path | None = None,
        template_args_list: list[dict[str, Any]] | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a GPR keyword dictionary using the specified strategy.

        Args:
            strategy: Prompting strategy to use ('direct', 'translation', 'guided', 'prompt_file')
            categories: Optional list of categories to include in the dictionary
            save: Whether to save the generated dictionary
            name: Name for the saved dictionary file
            model: Model to use (overrides default)
            temperature: Temperature to use (overrides default)
            prompt_file_path: Full path to the prompt template file (used if strategy is 'prompt_file')
            template_args_list: Optional list of dictionaries for template rendering (used if strategy is 'prompt_file').
            **kwargs: Additional arguments for specific strategies

        Returns:
            Dictionary containing word sets, categories, and metadata
        """
        dictionary = {}
        # Different strategies use different methods
        if strategy == "direct":
            dictionary = self._direct_generation(categories, model, temperature)
        elif strategy == "translation":
            if not prompt_file_path:
                raise ValueError("prompt_file_path is required for 'translation' strategy")
            dictionary = self._prompt_file_generation(prompt_file_path, model, temperature, template_args_list, **kwargs)
        elif strategy == "guided":
            dictionary = self._guided_generation(categories, model, temperature)
        elif strategy == "prompt_file":
            if not prompt_file_path:
                raise ValueError("prompt_file_path parameter is required for 'prompt_file' strategy")
            dictionary = self._prompt_file_generation(prompt_file_path, model, temperature, template_args_list, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Add/update metadata if not already present or fully populated
        if "metadata" not in dictionary:
            dictionary["metadata"] = {}
        
        # Ensure minimum metadata exists if not added by generator method
        dictionary["metadata"].setdefault("strategy", strategy)
        dictionary["metadata"].setdefault("generated_at", datetime.now().isoformat())
        if model: dictionary["metadata"].setdefault("model_used", model)
        if temperature: dictionary["metadata"].setdefault("temperature_used", temperature)

        # Save dictionary if requested
        if save:
            filename = name or f"dictionary_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._save_dictionary(dictionary, filename)
            # Add filename to metadata
            dictionary["metadata"]["filename"] = f"{filename}.json"

        return dictionary

    def _direct_generation(
        self, 
        categories: Optional[List[str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate dictionary by directly asking LLM for Chinese keywords.

        Args:
            categories: Optional list of categories
            model: Optional model override
            temperature: Optional temperature override

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
        response = self.prompt_manager.call_llm(
            prompt, 
            system_message,
            params={"model": model, "temperature": temperature} if model else None
        )

        # Try to parse JSON from the response
        dictionary = self._extract_json_from_response(response["text"])
        
        # Add raw response to metadata
        dictionary["metadata"] = {
            "strategy": "direct",
            "generated_at": datetime.now().isoformat(),
            "model": model or self.prompt_manager.model_name,
            "temperature": temperature or self.prompt_manager.temperature,
            "raw_response": response
        }

        return dictionary

    def _translation_generation(
        self, 
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate dictionary by using the translation prompt on Caldara's dictionary.

        Args:
            model: Optional model override
            temperature: Optional temperature override
            **kwargs: Additional arguments

        Returns:
            Generated dictionary with translations
        """
        # Use the prompt_file strategy with the caldara_translation prompt
        return self._prompt_file_generation(
            "caldara_translation", 
            model, 
            temperature, 
            **kwargs
        )

    def _guided_generation(
        self, 
        categories: Optional[List[str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate dictionary through guided prompting with specific instructions.

        Args:
            categories: Optional list of categories
            model: Optional model override
            temperature: Optional temperature override

        Returns:
            Generated dictionary
        """
        # Placeholder implementation - to be replaced with actual implementation
        mock_result = {
            "word_sets": {
                "军事": ["战争", "冲突"], 
                "恐怖主义": ["袭击", "爆炸"]
            },
            "metadata": {
                "strategy": "guided",
                "generated_at": datetime.now().isoformat(),
                "model": model or self.prompt_manager.model_name,
                "temperature": temperature or self.prompt_manager.temperature,
                "note": "This is a placeholder implementation"
            }
        }
        return mock_result

    def _prompt_file_generation(
        self,
        prompt_file_path: Path,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        template_args_list: list[dict[str, Any]] | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate dictionary using a prompt file.
        
        Args:
            prompt_file_path: Full path to the prompt template file.
            model: Optional model override.
            temperature: Optional temperature override.
            template_args_list: Optional list of dictionaries for template rendering (e.g., containing role).
            **kwargs: Catch-all for potential future args.
            
        Returns:
            Generated dictionary including metadata, categories, and word_sets.
        """
        # Warn about unexpected kwargs if any
        if kwargs:
             logger.warning(f"Unexpected kwargs in _prompt_file_generation: {kwargs}")

        # Load prompt file content to access its metadata
        prompt_metadata = {}
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                prompt_metadata = json.load(f).get("metadata", {})
        except Exception as e:
            logger.warning(f"Could not load prompt metadata from {prompt_file_path}: {e}")

        # Call LLM using the prompt file path
        response = self.prompt_manager.call_llm_with_prompt_file(
            prompt_template_file=prompt_file_path,
            template_args_list=template_args_list,
            model_override=model,
            temperature_override=temperature
        )
        
        # Try to parse JSON containing categories and word_sets from the response
        llm_output_dict = self._extract_json_from_response(response["text"])

        # --- Construct final dictionary --- 
        final_dict = {}

        # 1. Build Metadata
        metadata_dict = {}
        generation_date = datetime.now().strftime("%Y-%m-%d")
        model_used = response.get("metadata", {}).get("model", model or self.prompt_manager.model_name)
        role_used = "N/A" # Default role
        if template_args_list and len(template_args_list) > 0 and isinstance(template_args_list[0], dict):
            role_used = template_args_list[0].get("role", "N/A") # Assuming role is in the first arg dict

        metadata_dict["description"] = prompt_metadata.get("description", "GPR dictionary generated from prompt.")
        metadata_dict["target_context"] = prompt_metadata.get("target_context", "Unknown")
        metadata_dict["generated_by_role"] = role_used
        metadata_dict["generation_date"] = generation_date
        metadata_dict["model_used"] = model_used
        metadata_dict["source_prompt_file"] = str(prompt_file_path)
        metadata_dict["llm_request_id"] = response.get("metadata", {}).get("request_id", "N/A")

        final_dict["metadata"] = metadata_dict

        # 2. Add Categories and Word Sets (if valid)
        if "error" in llm_output_dict or not isinstance(llm_output_dict, dict):
            logger.error(f"LLM response parsing failed or returned error for {prompt_file_path}. Raw text: {response.get('text', '')}")
            final_dict["error"] = llm_output_dict.get("error", "LLM response parsing failed.")
            final_dict["categories"] = {}
            final_dict["word_sets"] = {}
            final_dict["raw_text_response"] = response.get('text', '') # Add raw text on error
        elif "categories" not in llm_output_dict or "word_sets" not in llm_output_dict:
            logger.error(f"LLM response missing 'categories' or 'word_sets' for {prompt_file_path}. Raw text: {response.get('text', '')}")
            final_dict["error"] = "LLM response missing required keys."
            final_dict["categories"] = llm_output_dict.get("categories", {})
            final_dict["word_sets"] = llm_output_dict.get("word_sets", {})
            final_dict["raw_text_response"] = response.get('text', '') # Add raw text on error
        else:
            # Looks good, add categories and word sets
            final_dict["categories"] = llm_output_dict["categories"]
            final_dict["word_sets"] = llm_output_dict["word_sets"]
        
        return final_dict

    def _extract_json_from_response(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response text.
        
        Args:
            text: Response text that may contain JSON
            
        Returns:
            Extracted JSON as dictionary
        """
        # Try to extract JSON from the response
        try:
            # First try to parse the entire response as JSON
            return json.loads(text)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON using regex
            json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
            json_match = re.search(json_pattern, text)
            
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # If regex extraction fails, look for { and } to find JSON object
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                try:
                    return json.loads(text[start_idx:end_idx+1])
                except json.JSONDecodeError:
                    pass
            
            # If all extraction methods fail, return a default structure
            return {
                "error": "Failed to parse JSON from response",
                "raw_text": text,
                "word_sets": {}
            }

    def _save_dictionary(self, dictionary: Dict[str, Any], name: str) -> str:
        """Save dictionary to a JSON file.

        Args:
            dictionary: Dictionary to save
            name: Base name for the file
            
        Returns:
            Path to the saved file
        """
        filename = os.path.join(self.output_dir, f"{name}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, ensure_ascii=False, indent=2)
            
        return filename

    def load_dictionary(self, name: str) -> Dict[str, Any]:
        """Load a previously saved dictionary.

        Args:
            name: Name of the dictionary file (without .json extension)

        Returns:
            Loaded dictionary
        """
        filename = os.path.join(self.output_dir, f"{name}.json")
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_dictionary(self, dictionary: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the quality of a generated dictionary.

        Args:
            dictionary: Dictionary to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        # Placeholder for implementation
        return {"coverage": 0.85, "precision": 0.92}

    def load_reference_dictionary(self, path: str | None = None) -> dict[str, Any]:
        """Load a reference dictionary from file.
        
        Args:
            path: Path to the reference dictionary
            
        Returns:
            Reference dictionary
        """
        path = path or self.reference_dictionary_path
        if not path:
            raise ValueError("No reference dictionary path provided")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise Exception(f"Error loading reference dictionary: {e}")

    def save_dictionary(self, dictionary: dict[str, Any], name: str) -> str:
        """Save a generated dictionary to file.
        
        Args:
            dictionary: The dictionary to save
            name: Base name for the dictionary file
            
        Returns:
            Path where the dictionary was saved
        """
        output_path = os.path.join(self.output_dir, f"{name}.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, indent=2, ensure_ascii=False)
            
        return output_path

    def generate_direct(
        self,
        prompt_file_path: Path,
        output_name: str,
        **kwargs
    ) -> dict[str, Any]:
        """Generate a dictionary using a prompt file (direct implies using a specific prompt file here).
        
        Args:
            prompt_file_path: Full path to the prompt file to use.
            output_name: Base name for the output dictionary
        """
        template_args_list = kwargs.pop("template_args_list", [{}, {}])
        model_override = kwargs.pop("model_override", None)
        temperature_override = kwargs.pop("temperature_override", None)

        if kwargs:
            print(f"Warning: Unexpected keyword arguments in generate_direct: {kwargs}")

        response = self.prompt_manager.call_llm_with_prompt_file(
            prompt_template_file=prompt_file_path,
            template_args_list=template_args_list,
            model_override=model_override,
            temperature_override=temperature_override
        )
        
        # Extract and parse dictionary from response
        try:
            # Try to extract JSON from the response
            response_text = response["text"]
            
            # Check if we have a JSON object anywhere in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end+1]
                dictionary = json.loads(json_str)
            else:
                raise ValueError("No JSON object found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            raise Exception(f"Error parsing dictionary from response: {e}")
        
        # Save dictionary
        self.save_dictionary(dictionary, output_name)
        
        return dictionary

    def generate_translation(
        self,
        prompt_file_path: Path,
        reference_path: str | None = None,
        output_name: str | None = None,
        **kwargs
    ) -> dict[str, Any]:
        """Generate a dictionary by translating, using a specific prompt file.
        
        Args:
            prompt_file_path: Full path to the prompt file to use for translation.
        """
        if not output_name:
            output_name = "translated_dictionary"
        
        if reference_path:
            reference_dict = self.load_reference_dictionary(reference_path)
            kwargs["reference_dictionary"] = json.dumps(reference_dict, ensure_ascii=False)
        
        return self.generate_direct(prompt_file_path, output_name, **kwargs)

    def generate_guided(
        self,
        prompt_file_path: Path,
        output_name: str,
        **kwargs
    ) -> dict[str, Any]:
        """Generate a dictionary using guided approach with multiple LLM calls.
        Args:
            prompt_file_path: Full path to the base prompt file to use.
        """
        return self.generate_direct(prompt_file_path, output_name, **kwargs)
