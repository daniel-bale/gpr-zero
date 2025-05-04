"""Manager for LLM prompts, API calls, and response logging."""

from typing import Any
import json
import logging
import os
from datetime import datetime
import string
from pathlib import Path

import openai


class PromptManager:
    """Manages prompts, API calls, and logging for LLM interactions."""

    def __init__(
        self,
        model_name: str = "gpt-4",
        log_dir: str = "logs/llm",
        temperature: float = 0.0,
        prompts_dir: str = "prompts"
    ):
        """Initialize the prompt manager.

        Args:
            model_name: Name of the LLM to use
            log_dir: Directory to store logs
            temperature: Temperature parameter for LLM generation
            prompts_dir: Directory containing prompt templates
        """
        self.model_name = model_name
        self.log_dir = log_dir
        self.temperature = temperature
        self.prompts_dir = prompts_dir

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger("llm_prompt_manager")
        self.logger.setLevel(logging.INFO)
        
        # Initialize OpenAI client if environment variable exists
        if os.environ.get("OPENAI_API_KEY"):
            self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        else:
            self.client = None
            self.logger.warning("OPENAI_API_KEY not found in environment variables")

    def load_prompt_template(self, prompt_name: str) -> dict[str, Any]:
        """Load a prompt template from file.
        
        Args:
            prompt_name: Name of the prompt template file (without .json extension)
            
        Returns:
            Prompt template as a dictionary
        """
        prompt_path = os.path.join(self.prompts_dir, f"{prompt_name}.json")
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Prompt template {prompt_name} not found at {prompt_path}")
            raise FileNotFoundError(f"Prompt template {prompt_name} not found")
    
    def render_prompt(self, template: str, **kwargs) -> str:
        """Render a prompt template with the given arguments.
        
        Args:
            template: Prompt template string
            **kwargs: Arguments to fill in the template
            
        Returns:
            Rendered prompt
        """
        # Use a more flexible formatter that doesn't fail on missing keys
        class SafeFormatter(string.Formatter):
            def get_value(self, key, args, kwargs):
                try:
                    return super().get_value(key, args, kwargs)
                except (KeyError, IndexError):
                    return f"{{{key}}}"
        
        formatter = SafeFormatter()
        return formatter.format(template, **kwargs)

    def call_llm_with_prompt_file(
        self, 
        prompt_name: str, 
        model_override: str | None = None,
        temperature_override: float | None = None,
        **kwargs
    ) -> dict[str, Any]:
        """Call LLM using a prompt template loaded from a file.
        
        Args:
            prompt_name: Name of the prompt template file (without .json extension)
            model_override: Optional override for the model
            temperature_override: Optional override for temperature
            **kwargs: Variables to fill in the prompt template
            
        Returns:
            The LLM response and metadata
        """
        # Load prompt template
        prompt_file = self.load_prompt_template(prompt_name)
        
        # Check if this is a new format prompt with 'prompt' key
        if "prompt" in prompt_file:
            # New format with structured prompt
            # Get parameters from template - model and temperature still at top level or use defaults
            model = model_override or prompt_file.get("model", self.model_name)
            temperature = temperature_override or prompt_file.get("temperature", self.temperature)
            
            # Extract API parameters
            api_params = {
                "model": model,
                "temperature": temperature
            }
            
            # For the new responses.create API
            # - 'input' from the prompt becomes the 'input' parameter
            # - 'text' section becomes the 'text' parameter
            if "input" in prompt_file["prompt"]:
                api_params["input"] = prompt_file["prompt"]["input"]
            
            if "text" in prompt_file["prompt"]:
                api_params["text"] = prompt_file["prompt"]["text"]
            
            # Log the request
            request_id = self._log_request(
                json.dumps(api_params.get("input", []), ensure_ascii=False),
                "Using new prompt format with responses.create API",
                api_params
            )
            
            # Make the API call with structured parameters
            if self.client:
                try:
                    # Use the responses.create endpoint for structured outputs
                    response = self.client.responses.create(**api_params)
                    
                    # Extract response
                    result = {
                        "text": response.output_text,
                        "raw_response": response.model_dump()
                    }
                    
                    # For structured output, parse the JSON
                    try:
                        result["structured_output"] = json.loads(response.output_text)
                    except json.JSONDecodeError:
                        self.logger.warning("Failed to parse structured output as JSON")
                        
                except Exception as e:
                    self.logger.error(f"Error calling LLM API: {e}")
                    result = {
                        "text": f"Error: {str(e)}",
                        "error": str(e)
                    }
            else:
                # Use a mock response for testing when no API key is available
                self.logger.warning("No OpenAI client available, using mock response")
                result = {
                    "text": "This is a mock response as no OpenAI API key was provided.",
                    "mock": True
                }
                
            # Log the response
            self._log_response(request_id, result)
            
            # Add metadata
            result["metadata"] = {
                "prompt_name": prompt_name,
                "model": model,
                "temperature": temperature,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
        else:
            # Legacy format with prompt_template
            # Get parameters from template
            model = model_override or prompt_file.get("model", self.model_name)
            temperature = temperature_override or prompt_file.get("temperature", self.temperature)
            system_message = prompt_file.get("system_message", "")
            
            # Check if prompt_template exists
            if "prompt_template" not in prompt_file:
                raise ValueError(f"Invalid prompt format: missing 'prompt_template' in {prompt_name}")
            
            # Render the prompt template
            prompt_text = self.render_prompt(prompt_file["prompt_template"], **kwargs)
            
            # Call the LLM
            response = self.call_llm(prompt_text, system_message, {"model": model, "temperature": temperature})
            
            # Add metadata
            response["metadata"] = {
                "prompt_name": prompt_name,
                "model": model,
                "temperature": temperature,
                "timestamp": datetime.now().isoformat()
            }
            
            return response

    def call_llm(
        self,
        prompt: str,
        system_message: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call LLM with the given prompt and parameters.

        Args:
            prompt: The prompt to send to the LLM
            system_message: Optional system message to include
            params: Additional parameters for the API call

        Returns:
            The LLM response
        """
        # Default parameters
        default_params = {
            "model": self.model_name,
            "temperature": self.temperature,
        }
        
        # Override with provided params
        call_params = {**default_params, **(params or {})}
        
        # Log the request
        request_id = self._log_request(prompt, system_message, call_params)
        
        # Make the API call if client is available
        if self.client:
            try:
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    messages=messages,
                    **call_params
                )
                
                # Extract response
                result = {
                    "text": response.choices[0].message.content,
                    "raw_response": response.model_dump()
                }
            except Exception as e:
                self.logger.error(f"Error calling LLM API: {e}")
                result = {
                    "text": f"Error: {str(e)}",
                    "error": str(e)
                }
        else:
            # Use a mock response for testing when no API key is available
            self.logger.warning("No OpenAI client available, using mock response")
            result = {
                "text": "This is a mock response as no OpenAI API key was provided.",
                "mock": True
            }

        # Log the response
        self._log_response(request_id, result)

        return result

    def _log_request(
        self,
        prompt: str,
        system_message: str | None,
        params: dict[str, Any] | None,
    ) -> str:
        """Log an LLM request.

        Args:
            prompt: The prompt sent to the LLM
            system_message: The system message, if any
            params: Additional parameters

        Returns:
            A unique ID for the request
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        request_id = f"{timestamp}"

        log_data = {
            "request_id": request_id,
            "timestamp": timestamp,
            "model": params.get("model", self.model_name),
            "prompt": prompt,
            "system_message": system_message,
            "params": params or {},
        }

        log_file = os.path.join(self.log_dir, f"request_{request_id}.json")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Logged request {request_id}")

        return request_id

    def _log_response(self, request_id: str, response: dict[str, Any]) -> None:
        """Log an LLM response.

        Args:
            request_id: The ID of the request
            response: The response from the LLM
        """
        log_data = {
            "request_id": request_id,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            "response": response,
        }

        log_file = os.path.join(self.log_dir, f"response_{request_id}.json")
        
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Logged response for request {request_id}")
