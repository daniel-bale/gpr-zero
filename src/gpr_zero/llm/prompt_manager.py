"""Manager for LLM prompts, API calls, and response logging."""

from typing import Any
import json
import logging
import os
from datetime import datetime
import string
from pathlib import Path
import yaml

import openai


class PromptManager:
    """Manages prompts, API calls, and logging for LLM interactions."""

    def __init__(
        self,
        model_name: str | None = None,
        log_dir: str = "logs/llm",
        temperature: float | None = None,
        config_file_path: str | Path = "llm_config.yaml"
    ):
        """Initialize the prompt manager.

        Args:
            model_name: Optional override for the default model name from config.
            log_dir: Directory to store logs.
            temperature: Optional override for the default temperature from config.
            config_file_path: Path to the LLM configuration YAML file.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration from YAML
        self.config_data = None
        try:
            with open(config_file_path, 'r', encoding='utf-8') as f_config:
                self.config_data = yaml.safe_load(f_config)
        except FileNotFoundError:
            self.logger.warning(f"LLM config file not found at {config_file_path}. Using hardcoded defaults or __init__ args.")
        except Exception as e:
            self.logger.error(f"Error loading LLM config from {config_file_path}: {e}. Using hardcoded defaults or __init__ args.")

        # Set model and temperature from config or fallbacks/overrides
        if self.config_data:
            default_model_from_config = self.config_data.get("models", {}).get("default", "gpt-4")
            default_temp_from_config = self.config_data.get("task_defaults", {}).get("default_generation", {}).get("temperature", 0.0)
        else:
            default_model_from_config = "gpt-4" # Hardcoded fallback
            default_temp_from_config = 0.0    # Hardcoded fallback

        self.model_name = model_name if model_name is not None else default_model_from_config
        self.temperature = temperature if temperature is not None else default_temp_from_config

        # Set up logging (logger needs to be available before config loading potentially fails)
        self.logger = logging.getLogger("llm_prompt_manager")
        self.logger.setLevel(logging.INFO)
        
        # Initialize OpenAI client if environment variable exists
        if os.environ.get("OPENAI_API_KEY"):
            self.client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY")
            )
        else:
            self.client = None
            self.logger.warning("OPENAI_API_KEY not found in environment variables")

    def load_prompt_template(self, template_file: str | Path) -> dict[str, Any]:
        """Load a prompt template from file.
        
        Args:
            template_file: Path to the prompt template JSON file.
            
        Returns:
            Prompt template as a dictionary
        """
        prompt_path = Path(template_file)
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                loaded_template = json.load(f)
                # Basic validation for the new structure
                if "prompt" not in loaded_template or "input" not in loaded_template["prompt"]:
                    raise ValueError("Template must have 'prompt' with an 'input' list.")
                return loaded_template
        except FileNotFoundError:
            self.logger.error(f"Prompt template file not found at {prompt_path}")
            raise FileNotFoundError(f"Prompt template file not found at {prompt_path}")
        except ValueError as ve:
            self.logger.error(f"Invalid prompt template structure in {prompt_path}: {ve}")
            raise
    
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
        prompt_template_file: str | Path,
        template_args_list: list[dict[str, Any]] | None = None, # Made optional
        model_override: str | None = None,
        temperature_override: float | None = None
    ) -> dict[str, Any]:
        """Call LLM using a prompt template loaded from a file.
        
        Args:
            prompt_template_file: Full path to the prompt template JSON file.
            template_args_list: Optional list of dictionaries for template rendering.
                                  Only used if prompt's metadata indicates content is a template.
            model_override: Optional override for the model.
            temperature_override: Optional override for temperature.
            
        Returns:
            The LLM response and metadata
        """
        prompt_file_content = self.load_prompt_template(prompt_template_file)
        template_actual_name = Path(prompt_template_file).name

        # Check metadata flag for rendering content
        render_content_as_template = prompt_file_content.get("metadata", {}).get("render_content_as_template", True)
        
        # Determine model and temperature using hierarchy:
        # 1. Method overrides
        # 2. Template's api_config (for temperature only)
        # 3. PromptManager instance defaults (from YAML or __init__ overrides)
        template_api_config = prompt_file_content.get("api_config", {})
        
        # Model determination: override -> self.model_name (from config/init)
        model = model_override if model_override is not None else self.model_name
        
        # Temperature determination: override -> template_config -> self.temperature (from config/init)
        if temperature_override is not None:
            temperature = temperature_override
        elif "temperature" in template_api_config: # Check if template explicitly sets temperature
            temperature = template_api_config["temperature"]
        else: # Fallback to PromptManager's default temperature
            temperature = self.temperature

        # Prepare messages by rendering content from template's prompt.input
        rendered_messages = []
        if "prompt" in prompt_file_content and "input" in prompt_file_content["prompt"]:
            for i, msg_template in enumerate(prompt_file_content["prompt"]["input"]):
                if "role" in msg_template and "content" in msg_template:
                    rendered_content = msg_template["content"]
                    if render_content_as_template:
                        current_args_for_message = {}
                        if template_args_list and i < len(template_args_list):
                            current_args_for_message = template_args_list[i]
                        rendered_content = self.render_prompt(msg_template["content"], **current_args_for_message)
                    
                    rendered_messages.append({"role": msg_template["role"], "content": rendered_content})
                else:
                    self.logger.warning(f"Skipping invalid message template in {template_actual_name}: {msg_template}")
        else:
            raise ValueError(f"Prompt template {template_actual_name} is missing 'prompt.input' structure.")

        if not rendered_messages:
            raise ValueError(f"No valid messages could be rendered from {template_actual_name}.")

        # API parameters
        api_params = {
            "model": model,
            "temperature": temperature,
            "messages": rendered_messages
        }

        # Check for response formatting (e.g., JSON mode or JSON schema)
        prompt_text_format_config = prompt_file_content.get("prompt", {}).get("text", {}).get("format", {})
        response_format_type = prompt_text_format_config.get("type")

        if response_format_type == "json_schema":
            # Construct the expected structure for json_schema type
            schema_name = prompt_text_format_config.get("name")
            schema_definition = prompt_text_format_config.get("schema")
            schema_strict = prompt_text_format_config.get("strict", True) # Default to True

            if schema_name and schema_definition:
                api_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": schema_strict,
                        "schema": schema_definition
                    }
                }
            else:
                self.logger.error(
                    f"For 'json_schema' type, 'name' and 'schema' must be defined in the prompt template's text.format. Template: {template_actual_name}"
                )
                # Not setting response_format will let the API call proceed without schema enforcement, or error if model requires it.
        elif response_format_type == "json_object":
            # Simple JSON mode
            api_params["response_format"] = {"type": "json_object"}
        
        # Log the request
        # For simplicity in logging, let's take the content of the last message if available.
        representative_rendered_prompt = rendered_messages[-1]["content"] if rendered_messages else ""
        
        # template_args_list for logging might be None if not applicable
        request_id = self._log_request(
            prompt=representative_rendered_prompt,
            system_message=None, 
            params={"model": model, "temperature": temperature},
            template_name=template_actual_name,
            template_args=template_args_list # Pass original, could be None
        )
        
        # Make the API call
        if self.client:
            try:
                response = self.client.chat.completions.create(**api_params)
                
                result_text = response.choices[0].message.content
                result = {
                    "text": result_text,
                    "raw_response": response.model_dump()
                }
                
                # If JSON output was expected via a schema or json_object, try to parse it
                if response_format_type in ["json_schema", "json_object"]:
                    try:
                        result["structured_output"] = json.loads(result_text)
                    except json.JSONDecodeError:
                        self.logger.warning(f"LLM response was expected as JSON ({response_format_type}) but failed to parse. Request ID: {request_id}")
                        # Keep the raw text output
                            
            except Exception as e:
                error_details = str(e)
                if isinstance(e, openai.APIError) and e.body:
                    error_details = e.body
                self.logger.error(f"Error calling LLM API (request {request_id}): {error_details}")
                result = {
                    "text": f"Error: {error_details}",
                    "error": error_details
                }
        else:
            self.logger.warning("No OpenAI client available, using mock response")
            result = {
                "text": "This is a mock response as no OpenAI API key was provided.",
                "mock": True
            }
            
        self._log_response(request_id, result)
        
        result["metadata"] = {
            "prompt_name": template_actual_name,
            "model": model,
            "temperature": temperature,
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
        
        return result

    def call_llm(
        self,
        prompt: str, # User prompt string
        system_message: str | None = None,
        params: dict[str, Any] | None = None, # For model, temperature overrides
        template_name: str | None = None, # Conceptual name if this prompt is a known type
        template_args: dict[str, Any] | None = None # For rendering placeholders in prompt/system_message
    ) -> dict[str, Any]:
        """Call LLM with the given prompt string and parameters.

        Args:
            prompt: The user prompt string to send to the LLM (can contain placeholders).
            system_message: Optional system message (can contain placeholders).
            params: Additional parameters for the API call (e.g., model, temperature).
            template_name: Optional conceptual name for this type of prompt.
            template_args: Optional arguments to fill in placeholders in prompt and system_message.

        Returns:
            The LLM response, similar structure to call_llm_with_prompt_file.
        """
        current_template_args = template_args or {}

        # Determine model and temperature
        call_params_from_arg = params or {}
        model = call_params_from_arg.get("model", self.model_name)
        temperature = call_params_from_arg.get("temperature", self.temperature)

        # Prepare messages
        messages = []
        rendered_system_message_content = None
        if system_message:
            rendered_system_message_content = self.render_prompt(system_message, **current_template_args)
            messages.append({"role": "system", "content": rendered_system_message_content})
        
        rendered_prompt_content = self.render_prompt(prompt, **current_template_args)
        messages.append({"role": "user", "content": rendered_prompt_content})
        
        # API parameters for the call
        api_call_params = {
            "model": model,
            "temperature": temperature,
            "messages": messages
        }
        # Note: response_format for JSON mode is typically not used for ad-hoc string prompts unless specified via params
        # if call_params_from_arg.get("response_format_type") == "json_object": 
        # api_call_params["response_format"] = {"type": "json_object"}

        # Log the request
        request_id = self._log_request(
            prompt=rendered_prompt_content, # Representative prompt for logging
            system_message=rendered_system_message_content, # Log rendered system message if any
            params={"model": model, "temperature": temperature}, # Actual params used
            template_name=template_name, # Conceptual template name
            template_args=current_template_args
        )

        # Make the API call
        if self.client:
            try:
                response = self.client.chat.completions.create(**api_call_params)
                result_text = response.choices[0].message.content
                result = {
                    "text": result_text,
                    "raw_response": response.model_dump()
                }
                # No automatic JSON parsing here unless `params` specifically asked for it and it was handled.
            except Exception as e:
                error_details = str(e)
                if isinstance(e, openai.APIError) and e.body:
                    error_details = e.body
                self.logger.error(f"Error calling LLM API (request {request_id}): {error_details}")
                result = {
                    "text": f"Error: {error_details}",
                    "error": error_details
                }
        else:
            self.logger.warning("No OpenAI client available, using mock response (call_llm)")
            result = {
                "text": "This is a mock response as no OpenAI API key was provided.",
                "mock": True
            }

        self._log_response(request_id, result)

        result["metadata"] = {
            "prompt_name": template_name if template_name else "ad_hoc_prompt",
            "model": model,
            "temperature": temperature,
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
        return result

    def _log_request(
        self,
        prompt: str, 
        system_message: str | None, 
        params: dict[str, Any] | None, 
        template_name: str | None = None,
        template_args: list[dict[str, Any]] | None = None # Changed from dict to list | None
    ) -> str:
        """Log an LLM request.

        Args:
            prompt: The primary rendered prompt content.
            system_message: The rendered system message, if any.
            params: Key parameters like model and temperature used for the call.
            template_name: Optional name of the prompt template used (file or conceptual).
            template_args: Optional list of arguments used for rendering templates messages.

        Returns:
            A unique ID for the request
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        request_id = f"{timestamp}"

        log_data = {
            "request_id": request_id,
            "timestamp": timestamp,
            "model": params.get("model", self.model_name), # Log actual model used
            "temperature": params.get("temperature", self.temperature), # Log actual temp used
            "params_other": {k: v for k, v in (params or {}).items() if k not in ["model", "temperature"]}
        }

        if template_name:
            log_data["template_name"] = template_name
            # Ensure template_args is serializable (it's a list of dicts or None)
            log_data["template_args"] = template_args if template_args is not None else [] # Log as empty list if None
            log_data["rendered_user_prompt"] = prompt
            if system_message:
                 log_data["rendered_system_message"] = system_message
        else: # Ad-hoc prompt from call_llm without a template_name
            log_data["rendered_user_prompt"] = prompt
            if system_message:
                 log_data["rendered_system_message"] = system_message
        
        # To avoid confusion, 'rendered_prompt' from previous version is now more specific.
        # 'prompt' key from previous logging is effectively 'rendered_user_prompt'.

        log_file = self.log_dir / f"request_{request_id}.json"
        
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

        log_file = self.log_dir / f"response_{request_id}.json"
        
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Logged response for request {request_id}")
