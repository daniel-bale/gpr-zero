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
        config_file_path: str | Path = "llm_config.yaml",
        provider: str | None = None,
    ):
        """Initialize the prompt manager.

        Args:
            model_name: Optional override for the default model name from config.
            log_dir: Directory to store logs.
            temperature: Optional override for the default temperature from config.
            config_file_path: Path to the LLM configuration YAML file.
            provider: Optional override for the default provider from config ('openai' or 'openrouter').
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging (logger needs to be available before config loading potentially fails)
        self.logger = logging.getLogger("llm_prompt_manager")
        self.logger.setLevel(logging.INFO)

        # Load configuration from YAML
        self.config_data = None
        try:
            with open(config_file_path, "r", encoding="utf-8") as f_config:
                self.config_data = yaml.safe_load(f_config)
        except FileNotFoundError:
            self.logger.warning(
                f"LLM config file not found at {config_file_path}. Using hardcoded defaults or __init__ args."
            )
        except Exception as e:
            self.logger.error(
                f"Error loading LLM config from {config_file_path}: {e}. Using hardcoded defaults or __init__ args."
            )

        # Set provider from config or fallback/override
        if self.config_data:
            default_provider_from_config = self.config_data.get("providers", {}).get(
                "default", "openai"
            )
            default_model_from_config = self.config_data.get("models", {}).get(
                "default", "gpt-4"
            )
            default_temp_from_config = (
                self.config_data.get("task_defaults", {})
                .get("default_generation", {})
                .get("temperature", 0.0)
            )
        else:
            default_provider_from_config = "openai"  # Hardcoded fallback
            default_model_from_config = "gpt-4"  # Hardcoded fallback
            default_temp_from_config = 0.0  # Hardcoded fallback

        self.provider = (
            provider if provider is not None else default_provider_from_config
        )
        self.model_name = (
            model_name if model_name is not None else default_model_from_config
        )
        self.temperature = (
            temperature if temperature is not None else default_temp_from_config
        )

        # Initialize appropriate client based on provider
        if self.provider == "openrouter":
            self._init_openrouter_client()
        else:  # Default to OpenAI
            self._init_openai_client()

    def _init_openai_client(self):
        """Initialize OpenAI client.

        Raises:
            EnvironmentError: If OPENAI_API_KEY is not set.
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not found in environment variables. "
                "Set it in your .env file or export it."
            )
        self.client = openai.OpenAI(api_key=api_key)
        self.logger.info("Initialized OpenAI client")

    def _init_openrouter_client(self):
        """Initialize OpenRouter client.

        Raises:
            EnvironmentError: If OPENROUTER_API_KEY is not set.
        """
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY not found in environment variables. "
                "Set it in your .env file or export it."
            )
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.logger.info("Initialized OpenRouter client")

    def _make_api_call(self, api_params: dict[str, Any]) -> Any:
        """Make API call with appropriate client based on provider."""
        return self.client.chat.completions.create(**api_params)

    def load_prompt_template(self, template_file: str | Path) -> dict[str, Any]:
        """Load a prompt template from file.

        Args:
            template_file: Path to the prompt template JSON file.

        Returns:
            Prompt template as a dictionary
        """
        prompt_path = Path(template_file)
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                loaded_template = json.load(f)
                # Basic validation for the new structure
                if (
                    "prompt" not in loaded_template
                    or "input" not in loaded_template["prompt"]
                ):
                    raise ValueError(
                        "Template must have 'prompt' with an 'input' list."
                    )
                return loaded_template
        except FileNotFoundError:
            self.logger.error(f"Prompt template file not found at {prompt_path}")
            raise FileNotFoundError(f"Prompt template file not found at {prompt_path}")
        except ValueError as ve:
            self.logger.error(
                f"Invalid prompt template structure in {prompt_path}: {ve}"
            )
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
        template_args_list: list[dict[str, Any]] | None = None,  # Made optional
        model_override: str | None = None,
        temperature_override: float | None = None,
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
        render_content_as_template = prompt_file_content.get("metadata", {}).get(
            "render_content_as_template", True
        )

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
        elif (
            "temperature" in template_api_config
        ):  # Check if template explicitly sets temperature
            temperature = template_api_config["temperature"]
        else:  # Fallback to PromptManager's default temperature
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
                        rendered_content = self.render_prompt(
                            msg_template["content"], **current_args_for_message
                        )

                    rendered_messages.append(
                        {"role": msg_template["role"], "content": rendered_content}
                    )
                else:
                    self.logger.warning(
                        f"Skipping invalid message template in {template_actual_name}: {msg_template}"
                    )
        else:
            raise ValueError(
                f"Prompt template {template_actual_name} is missing 'prompt.input' structure."
            )

        if not rendered_messages:
            raise ValueError(
                f"No valid messages could be rendered from {template_actual_name}."
            )

        # API parameters
        api_params = {
            "model": model,
            "temperature": temperature,
            "messages": rendered_messages,
        }

        # Check for response formatting (e.g., JSON mode or JSON schema)
        prompt_text_format_config = (
            prompt_file_content.get("prompt", {}).get("text", {}).get("format", {})
        )
        response_format_type = prompt_text_format_config.get("type")

        if response_format_type == "json_schema":
            # Construct the expected structure for json_schema type
            schema_name = prompt_text_format_config.get("name")
            schema_definition = prompt_text_format_config.get("schema")
            schema_strict = prompt_text_format_config.get(
                "strict", True
            )  # Default to True

            if schema_name and schema_definition:
                api_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": schema_strict,
                        "schema": schema_definition,
                    },
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
        representative_rendered_prompt = (
            rendered_messages[-1]["content"] if rendered_messages else ""
        )

        # template_args_list for logging might be None if not applicable
        request_id = self._log_request(
            prompt=representative_rendered_prompt,
            system_message=None,
            params={
                "model": model,
                "temperature": temperature,
                "provider": self.provider,
            },
            template_name=template_actual_name,
            template_args=template_args_list,  # Pass original, could be None
        )

        # Make the API call
        try:
            response = self._make_api_call(api_params)

            result_text = response.choices[0].message.content
            result = {"text": result_text, "raw_response": response.model_dump()}

            # If JSON output was expected via a schema or json_object, try to parse it
            if response_format_type in ["json_schema", "json_object"]:
                try:
                    result["structured_output"] = json.loads(result_text)
                except json.JSONDecodeError:
                    self.logger.warning(
                        f"LLM response was expected as JSON ({response_format_type}) but failed to parse. Request ID: {request_id}"
                    )

        except Exception as e:
            error_details = str(e)
            if isinstance(e, openai.APIError) and e.body:
                error_details = e.body
            self.logger.error(
                f"Error calling LLM API (request {request_id}): {error_details}"
            )
            result = {"text": f"Error: {error_details}", "error": error_details}

        self._log_response(request_id, result)

        result["metadata"] = {
            "prompt_name": template_actual_name,
            "model": model,
            "temperature": temperature,
            "provider": self.provider,
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
        }
        return result

    def _log_request(
        self,
        prompt: str,
        system_message: str | None,
        params: dict[str, Any],
        template_name: str | None = None,
        template_args: list[dict[str, Any]] | None = None,
    ) -> str:
        """Log an LLM request.

        Args:
            prompt: The primary rendered prompt content.
            system_message: The rendered system message, if any.
            params: Parameters used for the call (model, temperature, provider).
            template_name: Name of the prompt template file used.
            template_args: Arguments used for rendering template messages.

        Returns:
            A unique ID for the request.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        request_id = timestamp

        log_data: dict[str, Any] = {
            "request_id": request_id,
            "timestamp": timestamp,
            "model": params.get("model", self.model_name),
            "temperature": params.get("temperature", self.temperature),
            "provider": params.get("provider", self.provider),
            "rendered_user_prompt": prompt,
        }

        if template_name:
            log_data["template_name"] = template_name
            log_data["template_args"] = (
                template_args if template_args is not None else []
            )
        if system_message:
            log_data["rendered_system_message"] = system_message

        log_file = self.log_dir / f"request_{request_id}.json"

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Logged request {request_id}")

        return request_id

    def _log_response(self, request_id: str, response: dict[str, Any]) -> None:
        """Log an LLM response.

        Logs text, structured_output, error, and key fields from raw_response
        (id, model, usage) — omitting the full choices payload to keep log
        files small across batch runs.

        Args:
            request_id: The ID of the request
            response: The response from the LLM
        """
        # Build a trimmed copy for logging
        loggable_response: dict[str, Any] = {}
        if "text" in response:
            loggable_response["text"] = response["text"]
        if "structured_output" in response:
            loggable_response["structured_output"] = response["structured_output"]
        if "error" in response:
            loggable_response["error"] = response["error"]

        raw = response.get("raw_response")
        if isinstance(raw, dict):
            summary = {k: raw[k] for k in ("id", "model", "usage") if k in raw}
            # Extract finish_reason from first choice for truncation debugging
            choices = raw.get("choices")
            if isinstance(choices, list) and choices:
                summary["finish_reason"] = choices[0].get("finish_reason")
            loggable_response["raw_response_summary"] = summary

        log_data = {
            "request_id": request_id,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            "response": loggable_response,
        }

        log_file = self.log_dir / f"response_{request_id}.json"

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Logged response for request {request_id}")
