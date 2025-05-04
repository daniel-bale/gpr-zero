# GPR Dictionary Translation Process - Phase 1

This documentation explains the step-by-step process of how the GPR-Zero system sends prompts to language models and saves the results, focusing on the dictionary translation workflow.

## Overview of Components

The system consists of three main components that work together:

1. **main.py** - The entry point that handles command-line arguments and initiates the process
2. **prompt_manager.py** - Manages prompt templates, API calls, and logging
3. **dictionary_generator.py** - Contains logic for generating dictionaries through various strategies

## Step-by-Step Workflow

### 1. Command Execution

The process begins when the user runs a command like:

```bash
uv run main.py generate-dictionary --strategy translation --prompt-file caldara_translation_combined --output-name caldara_traditional_chinese --model gpt-4o-2024-08-06
```

### 2. Command Line Argument Processing

In `main.py`, the command line arguments are processed:
- `--strategy translation` specifies the translation strategy
- `--prompt-file caldara_translation_combined` indicates which prompt file to use
- `--output-name caldara_traditional_chinese` defines the output file name
- `--model gpt-4o-2024-08-06` overrides the default model

### 3. Initialization of Components

1. The `PromptManager` is initialized with:
   - Model name (default or overridden)
   - Temperature setting
   - Log directory
   - Prompts directory

2. The `DictionaryGenerator` is initialized with:
   - The `PromptManager` instance
   - Output directory for saving results
   - Reference dictionary path (if applicable)

### 4. Prompt Processing

When `generate_translation()` is called in `dictionary_generator.py`:

1. The system calls `generate_direct()` method which:
   - Calls `call_llm_with_prompt_file()` in the `PromptManager`

2. The `PromptManager`:
   - Loads the prompt template from `prompts/caldara_translation_combined.json`
   - Detects if it's using the new format with a `prompt` key (structured format)
   - Extracts parameters like model and temperature
   - Constructs API parameters for the OpenAI call

### 5. OpenAI API Call

1. The API call is prepared with:
   - The prompt structure from the template
   - Response format specification (JSON schema)
   - Model and temperature settings

2. The API call is logged and executed through the OpenAI client

3. The response is processed and includes:
   - The generated text
   - Raw API response
   - Metadata (prompt name, model, temperature, timestamp)
   - Request ID for traceability

### 6. Response Processing and Dictionary Generation

1. In `generate_direct()`, the response is processed to extract the JSON:
   - The system looks for a JSON object in the response
   - The JSON is parsed into a Python dictionary

2. Metadata is added to the dictionary:
   - Strategy used
   - Generation timestamp
   - Model and temperature
   - Raw API response
   - Request ID (links to log files)

### 7. Saving Results

The final dictionary is saved to a file:
1. The output path is constructed using the provided name (`caldara_traditional_chinese.json`)
2. The dictionary is saved in JSON format with UTF-8 encoding to preserve Chinese characters
3. The full path where the dictionary was saved is returned

## File Structure and Data Flow

```
Command Line Arguments
       │
       ▼
     main.py
       │
       ├────────────────────┐
       │                    │
       ▼                    ▼
prompt_manager.py    dictionary_generator.py
       │                    │
       │                    │
       └────────┬───────────┘
                │
                ▼
          OpenAI API Call
                │
                ▼
        Response Processing
                │
                ▼
        Save Dictionary to File
```

## Prompt Structure

The `caldara_translation_combined.json` prompt file uses a structured format with:

1. **Metadata** section with information about the prompt
2. **Prompt** section containing:
   - `input`: Array of messages with system and user roles
   - `text`: Response format specification with JSON schema

This structure ensures that the LLM returns properly formatted JSON that matches the required schema for the dictionary.

## Logging and Traceability

Throughout the process, comprehensive logging happens with full traceability between logs and output files:

1. **Request Logging**:
   - Each request gets a unique `request_id` (timestamp-based)
   - Request details saved to `logs/llm/request_[request_id].json`
   - Includes prompt, model, temperature, and all API parameters

2. **Response Logging**:
   - Response saved to `logs/llm/response_[request_id].json` using the same request_id
   - Contains the raw API response and processed text

3. **Dictionary Output**:
   - Final dictionary saved to `data/dictionaries/[output_name].json`
   - Contains the same `request_id` in its metadata
   - Links directly to the corresponding log files

### Connecting Logs to Output Files

You can trace any generated dictionary back to its original API call and prompt by:

1. Opening the output file (e.g., `data/dictionaries/caldara_traditional_chinese.json`)
2. Finding the `request_id` in the metadata section
3. Locating the corresponding log files in `logs/llm/`:
   - `request_[request_id].json` - Contains the original prompt and parameters
   - `response_[request_id].json` - Contains the raw API response

This provides full reproducibility and traceability for debugging and verification purposes.

## Next Steps

After the dictionary is generated, it can be further processed or used in GPR index generation, but that would be covered in subsequent phases of the project. 