#!/usr/bin/env python3
"""
Generate LLM summaries for sampled spike articles.
Uses the same prompt template and approach as capture_gpr_spikes.py
Makes one LLM call per spike event.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import logging
import time
from datetime import datetime
from tqdm import tqdm
from src.gpr_zero.llm.prompt_manager import PromptManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SPIKE_ARTICLES_DIR = Path("data/gpr_spikes/new_spikes_articles_step2")
PREVIOUS_SPIKES_DIR = Path("data/gpr_spikes/spike_articles_all_step2")
OUTPUT_DIR = Path("data/gpr_spikes/spike_summaries_llm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_TEMPLATE_FILE = Path("prompts/analyze_gpr_spike_event.json")
MAX_CONTENT_PREVIEW_LEN = 600  # Same as capture_gpr_spikes.py

def load_spike_articles(spike_file: Path) -> dict:
    """Load spike articles from JSON file."""
    with open(spike_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_articles_summary(spike_data: dict) -> str:
    """Prepare articles summary for LLM prompt, matching capture_gpr_spikes.py format."""
    articles_summary_parts = []
    
    for article in spike_data['sampled_articles'][:200]:  # Limit to 200 articles like original
        content_preview = article.get('content', '')
        if len(content_preview) > MAX_CONTENT_PREVIEW_LEN:
            content_preview = content_preview[:MAX_CONTENT_PREVIEW_LEN] + "..."
        
        pub_date = article.get('publish_date', 'Unknown')
        outlet = article.get('outlet', 'Unknown')
        
        # Match the exact format from capture_gpr_spikes.py
        articles_summary_parts.append(
            f"Outlet: {outlet}\\nDate: {pub_date}\\nContent Snippet: {content_preview}\\n---\\n"
        )
    
    return "\\n".join(articles_summary_parts)

def process_llm_response(llm_response_data: dict, spike_id: str) -> dict:
    """Process LLM response and extract structured information."""
    result = {
        'llm_short_answer': "N/A",
        'llm_long_answer': "N/A",
        'llm_keywords': "N/A"
    }
    
    if llm_response_data.get("error"):
        error_msg = str(llm_response_data['error'])
        logger.error(f"LLM API error for {spike_id}: {error_msg}")
        result['llm_short_answer'] = f"LLM: API Error"
        result['llm_long_answer'] = error_msg
        return result
    
    if llm_response_data.get("mock"):
        logger.info(f"LLM call mocked for {spike_id}")
        result['llm_short_answer'] = "LLM: Mock Response"
        result['llm_long_answer'] = "LLM: Mock Response (No API Key)"
        return result
    
    if not llm_response_data.get("text"):
        logger.error(f"LLM response missing 'text' field for {spike_id}")
        return result
    
    raw_text = llm_response_data["text"]
    logger.info(f"LLM raw response for {spike_id} (first 200 chars): {raw_text[:200]}...")
    
    try:
        parsed_output = json.loads(raw_text)
        logger.info(f"Successfully parsed JSON for {spike_id}")
        
        short_answer = parsed_output.get("short_answer", "")
        long_answer = parsed_output.get("long_answer", "")
        keywords = parsed_output.get("keywords", "")
        
        if short_answer and short_answer not in ["", "N/A", None]:
            result['llm_short_answer'] = short_answer
        
        if long_answer and long_answer not in ["", "N/A", None]:
            result['llm_long_answer'] = long_answer
            
        if keywords and keywords not in ["", "N/A", None]:
            result['llm_keywords'] = keywords
        
        logger.info(f"Processed result for {spike_id}: short='{result['llm_short_answer']}', keywords='{result['llm_keywords']}'")
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON response for {spike_id}: {e}")
        result['llm_short_answer'] = "LLM: JSON parsing error"
        result['llm_long_answer'] = raw_text
    except Exception as e:
        logger.error(f"Unexpected error processing LLM response for {spike_id}: {e}")
        result['llm_short_answer'] = "LLM: Processing error"
        result['llm_long_answer'] = str(e)
    
    return result

def main():
    """Process all spike articles and generate LLM summaries."""
    logger.info("Starting LLM summary generation for spike articles...")
    
    # Initialize PromptManager with Gemini Flash
    prompt_manager = PromptManager(
        log_dir=OUTPUT_DIR / "llm_logs",
        model_name="google/gemini-2.5-flash-lite-preview-06-17"
    )
    
    if not PROMPT_TEMPLATE_FILE.exists():
        logger.error(f"Prompt template not found: {PROMPT_TEMPLATE_FILE}")
        return
    
    # Collect all spike files
    spike_files = []
    
    # New spikes
    new_spike_files = []
    if SPIKE_ARTICLES_DIR.exists():
        new_spike_files = list(SPIKE_ARTICLES_DIR.glob("spike_*_sampled_articles.json"))
        spike_files.extend(new_spike_files)
        logger.info(f"Found {len(new_spike_files)} new spike files")
    
    # Previous spikes (add all previously sampled spikes)
    old_spike_files = []
    if PREVIOUS_SPIKES_DIR.exists():
        # Match both spike_XX_sampled_articles.json and spike_YYYYMMDD_YYYYMMDD_sampled_articles.json
        old_spike_files = [f for f in PREVIOUS_SPIKES_DIR.glob("spike_*_sampled_articles.json") 
                          if f.name != "all_step2_spikes_sampled_20250802_051934.json"]
        spike_files.extend(old_spike_files)
        logger.info(f"Found {len(old_spike_files)} previous spike files")
    
    # Filter out 2025 spikes
    spike_files = [f for f in spike_files if "2025" not in f.name]
    logger.info(f"After filtering out 2025 spikes: {len(spike_files)} files")
    
    if not spike_files:
        logger.warning("No spike files found to process")
        return
    
    logger.info(f"Total spike files to process: {len(spike_files)}")
    
    # Process each spike
    all_summaries = []
    
    for spike_file in tqdm(spike_files, desc="Processing spikes"):
        spike_id = spike_file.stem.replace("_sampled_articles", "")
        logger.info(f"\nProcessing {spike_id}...")
        
        try:
            # Load spike data
            spike_data = load_spike_articles(spike_file)
            
            # Skip if no articles
            if not spike_data.get('sampled_articles'):
                logger.warning(f"No articles found for {spike_id}, skipping...")
                continue
            
            # Prepare articles summary
            articles_summary = prepare_articles_summary(spike_data)
            
            # Prepare template arguments
            template_args = [
                {},  # Empty dict for first template slot
                {
                    "gpr_category_analysed": "overall",  # Using "overall" as default
                    "start_date": spike_data['start_date'],
                    "end_date": spike_data['end_date'],
                    "peak_date": spike_data['peak_date'],
                    "peak_gpr_share": spike_data['peak_value'],
                    "news_articles_summary": articles_summary
                }
            ]
            
            # Make LLM call
            logger.info(f"Calling LLM for {spike_id}...")
            time.sleep(2)  # Rate limiting
            
            llm_response = prompt_manager.call_llm_with_prompt_file(
                prompt_template_file=PROMPT_TEMPLATE_FILE,
                template_args_list=template_args
            )
            
            # Process response
            llm_result = process_llm_response(llm_response, spike_id)
            
            # Create summary record
            summary = {
                'spike_id': spike_id,
                'start_date': spike_data['start_date'],
                'end_date': spike_data['end_date'],
                'peak_date': spike_data['peak_date'],
                'peak_value': spike_data['peak_value'],
                'total_articles_sampled': spike_data['total_articles_sampled'],
                'llm_short_answer': llm_result['llm_short_answer'],
                'llm_long_answer': llm_result['llm_long_answer'],
                'llm_keywords': llm_result['llm_keywords']
            }
            
            all_summaries.append(summary)
            
            # Save individual summary
            individual_output = OUTPUT_DIR / f"{spike_id}_llm_summary.json"
            with open(individual_output, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved summary for {spike_id}: {llm_result['llm_short_answer']}")
            
        except Exception as e:
            logger.error(f"Error processing {spike_id}: {e}")
            continue
    
    # Save all summaries
    if all_summaries:
        all_summaries_file = OUTPUT_DIR / f"all_spike_summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(all_summaries_file, 'w', encoding='utf-8') as f:
            json.dump(all_summaries, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\nProcessed {len(all_summaries)} spikes successfully")
        logger.info(f"All summaries saved to: {all_summaries_file}")
        
        # Print summary table
        logger.info("\n" + "="*80)
        logger.info("SPIKE SUMMARIES:")
        logger.info("="*80)
        for summary in all_summaries:
            logger.info(f"\n{summary['spike_id']} ({summary['peak_date']}):")
            logger.info(f"  Short Answer: {summary['llm_short_answer']}")
            logger.info(f"  Keywords: {summary['llm_keywords']}")
    else:
        logger.warning("No summaries generated")

if __name__ == "__main__":
    main()