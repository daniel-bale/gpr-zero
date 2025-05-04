# GPR Dictionary Translation Experiments

## Phase 1: Initial Translation (English to Chinese Traditional)

### Experiment 1: Direct Translation of Caldara Dictionary

**Date:** 2025-04-12

**Files:**
- Prompt: `prompts/caldara_translation_combined.json`
- Result: `data/dictionaries/caldara_traditional_chinese.json`
- Model: `gpt-4o-2024-08-06`

**Process:**
We used OpenAI's GPT-4o model to translate the Caldara & Iacoviello (2022) GPR dictionary from English to Traditional Chinese. The translation maintained the original categorical structure and handled word variations marked with asterisks (*) in the original dictionary.

**Issues Identified:**

1. **Operator Translation Issues**:  
   The N/2 operator (ProQuest syntax allowing up to 2 words between terms in any order) works well in English but doesn't translate directly to Chinese due to different segmentation requirements. Chinese text requires word segmentation first, and the concept of "N words between" doesn't map precisely to character-based languages like Chinese.

2. **Vocabulary Gaps**:  
   The term "挑釁" (provocation) is common in describing Taiwan's geopolitical tensions, but its English equivalent "provoke/provocation" is not contained in the original Caldara (2022) dictionary. However, "挑釁" is listed in Lau (2024). This raises an important question about whether such cases represent translation failures or indicate that direct translation alone is insufficient for culturally-specific geopolitical analysis.

3. **Category Definition Translation Issues**:  
   While the word sets have been translated, the category definitions themselves (e.g., `"war_threats": "War_words N/2 Threat_words"`) still reference English word set names and use the N/2 operator. For the dictionary to be functionally useful for calculating a GPR index on Chinese text, these category definitions need to be adapted to reference the translated word set names.

4. **Nuance and Contextual Accuracy of Translations**:  
   Direct translations may not capture the precise nuance intended in the original GPR dictionary. Some terms might be linguistically correct but miss the specific geopolitical risk connotation. Additionally, the chosen Traditional Chinese terms may not be the most common ones used in Taiwanese media and discourse about geopolitical risk.

5. **Effectiveness of Asterisk Handling**:  
   The prompt requested handling English terms with asterisks (like `threat*`) by considering variations. However, Chinese morphology differs significantly from English, and it's unclear whether single Chinese translations adequately capture the range of meanings represented by these wildcard terms in the original dictionary.

**Next Steps:**
1. **Addressing N/2 Operator Issues**:
   - Initially use AND operators to replace N/2 for news search to limit candidates
   - Apply Caldara's approach for constructing country-specific GPR to further narrow down news volume
   - Implement proper word segmentation and N/2-like proximity rules after the initial filtering

2. **Category Definition Adaptation**:
   - Modify the prompt to preserve English category names rather than translating them
   - This will maintain consistency with the original dictionary structure while allowing for translated word sets

3. **Vocabulary Gaps Analysis**:
   - Compare Caldara (2022) and Lau (2024) dictionaries directly using Lau's English translations
   - Assess the severity of vocabulary differences and their potential impact on GPR calculation
   - Determine whether supplementation is necessary based on this comparative analysis

4. **Improving Translation Nuance**:
   - Provide Caldara's original paper as context in the prompt
   - Explore using web search capabilities of LLMs to better understand geopolitical term usage
   - Review translations with domain knowledge of Taiwanese geopolitical discourse

5. **Chinese Morphology Handling**:
   - Current approach seems adequate as most term variations in Chinese are represented by the same word
   - Monitor and evaluate the effectiveness of this approach as we apply the dictionary to actual news analysis