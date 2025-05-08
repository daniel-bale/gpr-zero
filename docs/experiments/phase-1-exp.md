# GPR Dictionary Translation Experiments

**Summary of Experiments:**
- **Experiment 1:** Initial direct translation of Caldara dictionary to Traditional Chinese.
- **Experiment 2 (New):** Modified translation prompt to use AND operators in category definitions within the example dictionary and retain English word set names, aiming to improve category definition handling.

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

### Experiment 2: Adapting Category Definitions in Prompt

**Date:** 2025-05-09

**Files:**
- Prompt: `prompts/caldara_translation_chinese_and_operator.json`
- Result: `data/dictionaries/caldara_traditional_chinese_and_ops.json` 
- Model: `gpt-4.1-2025-04-14`

**Process:**
This experiment addresses Issue #3 ("Category Definition Translation Issues") and offers an alternative to Issue #1 ("Operator Translation Issues") from Experiment 1. A new prompt (`prompts/caldara_translation_chinese_and_operator.json`) was created.

The key modifications in this new prompt compared to `prompts/caldara_translation_combined.json` are within the example dictionary provided to the LLM:
1.  All ProQuest-style `N/2` operators in category definitions (e.g., `"war_threats": "War_words N/2 Threat_words"`) have been replaced with `AND` operators (e.g., `"war_threats": "War_words AND Threat_words"`).
2.  The names of the word sets referenced within these category definitions (e.g., `War_words`, `Threat_words`) are kept in English.

**Goal:**
The objective is to guide the LLM to produce a Chinese dictionary where the translated word sets are correctly structured, while the category *definitions* (which would be part of the output metadata if the LLM follows the schema) use the English word set names and a simplified `AND` logic. This simplifies the subsequent parsing and usage of the translated dictionary, as the category logic remains tied to the original English set names, even if the words within those sets are translated to Chinese.

**Expected Outcome:**
A translated dictionary where word sets are in Traditional Chinese, and the `categories` field in the output JSON reflects the `AND` based structure with English word set keys, as demonstrated in the modified input prompt. This should make the resulting dictionary more directly usable for GPR calculations on Chinese text by avoiding the complexities of translating operator logic and category references.

**Actual Outcome:**
The script ran successfully using the revised prompt (`prompts/caldara_translation_chinese_and_operator.json`) and model (`gpt-4.1-2025-04-14`). The generated output file (`data/dictionaries/caldara_traditional_chinese_and_ops.json`) contains the full expected JSON structure, including the `metadata`, `categories` (untranslated with AND operators), and `word_sets` (with translated Traditional Chinese terms) keys, matching the requirements.

**Comparison with Lau (2024) Dictionary (`data/dictionaries/yeu_tong_lau_2024.json`)**

A comparison between the Experiment 2 dictionary (translated Caldara) and Lau's (2024) dictionary reveals significant differences in scope, structure, and terminology:

1.  **Scope and Focus**:
    *   **Lau (2024)**: Highly specific to Taiwan/Cross-Strait geopolitical context (e.g., includes terms like "Taiwan Strait War", "Attack Taiwan", "PLA Aircraft", "Midline of the Taiwan Strait").
    *   **Exp2 (Translated Caldara)**: Reflects the global scope of the original Caldara & Iacoviello (2022) dictionary, containing more generic terms (e.g., "war", "conflict", "revolution", "troops").

2.  **Categories**:
    *   **Lau**: Uses fewer, more focused categories with simple `+` logic.
    *   **Exp2**: Uses a broader set of categories derived from Caldara, combined with `AND` logic.

3.  **Word Sets (Notable Differences)**:
    *   **War/Peace/Military**: Lau includes Taiwan-specific conflict/peace scenarios and PLA-related terms. Exp2 uses general terms. Minimal overlap except for basic terms like "Military".
    *   **Threat**: Some overlap exists (threat, warning, risk, danger, crisis). Lau includes Taiwan-relevant terms like "provocation" and "pressure". Exp2 includes more general terms like "imminent" and "dispute".
    *   **Buildup**: Focus differs. Lau: military readiness/exercises. Exp2: mobilization, sanctions, embargoes.
    *   **Terminology Nuance**: Exp2 uses `飛彈` (missile - common in Taiwan), while Lau uses `導彈` (missile - common in Mainland China).

4.  **Missing Categories**: Exp2 includes several Caldara categories absent in Lau's structure (e.g., nuclear, terrorism, actor-related sets).

**Overall**: The dictionaries serve different analytical goals. Lau's is tailored for Taiwan-specific GPR analysis. Exp2 provides a broader, globally-oriented GPR dictionary translated into Traditional Chinese.

### Experiment 3: Role-Based Taiwan GPR Dictionary Generation (Completed)

**Date:** 2025-05-09 (Assumed)

**Files:**
- Prompt Template: `prompts/generate_taiwan_gpr_dictionary_role_based.json` (Revised version without metadata requirement)
- Results:
    - `data/dictionaries/taiwan_gpr_dict_defense_minister.json`
    - `data/dictionaries/taiwan_gpr_dict_president.json`
    - `data/dictionaries/taiwan_gpr_dict_poli_professor.json`
    - `data/dictionaries/taiwan_gpr_dict_journalist.json`
    - `data/dictionaries/taiwan_gpr_dict_nsb_director.json`
- Model: `gpt-4.1-2025-04-14`

**Goal:**
To generate a GPR dictionary specifically focused on the Taiwan context by moving away from direct translation and employing a role-playing strategy. Pure translation (Exp 1 & 2) proved insufficient for capturing Taiwan-specific terminology and GPR nuances as highlighted by comparison with Lau (2024).

**Process:**
A new prompt template (`prompts/generate_taiwan_gpr_dictionary_role_based.json`) was created with the following characteristics:
1.  **Role-Playing System Message:** The system message uses a `{role}` parameter (e.g., "Taiwanese Prime Minister", "Taiwanese Minister of Defense", "Senior Taiwanese Journalist specializing in Cross-Strait relations") to guide the LLM's perspective.
2.  **Flexible Structure Generation:** The user message instructs the LLM to create `metadata`, `categories`, and `word_sets` focused on Taiwan GPR. It allows flexibility in naming word sets and categories but requires `AND` logic for category definitions.
3.  **Taiwan Context Emphasis:** Instructions explicitly ask for keywords and concepts relevant to Taiwanese news media discussions of GPR.
4.  **Template Rendering:** The prompt's metadata includes `"render_content_as_template": true` to enable the `{role}` parameter substitution by the `PromptManager`.

**Execution:**
The `main.py generate-dictionary` script was executed 5 times using the `prompt_file` strategy with the `generate_taiwan_gpr_dictionary_role_based.json` template. Each run used a different `--role` argument:
1.  `"Taiwanese Minister of Defense"` -> `taiwan_gpr_dict_defense_minister.json`
2.  `"Taiwanese President"` -> `taiwan_gpr_dict_president.json`
3.  `"Taiwanese Politics Professor"` -> `taiwan_gpr_dict_poli_professor.json`
4.  `"Senior Taiwanese Journalist specializing in Cross-Strait Relations"` -> `taiwan_gpr_dict_journalist.json`
5.  `"Director of Taiwan's National Security Bureau (NSB)"` -> `taiwan_gpr_dict_nsb_director.json`

All runs completed successfully. The `DictionaryGenerator` code was modified to add metadata programmatically (including the role, model, date, etc.) to the LLM-generated `categories` and `word_sets` before saving.

**Expected Outcome:**
A set of dictionaries, each generated by assigning a different expert role to the `{role}` parameter. These dictionaries are expected to contain more relevant and nuanced Traditional Chinese keywords and GPR categories for the Taiwan context compared to the direct translation approach. Analysis will compare the dictionaries generated by different roles.

**Next Steps:**
1. Analyze and compare the content (`categories` and `word_sets`) of the 5 generated dictionaries to assess the impact of the role-playing strategy.

**Initial Comparison Summary (Role-Based Dictionaries vs. Lau)**

A comparison between the 5 role-based dictionaries generated in Exp 3 and Lau's (2024) dictionary was conducted:

1.  **Specificity & Relevance**: All 5 role-based dictionaries are significantly more focused on the Taiwan/Cross-Strait context compared to the translated Caldara dictionary (Exp 2). They include Taiwan-specific terms (e.g., related to PLA activity, cross-strait relations, US-Taiwan relations) making them more aligned with Lau's dictionary in terms of contextual relevance.
2.  **Structure**: The role-based prompts successfully produced the desired structure (`categories` and `word_sets` keys) with descriptive, flexible names for the word sets (e.g., `pla_activity`, `diplomatic_pressure`, `cyber_security`) and categories using `AND` logic. This contrasts with Lau's fixed set names and simpler `+` logic.
3.  **Role Influence**: The assigned role clearly influenced the generated themes and keywords. For example, the 'Defense Minister' dictionary emphasized military threats and readiness; the 'President' dictionary included broader political themes like internal stability and foreign intervention; the 'NSB Director' dictionary placed a stronger emphasis on intelligence-related topics like cyber security (adding "cyber espionage"), disinformation (adding "cognitive warfare", "opinion infiltration"), and internal stability threats (adding "internal contradictions", "social division").
4.  **Terminology**: The role-based dictionaries generally used Taiwan-specific terminology (e.g., `解放軍`, `共軍`). Notably, like Exp 2, they tended to use `飛彈` (fēidàn - missile, common in Taiwan) rather than `導彈` (dǎodàn - common in Mainland China) found in Lau's dictionary.
5.  **Comprehensiveness**: Compared to Lau's focused sets, the role-based dictionaries consistently generated a wider range of thematic keyword sets, explicitly covering areas like cyber/disinformation, economic coercion, and internal politics.

**Conclusion**: Experiment 3 demonstrates that the role-playing strategy successfully generates contextually relevant dictionaries for Taiwan GPR, offering potentially richer thematic coverage than Lau's dictionary and significantly more specificity than direct translation (Exp 2). The choice of role impacts the specific focus and terminology within this relevant set, with roles like NSB Director highlighting specific threat vectors like information warfare and internal security.

2. Monitor and evaluate the effectiveness of the role-playing strategy in capturing Taiwan-specific terminology and GPR nuances.
3. Determine whether supplementation is necessary based on the analysis of the generated dictionaries.