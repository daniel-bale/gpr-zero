# Improving Geopolitical Risk Assessment: A Novel LLM-Based Approach with Application to Taiwan

## Research Context and Problem Statement
This master's thesis addresses key limitations in current Geopolitical Risk (GPR) index methodologies, which primarily rely on English-language news sources and manually curated dictionaries. Building upon the foundational work of Caldara & Iacoviello (2022) and the Taiwan-specific application by Yeu-Tong Lau (2024), this research aims to develop a more robust, accessible, and culturally sensitive approach to geopolitical risk assessment with a focus on Taiwan as a case study.

## Research Objectives
1. To develop an LLM-generated keyword dictionary for Chinese-language geopolitical risk assessment
2. To quantify and explain discrepancies between GPR indices based on mainstream English-language sources and those derived from local Chinese-language news
3. To evaluate the effectiveness of zero-shot LLM classification for geopolitical risk assessment compared to traditional dictionary-based methods

## Methodological Approach
The research employs a progressive three-phase methodology:

### Phase 1: LLM-Generated Chinese Keyword Dictionary
- Design multiple prompting techniques to generate Chinese geopolitical risk dictionaries
- Document improvements in cross-cultural applicability compared to human-generated dictionaries

### Phase 2: Comparative GPR Index Analysis and Dictionary Validation
- Calculate the GPR index following Caldara & Iacoviello's (2022) methodology using the LLM-generated dictionary
- As a secondary exploration, examine the effect of normalizing by total news volume
- Validate the LLM-generated dictionary by comparing index spikes with actual geopolitical events
- Analyze discrepancies between established GPR indices (based on English sources) and the new Chinese-language index
- Manually review news during divergent periods to explain why certain risks appear in local news but not in international coverage

### Phase 3: Zero-Shot LLM Classification Method
- Implement direct LLM-based classification of geopolitical risk and sentiment without fine-tuning
- Compare the effectiveness of this approach with traditional dictionary-based methods
- Document advantages in accessibility, transferability, and comprehensiveness

## Data and Scope
- Utilize the same Taiwanese news sources as Lau (2024) for methodological consistency
- Extend the temporal coverage to approximately 30 years (vs. Lau's shorter timeframe)
- Compare results with existing GPR indices based on international English-language sources

## Expected Contributions
1. A methodological framework for developing localized GPR indices that reduces barriers to implementation
2. Empirical evidence of structural differences in geopolitical risk detection between localized and international indices
3. Validation of zero-shot LLM approaches as viable alternatives to dictionary-based methods
4. A more comprehensive, accessible, and transferable approach to geopolitical risk assessment

## Comparative Advantages Over Previous Work
This research improves upon Yeu-Tong Lau's (2024) work by:
1. Providing thorough justification for dictionary selection through systematic validation
2. Exploring the potential impact of normalization by total news volume
3. Extending the time range to 30 years for more robust historical analysis
4. Documenting a methodological approach that can be replicated across different cultural contexts 

## Broader Implications
While this research focuses specifically on improving GPR assessment with Taiwan as a case study, the methodological approach has potential universal applications:

1. The LLM-based keyword generation and classification methodology could be adapted to create indices for any specialized domain requiring text analysis (e.g., environmental risk, technological disruption, public health concerns).

2. The approach significantly reduces human intervention requirements, democratizing index creation across domains where specialized expertise might be limited.

3. The comparative framework between general and localized news sources could be valuable for understanding information asymmetries in various contexts beyond geopolitics.

This research may thus serve as a proof-of-concept for a more generalizable approach to automated, language-agnostic index construction with minimal human input.

## Project Timeline and Scope Considerations

Given the 3-month timeframe for completion, this project adopts focused scope boundaries and a practical implementation approach:

### Monthly Milestones
1. **Month 1: LLM Dictionary Generation (Phase 1)**
   - Test multiple prompt engineering approaches for Chinese keyword generation
   - Select and refine the most effective dictionary
   - Document methodology and initial dictionary evaluation

2. **Month 2: GPR Index Calculation and Comparison (Phase 2)**
   - Implement Caldara & Iacoviello's (2022) methodology with the new dictionary
   - Calculate indices and identify key divergence points from existing indices
   - Analyze a select number of significant divergent events through manual review

3. **Month 3: Zero-Shot Classification and Integration (Phase 3)**
   - Implement and evaluate LLM-based direct classification approach
   - Compare results with dictionary-based methods
   - Compile findings and complete thesis documentation

### Practical Scope Boundaries
- **Time Period**: While originally planning to cover 30 years, a more focused 10-15 year period may be selected based on data availability and processing constraints
- **News Sources**: Utilize a manageable set of representative Taiwanese news sources rather than attempting comprehensive coverage
- **Model Selection**: Use readily available LLMs rather than custom training or fine-tuning
- **Evaluation Metrics**: Focus on 1-2 clear metrics per phase (e.g., correlation with established indices, coverage of known events)

### Implementation Approach
- Leverage existing libraries and tools for text processing and visualization
- Document limitations honestly rather than attempting to address all methodological challenges
- Focus on demonstrating proof-of-concept rather than creating a production-ready system
- Build in buffer time (1-2 weeks) for unexpected technical challenges 