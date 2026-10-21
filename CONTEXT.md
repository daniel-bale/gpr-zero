# Geopolitical risk measurement

GPR-Zero measures geopolitical risk in Taiwanese news using dictionaries generated
by language models. Analyses compare risk across time, categories, and news outlets.

## Language

**GPR dictionary**:
Named keyword groups and category expressions that identify geopolitical risk in
articles. Step 1 dictionaries reflect prompting roles; Step 2 uses eight fixed categories.

**Dictionary inventory**:
The collection of dictionaries whose combined vocabulary defines word-presence
indicators. Selecting dictionaries for an analysis does not change that vocabulary.

**Article cohort**:
The articles included in an analysis, identified by their news outlets and date
range. The three-outlet cohort contains China Times, Liberty Times, and United Daily.

**Binary matrix**:
An article collection with an indicator of whether each vocabulary word occurs in
each article. Its vocabulary and article cohort are part of its identity.

**Article matching**:
The assignment of articles to dictionary categories using their text, word-presence
indicators, or category labels supplied by historical source exports.

**Weekly spike event**:
A run of observed weeks above a GPR threshold, with a start week, end week, and first
highest week as its peak. Its article date window includes all seven days of the end week.
