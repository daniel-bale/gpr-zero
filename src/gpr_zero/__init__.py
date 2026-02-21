"""GPR-Zero: LLM-based surrogate index construction and analysis.

A framework for constructing measurement indices from news corpora using
LLM-generated dictionaries. While developed for geopolitical risk (GPR)
assessment of Taiwan, the methodology applies to any domain where
traditional index construction is ambiguous or labor-intensive.
"""

from gpr_zero.gpr_index.calculator import GPRCalculator as GPRCalculator
from gpr_zero.llm.prompt_manager import PromptManager as PromptManager
