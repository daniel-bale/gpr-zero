"""Centralized constants for the GPR-Zero project.

Shared outlet mappings, categories, and study parameters. Runtime paths belong
to ``gpr_zero.project.Project`` and are resolved when a command runs.
"""

# --- Database ---
BINARY_MATRIX_TABLE = "gpr_binary_matrix_all_words"

# --- Outlets ---
# Canonical English names (as stored in binary matrix / DB)
# TODO: this should be explored in stage1 and stored into DB / metadata, not hardcoded here
TARGET_OUTLETS = (
    "China Times (中國時報)",
    "Liberty Times (自由時報)",
    "United Daily (聯合報)",
)

OUTLET_CODE_TO_ENGLISH: dict[str, str] = {
    "chinatimes": "China Times (中國時報)",
    "ltn": "Liberty Times (自由時報)",
    "udn": "United Daily (聯合報)",
}

OUTLET_ENGLISH_TO_CODE: dict[str, str] = {
    v: k for k, v in OUTLET_CODE_TO_ENGLISH.items()
}

OUTLET_CHINESE_TO_CODE: dict[str, str] = {
    "中國時報": "chinatimes",
    "自由時報": "ltn",
    "聯合報": "udn",
}

# --- Categories (Step 2) ---
# TODO: these should be explored after step1 and stored into DB / metadata, not hardcoded here
STANDARD_CATEGORIES = (
    "Military Escalation",
    "Diplomatic Isolation",
    "Economic Coercion",
    "Regional Security",
    "Internal Political Instability",
    "Gray Zone Operations",
    "US-China-Taiwan Relations",
    "Cyber and Information Warfare",
)

# Map Chinese / variant labels → canonical English category name
# TODO: this should be explored after step1 and stored into DB / metadata, not hardcoded here
CATEGORY_MAPPING: dict[str, str] = {
    # Identity mappings
    "Military Escalation": "Military Escalation",
    "Diplomatic Isolation": "Diplomatic Isolation",
    "Economic Coercion": "Economic Coercion",
    "Regional Security": "Regional Security",
    "Internal Political Instability": "Internal Political Instability",
    "Gray Zone Operations": "Gray Zone Operations",
    "US-China-Taiwan Relations": "US-China-Taiwan Relations",
    "Cyber and Information Warfare": "Cyber and Information Warfare",
    # Bilingual variants
    "Military Escalation (軍事升級)": "Military Escalation",
    "Diplomatic Isolation (外交孤立)": "Diplomatic Isolation",
    "Economic Coercion (經濟脅迫)": "Economic Coercion",
    "Regional Security (區域安全)": "Regional Security",
    "Internal Political Instability (內部政治不穩定)": "Internal Political Instability",
    "Gray Zone Operations (灰色地帶行動)": "Gray Zone Operations",
    "US-China-Taiwan Relations (美中台關係)": "US-China-Taiwan Relations",
    "Cyber and Information Warfare (網路與資訊戰)": "Cyber and Information Warfare",
    # Chinese-only
    "軍事升級": "Military Escalation",
    "外交孤立": "Diplomatic Isolation",
    "經濟脅迫": "Economic Coercion",
    "區域安全": "Regional Security",
    "內部政治不穩定": "Internal Political Instability",
    "灰色地帶行動": "Gray Zone Operations",
    "美中台關係": "US-China-Taiwan Relations",
    "網路與資訊戰": "Cyber and Information Warfare",
}

# --- Analysis Parameters ---
DATE_RANGE_START = "2017-01-01"
DATE_RANGE_END = "2024-12-31"
