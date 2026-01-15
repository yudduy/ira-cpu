"""
Configuration for Climate Policy Uncertainty (CPU) Index Builder

HOW TO USE THIS FILE:
--------------------
1. Review the keywords below - do they match your research goals?
2. Set your date range (START_DATE, END_DATE)
3. After getting API access, fill in SOURCE_IDS (see instructions below)

You generally don't need to change anything else.
"""

# =============================================================================
# DATE RANGE
# =============================================================================
# What time period should we analyze?
# Format: "YYYY-MM-DD"

START_DATE = "2021-01-01"
END_DATE = "2025-12-31"  # Set to today's date or leave as future date


# =============================================================================
# NEWS SOURCES
# =============================================================================
# These are the approved high-credibility sources for the index.
#
# IMPORTANT: You need to look up the Source IDs using the LexisNexis WSAPI:
# 1. Go to https://solutions.nexis.com/wsapi
# 2. Login with your Stanford credentials
# 3. Go to Ref. Materials > Sources
# 4. Search for each source and copy its ID
#
# Leave as None if a source is not available in LexisNexis.

SOURCE_IDS = {
    "Financial Times": None,          # Example: "MTA2OTUwNQ"
    "Wall Street Journal": None,
    "New York Times": None,
    "Washington Post": None,
    "Reuters": None,
    "Bloomberg": None,
    "Politico": None,
    "The Economist": None,
}


# =============================================================================
# CLIMATE & ENERGY KEYWORDS
# =============================================================================
# Articles must contain AT LEAST ONE term from this list.
# These identify articles about climate/energy topics.

CLIMATE_TERMS = [
    "climate",
    "climate change",
    "clean energy",
    "renewable",
    "renewable energy",
    "decarbonization",
    "carbon",
    "greenhouse gas",
    "net zero",
    "EV",
    "electric vehicle",
    "hydrogen",
    "solar",
    "wind",
    "battery",
    "grid",
]


# =============================================================================
# POLICY KEYWORDS
# =============================================================================
# Articles must ALSO contain AT LEAST ONE term from this list.
# These identify articles about policy/government action.

POLICY_TERMS = [
    "policy",
    "regulation",
    "tax credit",
    "subsidy",
    "grant",
    "incentive",
    "Congress",
    "DOE",
    "Department of Energy",
    "EPA",
    "Treasury",
    "IRS",
    "White House",
    "legislation",
    "Inflation Reduction Act",
    "IRA",
]


# =============================================================================
# UNCERTAINTY KEYWORDS
# =============================================================================
# For the NUMERATOR: articles that express uncertainty about policy.
# These words suggest doubt, delay, or instability.

UNCERTAINTY_TERMS = [
    "uncertain",
    "uncertainty",
    "unclear",
    "unpredictable",
    "delay",
    "delayed",
    "freeze",
    "frozen",
    "rollback",
    "repeal",
    "reversal",
    "litigation",
    "lawsuit",
    "challenge",
    "suspend",
    "halt",
    "block",
]


# =============================================================================
# LLM VALIDATION SETTINGS
# =============================================================================
# We use GPT-5 Nano to spot-check if our keywords are accurate.
# This is very cheap (~$0.01 per 100 articles).

LLM_MODEL = "gpt-5-nano"      # Cheapest OpenAI model, great for classification
LLM_SAMPLE_SIZE = 100         # Articles to classify per validation run
LLM_TEMPERATURE = 0.0         # Deterministic output (no randomness)


# =============================================================================
# API SETTINGS (Advanced - usually don't need to change)
# =============================================================================

LEXISNEXIS_BASE_URL = "https://services-api.lexisnexis.com/v1"
MAX_RESULTS_PER_QUERY = 50    # LexisNexis limit
REQUEST_DELAY_SECONDS = 0.5   # Pause between API calls (be nice to shared quota)


# =============================================================================
# FILE PATHS (Advanced - usually don't need to change)
# =============================================================================

DB_PATH = "data/cpu.db"
EXPORT_DIR = "data/exports"
