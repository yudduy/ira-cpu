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
    # OBBBA-related terms (One Big Beautiful Bill Act - 2025)
    "OBBBA",
    "One Big Beautiful Bill",
    "Big Beautiful Bill",
    "reconciliation",
    "phase-out",
    "phase out",
]


# =============================================================================
# UNCERTAINTY KEYWORDS (General)
# =============================================================================
# For the standard NUMERATOR: articles that express uncertainty about policy.
# These words suggest doubt, delay, or instability (direction-neutral).

UNCERTAINTY_TERMS = [
    "uncertain",
    "uncertainty",
    "unclear",
    "unpredictable",
    "unknown",
    "question",
    "doubt",
    "risk",
    "risky",
]


# =============================================================================
# DOWNSIDE UNCERTAINTY KEYWORDS (CPU-Down)
# =============================================================================
# For CPU-Down index: articles suggesting policy WEAKENING or ROLLBACK.
# Based on Segal et al. (2015) "bad uncertainty" and Forni et al. (2025)
# "downside uncertainty shocks" methodology.
#
# These indicate potential policy deterioration, removal, or weakening.

DOWNSIDE_TERMS = [
    # Direct rollback/removal actions
    "rollback",
    "repeal",
    "reverse",
    "reversal",
    "revoke",
    "rescind",
    "rescission",
    "withdraw",
    "withdrawal",
    "eliminate",
    "elimination",
    "eliminated",
    "abolish",
    "dismantle",
    "gut",
    "scrap",
    "kill",
    "end",
    "expire",
    "expiration",
    # Weakening/reduction actions
    "cut",
    "cuts",
    "reduce",
    "reduction",
    "weaken",
    "undermine",
    "water down",
    "scale back",
    "phase out",
    "phase-out",
    "wind down",
    # Stoppage/delay actions
    "halt",
    "stop",
    "freeze",
    "frozen",
    "pause",
    "suspend",
    "suspension",
    "delay",
    "delayed",
    "block",
    "blocked",
    "obstruct",
    # Legal/political challenges
    "litigation",
    "lawsuit",
    "challenge",
    "challenged",
    "overturn",
    "overturned",
    "invalidate",
    "strike down",
    "struck down",
    # Termination
    "terminate",
    "terminated",
    "cancel",
    "cancelled",
    "canceled",
    "abandon",
    "abandoned",
    "exit",
]


# =============================================================================
# UPSIDE UNCERTAINTY KEYWORDS (CPU-Up)
# =============================================================================
# For CPU-Up index: articles suggesting policy STRENGTHENING or EXPANSION.
# Based on Segal et al. (2015) "good uncertainty" methodology.
#
# These indicate potential policy enhancement, expansion, or new support.

UPSIDE_TERMS = [
    # Expansion/strengthening actions
    "expand",
    "expansion",
    "extend",
    "extension",
    "strengthen",
    "enhance",
    "increase",
    "boost",
    "accelerate",
    "ramp up",
    "scale up",
    # New support/investment
    "invest",
    "investment",
    "fund",
    "funding",
    "support",
    "commit",
    "commitment",
    "pledge",
    "allocate",
    "allocation",
    # Policy creation/mandate
    "mandate",
    "require",
    "requirement",
    "introduce",
    "launch",
    "establish",
    "create",
    "implement",
    "implementation",
    "enact",
    "pass",
    "passed",
    "adopt",
    "adopted",
    # Incentivization
    "incentivize",
    "subsidize",
    "incentive",
    "subsidy",
    "credit",
    "rebate",
    "grant",
    # Ambition/targets
    "ambitious",
    "aggressive",
    "bold",
    "target",
    "goal",
    # Transition terms (positive framing)
    "transition",
    "transform",
    "transformation",
    "modernize",
    "upgrade",
    "advance",
    "progress",
    # Net-zero/climate commitment
    "net-zero",
    "net zero",
    "carbon neutral",
    "carbon-neutral",
    "decarbonize",
    "decarbonization",
    "clean energy",
    "renewable",
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
