# Climate Policy Credibility & Uncertainty Measure (CPU)

## Overview
This project constructs news-based, time-varying measures of climate policy uncertainty and credibility. The measures follow the logic of Baker, Bloom & Davis (2016), adapted to the climate-policy context, and are intended as mechanism/validation evidence for how policy credibility and stability shape private capital allocation in climate technology.

## Conceptual Goal
- Measure shared, market-level uncertainty and credibility of climate policy as reflected in authoritative public discourse.
- Use the measures for validation and mechanism analysis (not as primary causal treatments).

## Approved News Sources
Only use national or international outlets with high editorial credibility:
- Financial Times
- Wall Street Journal
- New York Times
- Washington Post
- Reuters
- Bloomberg
- Politico (Energy & Climate sections)
- The Economist

(Optional if available: E&E News)

Do NOT use social media, blogs, firm websites, local newspapers, or advocacy-only sources.

## Time Coverage
- Period: January 2021 — most recent available month
- Aggregation: Monthly

## Article Selection
1. Identify climate-policy articles containing at least one term from each group below.
	- Climate / Energy terms: climate, clean energy, renewable, decarbonization, carbon, EV, hydrogen, grid, battery, solar, wind
	- Policy / Government terms: policy, regulation, tax credit, subsidy, grant, DOE, Treasury, IRS, Congress

2. Flag articles expressing uncertainty using uncertainty-language tokens such as: uncertain, uncertainty, unclear, delay, freeze, rollback, repeal, reversal, litigation, suspend, halt, block

## Index Construction
For each month t compute:

CPU_t = (# climate-policy articles with uncertainty language in month t) / (# all climate-policy articles in month t)

Normalize the index for interpretability (e.g., set mean = 100).

## Sanity Checks (Required)
- Event validation: verify spikes around major events (e.g., legislative/administrative freezes, legal challenges) and stability or decline post-major policy milestones (e.g., IRA).
- Source robustness: recompute the index excluding one outlet at a time to assess sensitivity.
- Placebo test: show CPU is not simply driven by overall climate news volume.

## Deliverables
- Monthly CPU time series (CSV)
- Code repository for data collection and index construction (scripts + documentation)
- 2–3 page summary memo describing sources, keyword lists, validation tests, and limitations

## Key Principle
We measure shared uncertainty and credibility of climate policy (not sentiment).

## Contact / Notes
For questions or to propose additional sources/keywords, please open an issue in the repository or contact the project lead.
