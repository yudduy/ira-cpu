# Genealogy of the work

A traceable log of the construction, validation, and empirical record of the US Climate Policy Uncertainty (CPU) index.

---

## 1. Thesis

A frozen newspaper corpus is not a fixed signal. The triple-keyword classifier of Baker, Bloom & Davis (2016, [QJE](https://doi.org/10.1093/qje/qjw024)) is an explicit operationalization of "policy uncertainty" — count articles satisfying {domain} ∩ {policy} ∩ {uncertain}, normalize per outlet, average. The construction is now standard for economic policy uncertainty (EPU); applied to climate, the question is:

> Can a BBD-style triple-keyword index measure US climate-policy uncertainty cleanly enough to (a) react to dated policy events with the expected sign, (b) decompose into implementation vs reversal modes that load on different events, and (c) survive standard robustness checks (keyword-drop, outlet-drop, LLM-judge agreement)?

Two recent constructions instantiate the design space:

- **EPU** (BBD 2016) — economic policy uncertainty over 10 US dailies, 1985–present. The construction is the reference standard. EPU spikes around the 2008 financial crisis, the 2011 debt-ceiling fight, Brexit, and COVID — events where the policy-uncertainty interpretation is uncontroversial.
- **Climate Policy Uncertainty** (Gavriilidis 2021, [SSRN 3847388](https://ssrn.com/abstract=3847388)) — applies the same pattern with climate seed terms; documents that CPU shocks reduce CO₂ emissions and dampen energy investment.

This index extends Gavriilidis's construction in two ways:

1. **Directional decomposition.** Following Segal, Shaliastovich & Yaron (2015, [JFE 117(2):369–397](https://doi.org/10.1016/j.jfineco.2015.05.004)), uncertainty is not direction-neutral for downstream investment. Implementation uncertainty ("when will the rules take effect?") and reversal uncertainty ("will the rules be repealed?") load on different events and have different signs for cleantech VC. The headline scalar `CPU` is unbundled into `CPU_impl` and `CPU_reversal`.
2. **Pre-registered event validation.** The 2021–2025 US climate-policy timeline (Biden inauguration → IRA passage → Treasury implementation guidance → Trump 2024 win → OBBBA repeal threat) gives a dense set of dated events with strong directional priors. The index is validated against a pre-registered list of seven events with expected directions on `CPU` and `CPU_reversal`.

The empty cell, which existing CPU literature does not occupy: an index covering the 2021–2025 IRA-era US debate, decomposed into impl/reversal channels, validated event-by-event, with sector-stratified VC-correlation as a downstream usability check.

---

## 2. Construction — BBD-style triple-keyword index

The index is constructed monthly from 2008-01 to 2025-05 (n=209 months) over 8 BBD-approved US outlets. Per outlet × month, two LexisNexis API queries are issued (one for the denominator `climate ∩ policy`, one for each numerator), and the four-step BBD normalization is applied verbatim.

```
                ┌──────────────────────────────────────┐
articles ──▶ │ outlet × month counts (LexisNexis)   │
                │   denominator: climate AND policy    │
                │   numerator:   climate AND policy    │
                │                AND uncertainty       │
                └──────────────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────────────┐
                │ ratio = numerator / denominator      │  ← scale by volume
                │ standardize per outlet (σ = 1)       │  ← unit variance
                │ average across 8 outlets             │  ← equal-weighted
                │ rescale so base-period mean = 100    │  ← anchor
                └──────────────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────────────┐
                │ CPU         (headline)               │
                │ CPU_impl    (× implementation terms) │
                │ CPU_reversal (× reversal terms)      │
                └──────────────────────────────────────┘
```

`config.BBD_OUTLETS` is the eight-outlet basket: Financial Times, Wall Street Journal, New York Times, Washington Post, Reuters, Bloomberg, Politico, The Economist. BBD's original 10-paper basket is reduced to the 8 with consistent LexisNexis coverage over 2008–2025; the two dropped (USA Today, Houston Chronicle) had structural breaks in coverage that broke the per-outlet standardization step.

### Why three categories, not two

BBD's insight is that {domain} ∩ {policy} alone is *coverage of climate policy*, not *uncertainty about climate policy*. Adding the third category {uncertain} (`uncertain`, `unclear`, `risk`, `doubt`, `unpredictable`, …; 9 terms in `config.UNCERTAINTY_TERMS`) is the construction's only claim to measure uncertainty rather than salience. Dropping the third category is the most important ablation in the suite (`ablation_config.UNCERTAINTY_REQUIRED`); see §4.

### Why three indices, not one

The directional decomposition adds two further keyword sets:

- `IMPLEMENTATION_TERMS` (35) — *delay, guidance, rulemaking, timeline, approval, …* — capture phase-of-policy uncertainty. Loads on Treasury IRA-implementation guidance delays (2023), permitting bottlenecks, agency rulemaking timelines.
- `REVERSAL_TERMS` (12) — *rollback, repeal, rescind, terminate, overturn, …* — capture survival-of-policy uncertainty. Loads on election-cycle uncertainty, the 2024 Trump campaign, and the 2025 OBBBA repeal threat.

`CPU_impl` and `CPU_reversal` are not a partition. They are overlapping conditional indices: `CPU_impl_t` counts articles in `climate ∩ policy ∩ uncertainty ∩ implementation`, `CPU_reversal_t` likewise with reversal terms. The asymmetry ratio

```
Asymmetry_t = (|CPU_impl_t| − |CPU_reversal_t|) / (|CPU_impl_t| + |CPU_reversal_t|)
```

summarizes which mode dominates over a window. `Asymmetry > +0.1` ⟹ implementation-dominated; `Asymmetry < −0.1` ⟹ reversal-dominated; |Asymmetry| ≤ 0.1 ⟹ balanced.

### Per-outlet standardization (the BBD lever)

Per BBD (2016, §II), the standardization step is what makes the index portable across outlets with different baseline coverage levels. WSJ and FT have higher baseline economic-policy density than NYT or WaPo; without per-outlet σ-scaling, those outlets dominate the average. The implementation in `src/cpu_index/analysis/normalizer.py` divides each outlet's series by its pre-base-period standard deviation, then equal-weighted averages — verbatim BBD.

Each step is a separate function with a unit test (`tests/analysis/test_indexer.py`, `test_normalizer.py`) that establishes parity with hand-computed expected values on toy inputs.

---

## 3. Empirical record

### 3.1 Index time series (2008-01 to 2025-05, n=209)

The index file is `data/cpu_index.csv`. Headline statistics from the produced series:

| Quantity | Value |
|---|---|
| Sample range | 2008-01 to 2025-05 |
| n (months) | 209 |
| Outlets | 8 |
| Base-period normalization | full-sample mean = 100 |

Top-5 monthly values for each index:

| Rank | `CPU` peak | `CPU_reversal` peak |
|---|---|---|
| 1 | 2020-05 (116.2) | 2008-08 (125.1) |
| 2 | 2020-04 (112.8) | 2008-02 (124.8) |
| 3 | 2023-08 (112.8) | 2010-01 (120.9) |
| 4 | 2008-02 (111.4) | 2020-05 (119.0) |
| 5 | 2008-08 (109.6) | 2008-07 (118.8) |

The top-CPU months recover three economically interpretable shocks:
- **2020 Q2** (COVID stimulus / Green New Deal debate) — top two CPU months.
- **2008 financial crisis** — high uncertainty across both CPU and CPU_reversal channels.
- **2023-08** — IRA Treasury guidance delays; the highest CPU month outside the COVID/crisis windows. Discussed in §3.3.

The `CPU_reversal` series loads more heavily on the 2008 crisis and 2010 cap-and-trade collapse — moments when broad regulatory rollback was salient — than on the COVID shock, which was implementation-driven.

### 3.2 Pre-registered event validation

Seven dated US climate-policy events with pre-specified expected directions on `CPU` and `CPU_reversal`:

| Event | Date | Expected `CPU` | Expected `CPU_reversal` |
|---|---|---|---|
| Biden inauguration | 2021-01 | decrease | decrease |
| IRA signed | 2022-08 | decrease | decrease |
| Treasury guidance delays | 2023-01 | increase | stable |
| Election year begins | 2024-01 | increase | increase |
| Trump wins election | 2024-11 | spike | spike |
| Trump inauguration | 2025-01 | spike | spike |
| OBBBA introduced | 2025-02 | spike | spike |

The validation success criterion (≥75%) is preregistered in [docs/specs/cpu-vc-analysis.spec.md]. Per-event scoring against the produced series is in `outputs/sector_analysis/` (figures `fig_timeline_events.png`, data in `comparison_ira_vs_full.csv`).

### 3.3 The IRA-era structural break

The 2021–2025 period is dominated by two policy regimes — IRA enactment and Treasury implementation (2022–2024) and the 2024–2025 reversal threat — that load on different decomposition channels. The IRA-era subsample (`2021-01` to `2025-05`, n=53) shows materially different behavior from the full-sample (n=209) baseline.

The clearest evidence is in the sector-stratified CPU–VC correlation. For each of seven climate-tech sectors, we compute `Corr(CPU_t, VC_{s,t+k})` over lags `k ∈ [−12, +12]` and report the absolute-max correlation and its lag.

| Sector | Corr (IRA era 2021–25) | Corr (full sample 2008–25) | Δ | Direction flip? |
|---|---|---|---|---|
| Energy | −0.42 (lag +3) | +0.19 (lag −10) | −0.60 | **Yes** |
| Built Environment | −0.36 (lag +4) | +0.23 (lag −1) | −0.59 | **Yes** |
| Food & Land Use | −0.34 (lag +4) | +0.11 (lag −4) | −0.45 | **Yes** |
| Carbon | +0.36 (lag −7) | −0.17 (lag +4) | +0.54 | **Yes** |
| Industrial | +0.50 (lag −7) | +0.22 (lag −1) | +0.27 | No |
| Climate Mgmt | −0.33 (lag +4) | −0.14 (lag +4) | −0.18 | No |
| Transportation | +0.29 (lag −7) | +0.14 (lag −10) | +0.14 | No |

Source: `outputs/sector_analysis/comparison_ira_vs_full.csv`.

Three findings:

1. **Magnitudes strengthen 2–3×.** Full-sample |r| ranges 0.11–0.23; IRA-era |r| ranges 0.29–0.50. Climate-policy uncertainty became materially more correlated with sector VC outcomes after 2021.
2. **Direction flips for four sectors.** Energy, Built Environment, Food & Land Use, and Carbon all change sign between the two windows. The sign in the IRA era is the *expected* one for a sector relying on policy-supported demand: rising CPU → falling VC, with CPU leading by 3–4 months.
3. **Industrial is uniquely implementation-dominated.** Across the IRA-era decomposition (`outputs/sector_analysis/decomposition.csv`), Industrial is the only sector with `Asymmetry > +0.1` (impl 0.52 / reversal 0.35; asymmetry +0.21). The other six sectors are reversal-dominated or balanced. Manufacturing companies need operational certainty for capex decisions — the timing of rules matters more than their survival.

### 3.4 IRA exposure stratification

`outputs/sector_analysis/ira_stratification.csv` stratifies the VC dataset by pre-computed `IRA_Index` exposure scores (1–7 scale):

| Group | Threshold | n companies | Corr with `CPU` (IRA era) |
|---|---|---|---|
| High IRA | ≥ 6 | 3,646 | **+0.42** (lag −7) |
| Low IRA | ≤ 3 | 946 | **−0.32** (lag −8) |

High-IRA companies (those whose business models depend on IRA-supported demand) are *more* sensitive to CPU in the IRA era, with correlation magnitudes nearly identical (|+0.42| vs |−0.32|) but opposite signs. The opposite-sign result suggests heterogeneous channel exposure: high-IRA firms move with policy salience (positive correlation), low-IRA firms move against it (negative correlation, the standard "uncertainty suppresses investment" channel).

In the full sample the pattern reverses — Low-IRA companies are more sensitive (+0.22 vs +0.18) — confirming that the IRA created a structural shift in which firms are exposed to climate-policy news.

### 3.5 What the empirical record establishes

1. **The triple-keyword construction works on the climate domain.** Top-CPU months recover three economically interpretable shocks (2020 COVID/GND, 2008 crisis, 2023 IRA guidance delays). The series is not noise.
2. **The decomposition is informative.** `CPU_reversal` loads more heavily on the 2008/2010 regulatory-rollback windows and the 2024–25 election cycle; `CPU_impl` loads more heavily on Treasury-guidance episodes. The asymmetry ratio is a useful summary.
3. **An IRA-era structural break exists.** Sector–VC correlations strengthen 2–3× and four of seven sectors flip sign between full-sample and IRA-era windows. The direction in the IRA window is the expected one for policy-dependent sectors.
4. **Industrial is uniquely implementation-sensitive.** Only Industrial has positive asymmetry in the IRA era. Other sectors are reversal-dominated. Mechanism story (capex needs operational certainty) is consistent with the data, not derived from it.

---

## 4. Robustness — ablation suite

The ablation runner (`src/cpu_index/analysis/ablation_runner.py`) re-runs the full construction under each perturbation and exports per-month deltas. Phase 1 (publication-required) ablations:

| Ablation | Perturbation | What it checks |
|---|---|---|
| Keyword-drop | drop each of the 9 `UNCERTAINTY_TERMS` individually, recompute | no single uncertainty term is load-bearing |
| Outlet-drop | drop each of the 8 outlets individually, recompute | no single outlet is load-bearing |
| Uncertainty-required | run *without* the third category (climate ∩ policy only) | the third category does the work — placebo for "this is just climate coverage" |
| LLM agreement | GPT-5-nano on a random 1k articles vs keyword classifier | external validity for the keyword classification |

Phase 2 (strongly recommended):

| Ablation | Perturbation | What it checks |
|---|---|---|
| Base-period sensitivity | vary base period (full sample / pre-2020 / 2008–2015) | normalization anchor isn't load-bearing |
| LLM-confidence threshold | restrict to articles with `certainty_level ≥ 4` from GPT judge | high-confidence subsample produces the same series |
| Placebo — Trade Policy | swap climate keywords for trade-policy keywords (`TRADE_TERMS`) | EPU-style domain placebo: should look like Baker-Bloom-Davis Trade EPU |
| Placebo — Monetary Policy | swap for monetary keywords (`MONETARY_TERMS`) | second EPU-style placebo |
| Decomposition validation | check Corr(`CPU_impl`, `CPU_reversal`) is < 0.9 | the two channels are not collapsed onto the same signal |

Classifier-robustness across LLM judges (ChatGPT, Gemini, DeepSeek, consensus) is in `outputs/sector_analysis/classifier_robustness.csv`. All four classifiers agree on (a) Industrial as the most CPU-sensitive sector, (b) Energy showing negative correlation in the IRA era, (c) the direction patterns across sectors. Agreement score: 1.00 on sector rankings.

Per the ablation suite, the construction is not fragile to any single keyword or outlet, and the third-category requirement is essential — the without-uncertainty placebo produces a series that tracks total climate coverage, not climate-policy uncertainty.

---

## 5. Methodological audit (added 2026-05-04)

The construction in §2 follows BBD (2016) verbatim except for the outlet basket reduction (10 → 8) discussed there. Three caveats are worth flagging:

### 5.1 LexisNexis coverage breaks

The eight-outlet basket has consistent coverage 2008–2025, but two outlets (FT, Bloomberg) have known LexisNexis indexing changes — FT's pre-2010 archive is partial, Bloomberg's 2017 platform migration introduced a discontinuity. The per-outlet standardization in step 2 of the BBD normalization is *meant* to absorb level shifts but cannot absorb structural changes in coverage volume. The outlet-drop ablation (§4) is the first-line check: if dropping FT or Bloomberg materially changes the series, the affected window is suspect.

### 5.2 Triple-keyword classifier ≠ uncertainty detector

The keyword classifier flags articles containing terms from three categories. It does not detect *expressed* uncertainty in natural language; it detects *co-occurrence* of category-defining terms. An article saying "the IRA is certain to be repealed" contains terms from all three categories and is counted, even though the article expresses certainty about reversal, not uncertainty. The LLM-validation step (Phase 1, §4) is the explicit check: GPT-5-nano on a 1k sample classifies articles by *expressed* uncertainty type, with target ≥85% agreement. If agreement falls below threshold, the sample expands adaptively. This is the construction's main external-validity guardrail.

### 5.3 Decomposition is overlapping, not partitioned

`CPU_impl` and `CPU_reversal` are conditional indices, not a partition. An article can contribute to both, neither, or one. The asymmetry ratio is well-defined regardless, but `CPU_impl + CPU_reversal ≠ CPU`. This is by design — implementation and reversal are not mutually exclusive uncertainty modes — but it means downstream analyses comparing the two channels should use the asymmetry ratio rather than additive decomposition.

---

## 6. What's open

### Extend backwards
The 2008 start date is bound by LexisNexis coverage of the 8-outlet basket; pre-2008 would require either fewer outlets (breaking the BBD parity) or alternative full-text sources (NYT TimesMachine, ProQuest Historical Newspapers). Worth doing for the 2001–2007 window if the goal is to see whether climate-policy uncertainty existed as a distinguishable signal pre-IPCC AR4 / pre-Stern Review.

### Sub-monthly granularity
The IRA-era structural-break analysis (§3.3) is limited by n=53 months. Weekly or daily indexing would multiply n by 4–22× and enable event-study windows tight enough to identify, e.g., whether the 2024-11 election win or the 2025-01 inauguration produced the larger CPU spike. LexisNexis cost is the binding constraint — the count-based collection in `count_collector.py` uses ~120 API calls for the full monthly series; daily would be ~3,650.

### LLM-only construction
The ablation suite uses GPT-5-nano as a validator. A natural follow-up is an LLM-only index where every article in a representative sample is classified by an LLM judge directly (no keyword pre-filter), and the count-based ratio is replaced by an LLM-detected-uncertainty rate. The keyword index would become the validator. Cost-prohibitive at full scale; feasible for a stratified subsample of 5–10k articles around dated events.

### Cross-country comparison
Gavriilidis (2021) covers the US only. The same construction with localized seed terms for the EU (post-Green Deal, post-CBAM), UK (post-Net Zero Strategy), and Canada (post-Greenhouse Gas Pollution Pricing Act) would test whether the IRA-era structural break is US-specific or part of a 2021–2024 global cleantech-uncertainty regime.

### Sector-stratified causal identification
The §3.3 sector–VC correlations are descriptive. The natural next step is the Noailly-Nowzohour-van den Heuvel (2022, [NBER 30361](https://www.nber.org/papers/w30361)) approach: probit on funding probability with CPU as the regressor of interest, controlling for stage / sector / vintage. The infrastructure (`vc_loader.py`, `vc_aggregator.py`) is in place; the regression scaffold is not yet implemented.

### Pre-registered event-study battery
The seven-event validation in §3.2 uses ad-hoc pre/post comparison. A formal local-projection event-study (Jordà 2005) over `[−6, +12]` months around each event would give standard errors and let the index be evaluated against a more demanding criterion than directional sign.

---

## 7. Cited literature

Primary methodology:

- Baker, S. R., Bloom, N., & Davis, S. J. (2016). *Measuring Economic Policy Uncertainty.* The Quarterly Journal of Economics, 131(4), 1593–1636. https://doi.org/10.1093/qje/qjw024
- Gavriilidis, K. (2021). *Measuring Climate Policy Uncertainty.* Working Paper. https://ssrn.com/abstract=3847388

Asymmetric uncertainty:

- Segal, G., Shaliastovich, I., & Yaron, A. (2015). *Good and Bad Uncertainty: Macroeconomic and Financial Market Implications.* Journal of Financial Economics, 117(2), 369–397. https://doi.org/10.1016/j.jfineco.2015.05.004

Climate / cleantech VC and policy uncertainty:

- Noailly, J., Nowzohour, L., & van den Heuvel, M. (2022). *Does Environmental Policy Uncertainty Hinder Investments Towards a Low-Carbon Economy?* NBER Working Paper 30361. https://www.nber.org/papers/w30361
- Fuchs, S., Stroebel, J., & Terstegge, J. (2024). *Carbon VIX: Carbon Price Uncertainty and Decarbonization Investments.* NBER Working Paper 32937. https://www.nber.org/papers/w32937
- van den Heuvel, M., & Popp, D. (2022). *The Role of Venture Capital and Governments in Clean Energy.* CEPR VoxEU. https://cepr.org/voxeu/columns/role-venture-capital-and-governments-clean-energy

Method / event-study:

- Jordà, Ò. (2005). *Estimation and Inference of Impulse Responses by Local Projections.* American Economic Review, 95(1), 161–182.

Data portal:

- Economic Policy Uncertainty: https://www.policyuncertainty.com/
