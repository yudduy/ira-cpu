# Research: Climate Policy Uncertainty and Venture Capital Financing
> Started: 2026-01-22

## Summary
Academic literature strongly supports a negative relationship between climate policy uncertainty and cleantech VC investment. Key mechanisms include: (1) policy-dependent demand creating risk for VC investments, (2) long development timelines making cleantech sensitive to regulatory shifts, and (3) carbon price volatility discouraging green investment independent of price levels. Time series methods including VAR, Granger causality, and ARDL models are commonly used to analyze these relationships.

## Concepts
| Concept | Definition | Source |
|---------|------------|--------|
| Environmental Policy Uncertainty (EnvPU) | News-based index measuring uncertainty about US environmental/climate policy; constructed using ML classification of newspaper articles | Noailly, Nowzohour & van den Heuvel (2022) |
| Carbon VIX | Market-based high-frequency measure of carbon price uncertainty using EU ETS options data | Fuchs, Stroebel & Terstegge (2024) |
| Granger Causality | Test whether one time series improves prediction of another; requires stationarity; indicates forecasting ability not true causation | Wikipedia, Time Series Handbook |
| ARDL (Autoregressive Distributed Lag) | Separates long-run and short-run effects; tests for cointegration; accommodates I(0) and I(1) variables | Pesaran, Shin & Smith (2001) |
| DCC-GARCH | Dynamic Conditional Correlation model capturing time-varying correlations between series | V-Lab NYU |

## Literature
| Paper | Authors | Year | Key Contribution | URL |
|-------|---------|------|------------------|-----|
| Does Environmental Policy Uncertainty Hinder Investments Towards a Low-Carbon Economy? | Noailly, Nowzohour, van den Heuvel | 2022 | Develops EnvPU index; shows policy uncertainty reduces VC funding probability for cleantech | https://www.nber.org/papers/w30361 |
| The Role of Venture Capital and Governments in Clean Energy | van den Heuvel, Popp | 2022 | Natural experiment using 2010 Senate shift; startups funded during high uncertainty show superior performance | https://cepr.org/voxeu/columns/role-venture-capital-and-governments-clean-energy |
| Carbon VIX: Carbon Price Uncertainty and Decarbonization Investments | Fuchs, Stroebel, Terstegge | 2024 | 10pp increase in carbon price uncertainty equivalent to €11.2 carbon price decline on investment | https://www.nber.org/papers/w32937 |
| Venture Capital and Cleantech: The Wrong Model | MIT Energy Initiative | 2016 | VC lost >50% of cleantech capital 2006-2011; policy-dependent demand misaligns with VC model | https://energy.mit.edu/publication/venture-capital-cleantech/ |
| Measuring Economic Policy Uncertainty | Baker, Bloom, Davis | 2016 | Foundational EPU index methodology; policy uncertainty reduces investment and employment | https://www.policyuncertainty.com/ |

## Methods
| Technique | Description | When to Use |
|-----------|-------------|-------------|
| Pearson Correlation | Parametric linear correlation; requires normality, stationarity | Quick preliminary assessment of stationary data |
| Spearman Rank Correlation | Non-parametric monotonic correlation; robust to outliers | Non-normal data or ordinal variables |
| Cross-Correlation | Correlation at different time lags | Identifying lead-lag relationships |
| Granger Causality | Tests if X improves prediction of Y; requires stationarity | Directional causality testing |
| VAR (Vector Autoregression) | Multivariate time series; analyzes shock propagation | Macro-level policy-investment relationships |
| ARDL Bounds Test | Long-run equilibrium testing; accommodates mixed I(0)/I(1) | Policy-investment long-term effects |
| DCC-GARCH | Time-varying correlation with heteroskedasticity | Dynamic risk assessment |

## Datasets
| Name | Provider | Coverage | Metrics | Access |
|------|----------|----------|---------|--------|
| PitchBook | PitchBook Data | Global; 2,312+ climate deals (2023) | Deal counts, amounts, stages, sectors | Subscription |
| Crunchbase | Crunchbase | 35,000+ US startups; 20+ years | VC funding by round, investors | Subscription (free tier) |
| BloombergNEF | Bloomberg | Global clean energy; 20+ years | Investment flows by sector | Bloomberg Terminal |
| IEA World Energy Investment | IEA | 2015-2025; global | Annual investment by sector/country | Free |
| Cambridge Associates VC Benchmark | Cambridge Associates | 2,625 US VC funds (1981-2024); $536B | IRR, DPI, RVPI, TVPI | Institutional |
| CTVC Market Intelligence | CTVC | 2024-2025 trends | Deal counts, sector distribution | Free reports |

## VC Metrics for Analysis
| Metric | Definition | Aggregation |
|--------|------------|-------------|
| Deal Count | Number of VC transactions | Monthly/Quarterly |
| Deal Size (Amount) | Capital invested per deal | Median, Mean, Total |
| Stage Distribution | Seed, Series A/B/C, Late-stage | Count by stage |
| Sector Distribution | Clean energy, EV, hydrogen, etc. | Count by sector |
| IRR | Internal Rate of Return | Quarterly fund-level |
| MOIC/TVPI | Multiple on Invested Capital | Fund performance |

## Stationarity Testing
| Test | Null Hypothesis | Interpretation |
|------|-----------------|----------------|
| ADF (Augmented Dickey-Fuller) | Series has unit root (non-stationary) | p < 0.05 → stationary |
| KPSS | Series IS stationary | p < 0.05 → non-stationary |
| Recommendation | Use BOTH tests together | See stationarity protocol |

## Lag Order Selection
| Data Frequency | Criterion | Maximum Lags |
|----------------|-----------|--------------|
| Monthly | AIC most accurate | 12 lags |
| Quarterly | HQC or SIC | 4-8 lags |

## Gaps
| Gap | Impact | Potential Approaches |
|-----|--------|---------------------|
| Limited real-time correlation data | Cannot track immediate policy-investment effects | Event study methodology |
| Mixed frequency data (monthly EPU vs quarterly VC) | Alignment challenges | MIDAS models or interpolation |
| Emerging market VC data limited | Geographic coverage gaps | Focus on US/Europe initially |

## Progress
| Area | Status | Notes |
|------|--------|-------|
| Policy uncertainty effects on VC | Complete | Strong academic evidence; Noailly et al. (2022) key paper |
| Methodological approaches | Complete | VAR, Granger, ARDL well documented |
| VC metrics | Complete | Deal count, amount, stage, sector key metrics |
| Time series correlation methods | Complete | Comprehensive methodology framework |
| Climate tech VC datasets | Complete | PitchBook, Crunchbase, BNEF, IEA main sources |
