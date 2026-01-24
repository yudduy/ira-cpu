"""
CPU-VC Correlation Analysis Module

Provides time series correlation analysis between Climate Policy Uncertainty (CPU)
and Venture Capital (VC) financing metrics. Implements methodology from academic
literature including Noailly, Nowzohour & van den Heuvel (2022).

Key features:
- Stationarity testing (ADF, KPSS)
- Cross-correlation at multiple lags
- Granger causality testing
- VAR model fitting (optional, requires statsmodels)

Reference:
- Baker, Bloom & Davis (2016) for EPU methodology
- Noailly et al. (2022) for EnvPU-VC analysis framework
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# OPTIONAL IMPORTS (statsmodels for advanced tests)
# =============================================================================

_STATSMODELS_AVAILABLE = False
try:
    from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
    from statsmodels.tsa.api import VAR
    _STATSMODELS_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# CONSTANTS
# =============================================================================

# Default significance level for hypothesis tests
DEFAULT_ALPHA = 0.05

# Maximum lags for cross-correlation
DEFAULT_MAX_LAG = 12

# Minimum observations required for meaningful analysis
MIN_OBSERVATIONS = 24  # 2 years of monthly data


# =============================================================================
# STATIONARITY TESTING
# =============================================================================

def check_stationarity(
    series: pd.Series,
    name: str = "series",
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """
    Test time series stationarity using ADF and KPSS tests.

    Uses both tests together as recommended in the literature:
    - ADF null hypothesis: series has unit root (non-stationary)
    - KPSS null hypothesis: series IS stationary

    Interpretation matrix:
    - ADF rejects, KPSS fails to reject: Stationary
    - ADF fails to reject, KPSS rejects: Non-stationary
    - Both reject: Trend stationary (remove trend)
    - Neither rejects: Inconclusive

    Args:
        series: Time series data as pandas Series
        name: Descriptive name for the series (for output)
        alpha: Significance level for hypothesis tests (default: 0.05)

    Returns:
        Dict containing:
        - name: Series name
        - n_observations: Number of observations
        - adf_statistic: ADF test statistic
        - adf_pvalue: ADF p-value
        - adf_critical_values: Critical values at 1%, 5%, 10%
        - adf_stationary: Whether ADF indicates stationarity
        - kpss_statistic: KPSS test statistic
        - kpss_pvalue: KPSS p-value (may be bounded)
        - kpss_critical_values: Critical values
        - kpss_stationary: Whether KPSS indicates stationarity
        - conclusion: Overall stationarity interpretation
        - differencing_needed: Whether differencing is recommended

    Raises:
        ValueError: If series has fewer than MIN_OBSERVATIONS
        ImportError: If statsmodels is not installed
    """
    if not _STATSMODELS_AVAILABLE:
        raise ImportError(
            "statsmodels is required for stationarity testing. "
            "Install with: pip install statsmodels"
        )

    # Clean series
    clean_series = series.dropna()

    if len(clean_series) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Series '{name}' has only {len(clean_series)} observations. "
            f"Minimum required: {MIN_OBSERVATIONS}"
        )

    # ADF test
    adf_result = adfuller(clean_series, autolag='AIC')
    adf_stat = adf_result[0]
    adf_pvalue = adf_result[1]
    adf_critical = adf_result[4]
    adf_stationary = adf_pvalue < alpha

    # KPSS test (suppress warning about p-value bounds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_result = kpss(clean_series, regression='c', nlags='auto')

    kpss_stat = kpss_result[0]
    kpss_pvalue = kpss_result[1]
    kpss_critical = kpss_result[3]
    # KPSS: low p-value means reject null (reject stationarity)
    kpss_stationary = kpss_pvalue >= alpha

    # Interpret combined results
    if adf_stationary and kpss_stationary:
        conclusion = "stationary"
        differencing_needed = False
    elif not adf_stationary and not kpss_stationary:
        conclusion = "non_stationary"
        differencing_needed = True
    elif adf_stationary and not kpss_stationary:
        conclusion = "trend_stationary"
        differencing_needed = True  # Remove trend or difference
    else:  # not adf_stationary and kpss_stationary
        conclusion = "inconclusive"
        differencing_needed = False  # Conservative: assume stationary

    return {
        "name": name,
        "n_observations": len(clean_series),
        # ADF results
        "adf_statistic": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "adf_critical_values": {k: float(v) for k, v in adf_critical.items()},
        "adf_stationary": adf_stationary,
        # KPSS results
        "kpss_statistic": float(kpss_stat),
        "kpss_pvalue": float(kpss_pvalue),
        "kpss_critical_values": {k: float(v) for k, v in kpss_critical.items()},
        "kpss_stationary": kpss_stationary,
        # Combined interpretation
        "conclusion": conclusion,
        "differencing_needed": differencing_needed,
    }


def make_stationary(
    series: pd.Series,
    max_differences: int = 2,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """
    Transform series to stationary if needed via differencing.

    Applies first differencing and tests stationarity. If still non-stationary,
    applies second differencing. Returns original if already stationary.

    Args:
        series: Time series data as pandas Series
        max_differences: Maximum number of differences to apply (default: 2)
        alpha: Significance level for stationarity tests (default: 0.05)

    Returns:
        Dict containing:
        - series: The (possibly differenced) stationary series
        - original_length: Original series length
        - final_length: Length after differencing
        - differences_applied: Number of differences applied (0, 1, or 2)
        - is_stationary: Whether final series is stationary
        - transformations: List of transformations applied

    Raises:
        ImportError: If statsmodels is not installed
    """
    if not _STATSMODELS_AVAILABLE:
        raise ImportError(
            "statsmodels is required for stationarity testing. "
            "Install with: pip install statsmodels"
        )

    clean_series = series.dropna()
    original_length = len(clean_series)
    transformations = []

    # Test original series
    if len(clean_series) >= MIN_OBSERVATIONS:
        try:
            result = check_stationarity(clean_series, "original", alpha)
            if not result["differencing_needed"]:
                return {
                    "series": clean_series,
                    "original_length": original_length,
                    "final_length": len(clean_series),
                    "differences_applied": 0,
                    "is_stationary": True,
                    "transformations": ["none (already stationary)"],
                }
        except ValueError:
            pass  # Too few observations, proceed with differencing

    # Apply differencing
    current = clean_series
    for d in range(1, max_differences + 1):
        current = current.diff().dropna()
        transformations.append(f"difference_{d}")

        if len(current) < MIN_OBSERVATIONS:
            break

        try:
            result = check_stationarity(current, f"diff_{d}", alpha)
            if not result["differencing_needed"]:
                return {
                    "series": current,
                    "original_length": original_length,
                    "final_length": len(current),
                    "differences_applied": d,
                    "is_stationary": True,
                    "transformations": transformations,
                }
        except ValueError:
            break

    # Return best effort even if not stationary
    return {
        "series": current,
        "original_length": original_length,
        "final_length": len(current),
        "differences_applied": len(transformations),
        "is_stationary": False,
        "transformations": transformations + ["(may still be non-stationary)"],
    }


# =============================================================================
# CROSS-CORRELATION ANALYSIS
# =============================================================================

def cross_correlation(
    series1: pd.Series,
    series2: pd.Series,
    max_lag: int = DEFAULT_MAX_LAG,
) -> pd.DataFrame:
    """
    Compute cross-correlation between two time series at multiple lags.

    Positive lag k means series1 leads series2 by k periods.
    Negative lag -k means series2 leads series1 by k periods.

    Args:
        series1: First time series (e.g., CPU index)
        series2: Second time series (e.g., VC deal count)
        max_lag: Maximum lag in both directions (default: 12)

    Returns:
        DataFrame with columns:
        - lag: Lag value (-max_lag to +max_lag)
        - correlation: Pearson correlation at that lag
        - n_observations: Number of overlapping observations
        - interpretation: Human-readable interpretation

    Notes:
        - Positive correlation at positive lag: series1 increase precedes
          series2 increase
        - Correlation calculated only where both series have valid data
    """
    # Align series by index
    df = pd.DataFrame({
        's1': series1,
        's2': series2,
    }).dropna()

    if len(df) < MIN_OBSERVATIONS:
        return pd.DataFrame({
            'lag': [],
            'correlation': [],
            'n_observations': [],
            'interpretation': [],
        })

    results = []
    lags = range(-max_lag, max_lag + 1)

    for lag in lags:
        if lag > 0:
            # Positive lag: s1 leads s2
            s1_shifted = df['s1'].iloc[:-lag] if lag < len(df) else pd.Series()
            s2_aligned = df['s2'].iloc[lag:] if lag < len(df) else pd.Series()
        elif lag < 0:
            # Negative lag: s2 leads s1
            s1_shifted = df['s1'].iloc[-lag:] if -lag < len(df) else pd.Series()
            s2_aligned = df['s2'].iloc[:lag] if lag < len(df) else pd.Series()
        else:
            # Zero lag: contemporaneous
            s1_shifted = df['s1']
            s2_aligned = df['s2']

        n_obs = len(s1_shifted)
        if n_obs >= 3:  # Minimum for correlation
            corr = s1_shifted.reset_index(drop=True).corr(
                s2_aligned.reset_index(drop=True)
            )
        else:
            corr = np.nan

        # Interpretation
        if pd.isna(corr):
            interp = "insufficient_data"
        elif lag > 0:
            interp = f"series1_leads_by_{lag}_periods"
        elif lag < 0:
            interp = f"series2_leads_by_{-lag}_periods"
        else:
            interp = "contemporaneous"

        results.append({
            'lag': lag,
            'correlation': corr,
            'n_observations': n_obs,
            'interpretation': interp,
        })

    return pd.DataFrame(results)


def find_optimal_lag(
    ccf_results: pd.DataFrame,
    min_correlation: float = 0.0,
) -> dict:
    """
    Find the lag with strongest absolute correlation.

    Args:
        ccf_results: DataFrame from cross_correlation()
        min_correlation: Minimum absolute correlation to consider significant

    Returns:
        Dict with:
        - optimal_lag: Lag with highest absolute correlation
        - max_correlation: Correlation at optimal lag
        - is_significant: Whether max correlation exceeds threshold
        - direction: 'positive' or 'negative' correlation
        - lead_series: Which series leads at optimal lag
    """
    valid = ccf_results.dropna(subset=['correlation'])

    if len(valid) == 0:
        return {
            "optimal_lag": None,
            "max_correlation": None,
            "is_significant": False,
            "direction": None,
            "lead_series": None,
        }

    # Find max absolute correlation
    abs_corr = valid['correlation'].abs()
    idx_max = abs_corr.idxmax()
    row = valid.loc[idx_max]

    opt_lag = int(row['lag'])
    max_corr = float(row['correlation'])
    is_significant = abs(max_corr) >= min_correlation

    # Determine direction and lead
    direction = "positive" if max_corr > 0 else "negative"
    if opt_lag > 0:
        lead_series = "series1"
    elif opt_lag < 0:
        lead_series = "series2"
    else:
        lead_series = "contemporaneous"

    return {
        "optimal_lag": opt_lag,
        "max_correlation": max_corr,
        "is_significant": is_significant,
        "direction": direction,
        "lead_series": lead_series,
    }


# =============================================================================
# GRANGER CAUSALITY TESTING
# =============================================================================

def granger_test(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    max_lag: int = 4,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """
    Test Granger causality from x to y.

    Tests whether past values of x improve prediction of y beyond
    past values of y alone. Does NOT imply true causation.

    Important: Both series should be stationary for valid results.

    Args:
        data: DataFrame containing both time series
        x_col: Column name of potential causal variable
        y_col: Column name of response variable
        max_lag: Maximum lag to test (default: 4)
        alpha: Significance level (default: 0.05)

    Returns:
        Dict containing:
        - x_col: Name of tested causal variable
        - y_col: Name of response variable
        - lags_tested: List of lags tested (1 to max_lag)
        - pvalues_by_lag: Dict mapping lag to p-value (F-test)
        - min_pvalue: Minimum p-value across all lags
        - optimal_lag: Lag with lowest p-value
        - granger_causes: Whether x Granger-causes y at any lag
        - interpretation: Human-readable result

    Raises:
        ImportError: If statsmodels is not installed
        ValueError: If columns not found or insufficient data
    """
    if not _STATSMODELS_AVAILABLE:
        raise ImportError(
            "statsmodels is required for Granger causality testing. "
            "Install with: pip install statsmodels"
        )

    # Validate columns
    if x_col not in data.columns:
        raise ValueError(f"Column '{x_col}' not found in data")
    if y_col not in data.columns:
        raise ValueError(f"Column '{y_col}' not found in data")

    # Prepare data
    test_data = data[[y_col, x_col]].dropna()

    min_required = max_lag * 2 + MIN_OBSERVATIONS
    if len(test_data) < min_required:
        return {
            "x_col": x_col,
            "y_col": y_col,
            "lags_tested": [],
            "pvalues_by_lag": {},
            "min_pvalue": None,
            "optimal_lag": None,
            "granger_causes": False,
            "interpretation": f"Insufficient data. Need {min_required}, have {len(test_data)}",
        }

    # Run Granger test (suppress verbose output)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            gc_results = grangercausalitytests(
                test_data, maxlag=max_lag, verbose=False
            )
        except Exception as e:
            return {
                "x_col": x_col,
                "y_col": y_col,
                "lags_tested": [],
                "pvalues_by_lag": {},
                "min_pvalue": None,
                "optimal_lag": None,
                "granger_causes": False,
                "interpretation": f"Test failed: {str(e)}",
            }

    # Extract p-values (use F-test)
    pvalues = {}
    for lag in range(1, max_lag + 1):
        if lag in gc_results:
            # gc_results[lag] is a tuple: (test_results_dict, ols_results_list)
            test_dict = gc_results[lag][0]
            # Use the F-test p-value
            pvalues[lag] = float(test_dict['ssr_ftest'][1])

    if not pvalues:
        return {
            "x_col": x_col,
            "y_col": y_col,
            "lags_tested": [],
            "pvalues_by_lag": {},
            "min_pvalue": None,
            "optimal_lag": None,
            "granger_causes": False,
            "interpretation": "No valid test results",
        }

    # Find minimum p-value
    min_pvalue = min(pvalues.values())
    optimal_lag = min(pvalues, key=pvalues.get)
    granger_causes = min_pvalue < alpha

    # Interpretation
    if granger_causes:
        interpretation = (
            f"'{x_col}' Granger-causes '{y_col}' at lag {optimal_lag} "
            f"(p={min_pvalue:.4f} < {alpha}). "
            "Past values of x improve prediction of y."
        )
    else:
        interpretation = (
            f"'{x_col}' does NOT Granger-cause '{y_col}' "
            f"(min p={min_pvalue:.4f} >= {alpha}). "
            "Past values of x do not significantly improve prediction of y."
        )

    return {
        "x_col": x_col,
        "y_col": y_col,
        "lags_tested": list(range(1, max_lag + 1)),
        "pvalues_by_lag": pvalues,
        "min_pvalue": min_pvalue,
        "optimal_lag": optimal_lag,
        "granger_causes": granger_causes,
        "interpretation": interpretation,
    }


def bidirectional_granger_test(
    data: pd.DataFrame,
    col1: str,
    col2: str,
    max_lag: int = 4,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """
    Test Granger causality in both directions.

    Args:
        data: DataFrame containing both time series
        col1: First column name
        col2: Second column name
        max_lag: Maximum lag to test
        alpha: Significance level

    Returns:
        Dict containing:
        - col1_to_col2: Granger test results (col1 -> col2)
        - col2_to_col1: Granger test results (col2 -> col1)
        - relationship: Interpretation of causal direction
    """
    result_1_to_2 = granger_test(data, col1, col2, max_lag, alpha)
    result_2_to_1 = granger_test(data, col2, col1, max_lag, alpha)

    causes_1_to_2 = result_1_to_2.get("granger_causes", False)
    causes_2_to_1 = result_2_to_1.get("granger_causes", False)

    # Interpret relationship
    if causes_1_to_2 and causes_2_to_1:
        relationship = "bidirectional"
    elif causes_1_to_2:
        relationship = f"{col1}_causes_{col2}"
    elif causes_2_to_1:
        relationship = f"{col2}_causes_{col1}"
    else:
        relationship = "no_granger_causality"

    return {
        "col1_to_col2": result_1_to_2,
        "col2_to_col1": result_2_to_1,
        "relationship": relationship,
    }


# =============================================================================
# VAR MODEL (OPTIONAL)
# =============================================================================

def fit_var_model(
    data: pd.DataFrame,
    columns: list[str],
    max_lag: int = 4,
) -> dict:
    """
    Fit a Vector Autoregression (VAR) model.

    VAR models capture interdependencies between multiple time series
    and allow analysis of shock propagation via impulse response functions.

    Args:
        data: DataFrame with time series columns
        columns: List of column names to include in VAR
        max_lag: Maximum lag order to consider (default: 4)

    Returns:
        Dict containing:
        - fitted: Whether model was successfully fit
        - optimal_lag: Selected lag order (by AIC)
        - aic: AIC at optimal lag
        - bic: BIC at optimal lag
        - coefficients: Dict of coefficient matrices by lag
        - summary_stats: Model summary statistics
        - error: Error message if fitting failed

    Raises:
        ImportError: If statsmodels is not installed
    """
    if not _STATSMODELS_AVAILABLE:
        raise ImportError(
            "statsmodels is required for VAR model fitting. "
            "Install with: pip install statsmodels"
        )

    # Validate columns
    missing = [c for c in columns if c not in data.columns]
    if missing:
        return {
            "fitted": False,
            "optimal_lag": None,
            "aic": None,
            "bic": None,
            "coefficients": {},
            "summary_stats": {},
            "error": f"Columns not found: {missing}",
        }

    # Prepare data
    var_data = data[columns].dropna()

    min_required = max_lag * len(columns) + MIN_OBSERVATIONS
    if len(var_data) < min_required:
        return {
            "fitted": False,
            "optimal_lag": None,
            "aic": None,
            "bic": None,
            "coefficients": {},
            "summary_stats": {},
            "error": f"Insufficient data. Need {min_required}, have {len(var_data)}",
        }

    try:
        # Fit VAR model
        model = VAR(var_data)

        # Select optimal lag by AIC
        lag_order_result = model.select_order(maxlags=max_lag)
        optimal_lag = lag_order_result.aic

        # Fit with optimal lag
        fitted_model = model.fit(optimal_lag)

        # Extract coefficients
        coefficients = {}
        for lag in range(1, optimal_lag + 1):
            coef_matrix = fitted_model.coefs[lag - 1]
            coefficients[f"lag_{lag}"] = {
                columns[i]: {
                    columns[j]: float(coef_matrix[i, j])
                    for j in range(len(columns))
                }
                for i in range(len(columns))
            }

        return {
            "fitted": True,
            "optimal_lag": optimal_lag,
            "aic": float(fitted_model.aic),
            "bic": float(fitted_model.bic),
            "coefficients": coefficients,
            "summary_stats": {
                "n_observations": len(var_data),
                "n_variables": len(columns),
                "variables": columns,
            },
            "error": None,
        }

    except Exception as e:
        return {
            "fitted": False,
            "optimal_lag": None,
            "aic": None,
            "bic": None,
            "coefficients": {},
            "summary_stats": {},
            "error": str(e),
        }


# =============================================================================
# COMPREHENSIVE ANALYSIS
# =============================================================================

def analyze_cpu_vc_correlation(
    cpu_series: pd.Series,
    vc_series: pd.Series,
    cpu_name: str = "cpu_index",
    vc_name: str = "vc_deal_count",
    max_ccf_lag: int = DEFAULT_MAX_LAG,
    max_granger_lag: int = 4,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """
    Comprehensive CPU-VC correlation analysis.

    Performs a complete time series analysis pipeline:
    1. Stationarity testing and transformation
    2. Cross-correlation at multiple lags
    3. Granger causality testing
    4. Summary statistics

    Args:
        cpu_series: CPU index time series (pandas Series with datetime index)
        vc_series: VC metric time series (pandas Series with datetime index)
        cpu_name: Name for CPU series in output (default: "cpu_index")
        vc_name: Name for VC series in output (default: "vc_deal_count")
        max_ccf_lag: Maximum lag for cross-correlation (default: 12)
        max_granger_lag: Maximum lag for Granger test (default: 4)
        alpha: Significance level (default: 0.05)

    Returns:
        Dict containing:
        - status: "success" or "error"
        - metadata: Analysis metadata (n_observations, date_range, etc.)
        - stationarity: Stationarity test results for both series
        - cross_correlation: CCF results and optimal lag
        - granger_causality: Bidirectional Granger test results
        - summary: Key findings summary
        - warnings: List of any warnings or data issues

    Notes:
        - Series are aligned by index before analysis
        - Granger tests are run on differenced data if needed
        - All p-values use the specified alpha for significance
    """
    warnings_list = []
    results = {
        "status": "success",
        "metadata": {},
        "stationarity": {},
        "cross_correlation": {},
        "granger_causality": {},
        "summary": {},
        "warnings": warnings_list,
    }

    # Align series
    combined = pd.DataFrame({
        cpu_name: cpu_series,
        vc_name: vc_series,
    }).dropna()

    n_obs = len(combined)

    # Metadata
    results["metadata"] = {
        "n_observations": n_obs,
        "date_range": {
            "start": str(combined.index.min()) if n_obs > 0 else None,
            "end": str(combined.index.max()) if n_obs > 0 else None,
        },
        "cpu_name": cpu_name,
        "vc_name": vc_name,
        "alpha": alpha,
        "statsmodels_available": _STATSMODELS_AVAILABLE,
    }

    if n_obs < MIN_OBSERVATIONS:
        results["status"] = "error"
        results["summary"]["error"] = (
            f"Insufficient data. Have {n_obs} observations, "
            f"need at least {MIN_OBSERVATIONS}."
        )
        return results

    # ==========================================================================
    # 1. Stationarity Testing
    # ==========================================================================
    if _STATSMODELS_AVAILABLE:
        try:
            cpu_stationarity = check_stationarity(combined[cpu_name], cpu_name, alpha)
            results["stationarity"][cpu_name] = cpu_stationarity

            if cpu_stationarity["differencing_needed"]:
                warnings_list.append(
                    f"{cpu_name} appears non-stationary. Consider differencing."
                )
        except Exception as e:
            results["stationarity"][cpu_name] = {"error": str(e)}

        try:
            vc_stationarity = check_stationarity(combined[vc_name], vc_name, alpha)
            results["stationarity"][vc_name] = vc_stationarity

            if vc_stationarity["differencing_needed"]:
                warnings_list.append(
                    f"{vc_name} appears non-stationary. Consider differencing."
                )
        except Exception as e:
            results["stationarity"][vc_name] = {"error": str(e)}
    else:
        warnings_list.append(
            "statsmodels not installed. Stationarity tests skipped."
        )

    # ==========================================================================
    # 2. Cross-Correlation Analysis
    # ==========================================================================
    ccf_df = cross_correlation(
        combined[cpu_name],
        combined[vc_name],
        max_lag=max_ccf_lag,
    )

    optimal = find_optimal_lag(ccf_df, min_correlation=0.1)

    results["cross_correlation"] = {
        "results": ccf_df.to_dict(orient='records'),
        "optimal_lag": optimal,
        "contemporaneous_correlation": float(
            ccf_df[ccf_df['lag'] == 0]['correlation'].iloc[0]
        ) if len(ccf_df) > 0 else None,
    }

    # ==========================================================================
    # 3. Granger Causality Testing
    # ==========================================================================
    if _STATSMODELS_AVAILABLE:
        # Check if differencing is needed for Granger
        cpu_diff_needed = results["stationarity"].get(cpu_name, {}).get(
            "differencing_needed", False
        )
        vc_diff_needed = results["stationarity"].get(vc_name, {}).get(
            "differencing_needed", False
        )

        if cpu_diff_needed or vc_diff_needed:
            # Use differenced data for Granger test
            granger_data = combined.diff().dropna()
            granger_note = "Granger tests performed on first-differenced data."
            warnings_list.append(granger_note)
        else:
            granger_data = combined
            granger_note = "Granger tests performed on level data."

        if len(granger_data) >= max_granger_lag * 2 + MIN_OBSERVATIONS:
            granger_results = bidirectional_granger_test(
                granger_data, cpu_name, vc_name, max_granger_lag, alpha
            )
            granger_results["data_transformation"] = granger_note
            results["granger_causality"] = granger_results
        else:
            results["granger_causality"] = {
                "error": "Insufficient data for Granger causality test",
                "data_transformation": granger_note,
            }
    else:
        results["granger_causality"] = {
            "error": "statsmodels not installed"
        }

    # ==========================================================================
    # 4. Summary
    # ==========================================================================
    summary = {
        "n_observations": n_obs,
        "contemporaneous_correlation": results["cross_correlation"].get(
            "contemporaneous_correlation"
        ),
    }

    # Optimal lag summary
    opt_lag = optimal.get("optimal_lag")
    if opt_lag is not None:
        summary["optimal_lag"] = opt_lag
        summary["max_correlation"] = optimal.get("max_correlation")
        summary["correlation_direction"] = optimal.get("direction")
        summary["lead_lag_relationship"] = optimal.get("lead_series")

    # Granger summary
    gc = results.get("granger_causality", {})
    if "relationship" in gc:
        summary["granger_relationship"] = gc["relationship"]

        # Check for expected negative relationship (CPU -> VC)
        cpu_to_vc = gc.get("col1_to_col2", {})
        if cpu_to_vc.get("granger_causes"):
            summary["cpu_granger_causes_vc"] = True
            summary["cpu_to_vc_pvalue"] = cpu_to_vc.get("min_pvalue")
        else:
            summary["cpu_granger_causes_vc"] = False

        vc_to_cpu = gc.get("col2_to_col1", {})
        if vc_to_cpu.get("granger_causes"):
            summary["vc_granger_causes_cpu"] = True
            summary["vc_to_cpu_pvalue"] = vc_to_cpu.get("min_pvalue")
        else:
            summary["vc_granger_causes_cpu"] = False

    results["summary"] = summary

    return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_statsmodels_available() -> bool:
    """Check if statsmodels is installed and available."""
    return _STATSMODELS_AVAILABLE


def get_required_observations(max_lag: int = 4) -> int:
    """
    Get minimum observations required for analysis.

    Args:
        max_lag: Maximum lag to be used in tests

    Returns:
        Minimum number of observations needed
    """
    return max_lag * 2 + MIN_OBSERVATIONS


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("=== CPU-VC Correlation Analysis Module ===")
    print(f"statsmodels available: {_STATSMODELS_AVAILABLE}")
    print(f"Minimum observations required: {MIN_OBSERVATIONS}")

    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range("2020-01", periods=48, freq="MS")

    # Simulate CPU (random walk with drift)
    cpu = 100 + np.cumsum(np.random.randn(48) * 5)

    # Simulate VC (negatively correlated with lag to CPU)
    vc = 50 + np.random.randn(48) * 10 - 0.3 * np.roll(cpu, 2)

    cpu_series = pd.Series(cpu, index=dates, name="cpu_index")
    vc_series = pd.Series(vc, index=dates, name="vc_deal_count")

    print("\nSample data created:")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    print(f"  CPU mean: {cpu_series.mean():.1f}, std: {cpu_series.std():.1f}")
    print(f"  VC mean: {vc_series.mean():.1f}, std: {vc_series.std():.1f}")

    # Run cross-correlation
    ccf = cross_correlation(cpu_series, vc_series, max_lag=6)
    print("\nCross-correlation results:")
    print(ccf[['lag', 'correlation', 'n_observations']].to_string(index=False))

    opt = find_optimal_lag(ccf)
    print(f"\nOptimal lag: {opt['optimal_lag']}, correlation: {opt['max_correlation']:.3f}")

    if _STATSMODELS_AVAILABLE:
        print("\n--- Stationarity Tests ---")
        try:
            stat = check_stationarity(cpu_series, "cpu_index")
            print(f"CPU: {stat['conclusion']} (ADF p={stat['adf_pvalue']:.4f})")
        except Exception as e:
            print(f"CPU stationarity test error: {e}")

        print("\n--- Full Analysis ---")
        results = analyze_cpu_vc_correlation(cpu_series, vc_series)
        print(f"Status: {results['status']}")
        print(f"Summary: {results['summary']}")
    else:
        print("\nInstall statsmodels for full analysis: pip install statsmodels")
