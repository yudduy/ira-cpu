"""Index calculation and analysis for CPU Index."""

from .indexer import build_index, get_index_summary, get_full_index_data
from .normalizer import normalize_bbd_style, compute_outlet_level_cpu
from .ablation_config import get_all_ablation_variants
from .ablation_runner import run_all_ablations
from .correlation import (
    analyze_cpu_vc_correlation,
    cross_correlation,
    find_optimal_lag,
    check_stationarity,
    make_stationary,
    granger_test,
    bidirectional_granger_test,
    fit_var_model,
    check_statsmodels_available,
)
from .vc_visualizations import (
    plot_cpu_vc_timeseries,
    plot_cross_correlation,
    plot_rolling_correlation,
    plot_stage_distribution,
    save_all_visualizations,
)
