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
from .sector_analysis import (
    analyze_sector_cpu_correlation,
    analyze_all_sectors,
    analyze_cpu_decomposition,
    stratify_by_ira_exposure,
    run_classifier_robustness,
)
from .vc_aggregator import (
    aggregate_monthly,
    aggregate_by_category_monthly,
    aggregate_by_subtopic_monthly,
    create_analysis_dataset,
)
from .sector_visualizations import (
    plot_sector_correlation_heatmap,
    plot_sector_timeseries,
    plot_sensitivity_ranking,
    plot_decomposition_comparison,
    save_sector_visualizations,
)
