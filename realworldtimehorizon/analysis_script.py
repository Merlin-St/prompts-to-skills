#!/usr/bin/env python3
"""
Real-World Task Horizon Analysis
Creates a 4-panel figure showing task success vs. duration for:
- 1P API + O*NET tasks
- 1P API + Request categories
- Claude.ai + O*NET tasks
- Claude.ai + Request categories

Each panel includes:
- Scatter plot with points sized by share percentage
- WLS regression line weighted by share percentage
- Standard deviation error bars
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import statsmodels.api as sm
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR

# Color scheme matching the original figure
COLORS = {
    '1P API': '#5B8FB9',  # Blue
    'Claude.ai': '#E57373'  # Coral/Orange
}

def load_data(platform: str) -> pd.DataFrame:
    """Load data for a specific platform."""
    if platform == '1P API':
        filepath = DATA_DIR / 'aei_raw_1p_api_2025-11-13_to_2025-11-20.csv'
    else:
        filepath = DATA_DIR / 'aei_raw_claude_ai_2025-11-13_to_2025-11-20.csv'

    df = pd.read_csv(filepath)
    # For Claude.ai, filter to GLOBAL level to match 1P API
    df = df[df['geo_id'] == 'GLOBAL'].copy()
    return df


def extract_onet_task_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract O*NET task level data: success rate, duration, stdev, weight."""

    # 1. Get task success rates (filter for ::yes suffix)
    success_df = df[
        (df['facet'] == 'onet_task::task_success') &
        (df['variable'] == 'onet_task_task_success_pct') &
        (df['cluster_name'].str.endswith('::yes'))
    ].copy()
    success_df['task_name'] = success_df['cluster_name'].str.replace('::yes$', '', regex=True)
    success_df = success_df[['task_name', 'value']].rename(columns={'value': 'success_pct'})

    # 2. Get human_only_time mean
    duration_mean_df = df[
        (df['facet'] == 'onet_task::human_only_time') &
        (df['variable'] == 'onet_task_human_only_time_mean')
    ].copy()
    duration_mean_df = duration_mean_df[['cluster_name', 'value']].rename(
        columns={'cluster_name': 'task_name', 'value': 'duration_mean'}
    )

    # 3. Get human_only_time stdev
    duration_stdev_df = df[
        (df['facet'] == 'onet_task::human_only_time') &
        (df['variable'] == 'onet_task_human_only_time_stdev')
    ].copy()
    duration_stdev_df = duration_stdev_df[['cluster_name', 'value']].rename(
        columns={'cluster_name': 'task_name', 'value': 'duration_stdev'}
    )

    # 4. Get task share percentage (weight)
    weight_df = df[
        (df['facet'] == 'onet_task') &
        (df['variable'] == 'onet_task_pct')
    ].copy()
    weight_df = weight_df[['cluster_name', 'value']].rename(
        columns={'cluster_name': 'task_name', 'value': 'weight_pct'}
    )

    # 5. Get success stdev by computing from yes/no counts
    # We'll use the count data to compute standard error if available
    success_count_yes = df[
        (df['facet'] == 'onet_task::task_success') &
        (df['variable'] == 'onet_task_task_success_count') &
        (df['cluster_name'].str.endswith('::yes'))
    ].copy()
    success_count_yes['task_name'] = success_count_yes['cluster_name'].str.replace('::yes$', '', regex=True)
    success_count_yes = success_count_yes[['task_name', 'value']].rename(columns={'value': 'count_yes'})

    success_count_no = df[
        (df['facet'] == 'onet_task::task_success') &
        (df['variable'] == 'onet_task_task_success_count') &
        (df['cluster_name'].str.endswith('::no'))
    ].copy()
    success_count_no['task_name'] = success_count_no['cluster_name'].str.replace('::no$', '', regex=True)
    success_count_no = success_count_no[['task_name', 'value']].rename(columns={'value': 'count_no'})

    # Merge all data
    result = success_df.merge(duration_mean_df, on='task_name', how='inner')
    result = result.merge(duration_stdev_df, on='task_name', how='left')
    result = result.merge(weight_df, on='task_name', how='left')
    result = result.merge(success_count_yes, on='task_name', how='left')
    result = result.merge(success_count_no, on='task_name', how='left')

    # Calculate total count and success SE (standard error for binomial proportion)
    result['total_count'] = result['count_yes'].fillna(0) + result['count_no'].fillna(0)
    result['success_se'] = np.sqrt(
        (result['success_pct'] / 100) * (1 - result['success_pct'] / 100) / result['total_count'].clip(lower=1)
    ) * 100  # Convert back to percentage scale

    # Remove not_classified and rows with missing data
    result = result[~result['task_name'].str.contains('not_classified', case=False, na=False)]
    result = result.dropna(subset=['success_pct', 'duration_mean', 'weight_pct'])

    # Filter out extreme outliers (duration > 24 hours is likely noise)
    result = result[result['duration_mean'] <= 24]

    return result


def extract_request_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract Request category level data: success rate, duration, stdev, weight."""

    # 1. Get request success rates (filter for ::yes suffix)
    success_df = df[
        (df['facet'] == 'request::task_success') &
        (df['variable'] == 'request_task_success_pct') &
        (df['cluster_name'].str.endswith('::yes'))
    ].copy()
    success_df['request_name'] = success_df['cluster_name'].str.replace('::yes$', '', regex=True)
    success_df = success_df[['request_name', 'value']].rename(columns={'value': 'success_pct'})

    # 2. Get human_only_time mean
    duration_mean_df = df[
        (df['facet'] == 'request::human_only_time') &
        (df['variable'] == 'request_human_only_time_mean')
    ].copy()
    duration_mean_df = duration_mean_df[['cluster_name', 'value']].rename(
        columns={'cluster_name': 'request_name', 'value': 'duration_mean'}
    )

    # 3. Get human_only_time stdev
    duration_stdev_df = df[
        (df['facet'] == 'request::human_only_time') &
        (df['variable'] == 'request_human_only_time_stdev')
    ].copy()
    duration_stdev_df = duration_stdev_df[['cluster_name', 'value']].rename(
        columns={'cluster_name': 'request_name', 'value': 'duration_stdev'}
    )

    # 4. Get request share percentage (weight)
    weight_df = df[
        (df['facet'] == 'request') &
        (df['variable'] == 'request_pct')
    ].copy()
    weight_df = weight_df[['cluster_name', 'value']].rename(
        columns={'cluster_name': 'request_name', 'value': 'weight_pct'}
    )

    # 5. Get success counts for SE calculation
    success_count_yes = df[
        (df['facet'] == 'request::task_success') &
        (df['variable'] == 'request_task_success_count') &
        (df['cluster_name'].str.endswith('::yes'))
    ].copy()
    success_count_yes['request_name'] = success_count_yes['cluster_name'].str.replace('::yes$', '', regex=True)
    success_count_yes = success_count_yes[['request_name', 'value']].rename(columns={'value': 'count_yes'})

    success_count_no = df[
        (df['facet'] == 'request::task_success') &
        (df['variable'] == 'request_task_success_count') &
        (df['cluster_name'].str.endswith('::no'))
    ].copy()
    success_count_no['request_name'] = success_count_no['cluster_name'].str.replace('::no$', '', regex=True)
    success_count_no = success_count_no[['request_name', 'value']].rename(columns={'value': 'count_no'})

    # Merge all data
    result = success_df.merge(duration_mean_df, on='request_name', how='inner')
    result = result.merge(duration_stdev_df, on='request_name', how='left')
    result = result.merge(weight_df, on='request_name', how='left')
    result = result.merge(success_count_yes, on='request_name', how='left')
    result = result.merge(success_count_no, on='request_name', how='left')

    # Calculate total count and success SE
    result['total_count'] = result['count_yes'].fillna(0) + result['count_no'].fillna(0)
    result['success_se'] = np.sqrt(
        (result['success_pct'] / 100) * (1 - result['success_pct'] / 100) / result['total_count'].clip(lower=1)
    ) * 100

    # Remove not_classified and rows with missing data
    result = result[~result['request_name'].str.contains('not_classified', case=False, na=False)]
    result = result.dropna(subset=['success_pct', 'duration_mean', 'weight_pct'])

    # Filter out extreme outliers
    result = result[result['duration_mean'] <= 24]

    return result


def fit_wls(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple:
    """Fit WLS regression and return slope, intercept, and prediction function."""
    X = sm.add_constant(x)

    # Ensure weights are positive
    weights = np.clip(weights, 1e-10, None)

    model = sm.WLS(y, X, weights=weights)
    results = model.fit()

    intercept = results.params[0]
    slope = results.params[1]

    def predict(x_new):
        return intercept + slope * x_new

    return slope, intercept, predict, results


def create_subplot(ax, data: pd.DataFrame, platform: str, data_type: str, color: str):
    """Create a single subplot with scatter, WLS regression, and error bars."""

    x = data['duration_mean'].values
    y = data['success_pct'].values
    weights = data['weight_pct'].values

    # Get standard deviations for error bars
    # For y-axis: use success_se (standard error of success rate)
    y_err = data['success_se'].fillna(0).values

    # For x-axis: use duration_stdev scaled down (optional, can be overwhelming)
    # We'll show duration stdev as horizontal error bars but scaled by 1/sqrt(n) for SE
    x_err = (data['duration_stdev'].fillna(0) / np.sqrt(data['total_count'].clip(lower=1))).values

    # Fit WLS regression
    slope, intercept, predict, results = fit_wls(x, y, weights)

    # Create scatter plot with size proportional to weight
    sizes = 50 + 500 * (weights / weights.max())  # Scale sizes for visibility
    scatter = ax.scatter(x, y, s=sizes, c=color, alpha=0.6, edgecolors='white', linewidth=0.5)

    # Add error bars (y only for cleaner visualization)
    ax.errorbar(x, y, yerr=y_err, fmt='none', ecolor=color, alpha=0.3, capsize=2, capthick=1)

    # Plot regression line
    x_line = np.linspace(x.min() - 0.5, x.max() + 0.5, 100)
    y_line = predict(x_line)
    ax.plot(x_line, y_line, '--', color=color, linewidth=2, alpha=0.8)

    # Add reference line at 50% success
    ax.axhline(y=50, color='red', linestyle=':', alpha=0.5, linewidth=1)

    # Labels and title
    n_points = len(data)
    ax.set_title(f'{platform} - {data_type}\n(n={n_points})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Duration (Human-Only Time, hours)', fontsize=10)
    ax.set_ylabel('Task Success (%)', fontsize=10)

    # Set axis limits
    ax.set_xlim(0, min(x.max() + 1, 12))
    ax.set_ylim(max(y.min() - 10, 0), min(y.max() + 10, 100))

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add regression info
    r_squared = results.rsquared
    ax.text(0.95, 0.95, f'WLS R² = {r_squared:.3f}\nslope = {slope:.2f}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return n_points


def main():
    print("Loading data...")

    # Load data for both platforms
    df_1p = load_data('1P API')
    df_claude = load_data('Claude.ai')

    print(f"1P API data: {len(df_1p)} rows")
    print(f"Claude.ai data: {len(df_claude)} rows")

    # Extract O*NET task data
    print("\nExtracting O*NET task data...")
    onet_1p = extract_onet_task_data(df_1p)
    onet_claude = extract_onet_task_data(df_claude)
    print(f"  1P API O*NET tasks: {len(onet_1p)}")
    print(f"  Claude.ai O*NET tasks: {len(onet_claude)}")

    # Extract Request data
    print("\nExtracting Request category data...")
    request_1p = extract_request_data(df_1p)
    request_claude = extract_request_data(df_claude)
    print(f"  1P API Requests: {len(request_1p)}")
    print(f"  Claude.ai Requests: {len(request_claude)}")

    # Create figure
    print("\nCreating figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Task Horizons in Real-World Usage\nSuccess vs. Task Duration by Platform and Category Type',
                 fontsize=14, fontweight='bold', y=0.98)

    # 1P API - O*NET Tasks (top-left)
    create_subplot(axes[0, 0], onet_1p, '1P API', 'O*NET Tasks', COLORS['1P API'])

    # Claude.ai - O*NET Tasks (top-right)
    create_subplot(axes[0, 1], onet_claude, 'Claude.ai', 'O*NET Tasks', COLORS['Claude.ai'])

    # 1P API - Request Categories (bottom-left)
    create_subplot(axes[1, 0], request_1p, '1P API', 'Request Categories', COLORS['1P API'])

    # Claude.ai - Request Categories (bottom-right)
    create_subplot(axes[1, 1], request_claude, 'Claude.ai', 'Request Categories', COLORS['Claude.ai'])

    # Add legend
    legend_elements = [
        Patch(facecolor=COLORS['1P API'], edgecolor='white', label='1P API'),
        Patch(facecolor=COLORS['Claude.ai'], edgecolor='white', label='Claude.ai'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 0.02), fontsize=10)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save figures
    output_png = OUTPUT_DIR / 'task_horizon_4panel.png'
    output_pdf = OUTPUT_DIR / 'task_horizon_4panel.pdf'

    fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_pdf, bbox_inches='tight', facecolor='white')

    print(f"\nFigures saved to:")
    print(f"  {output_png}")
    print(f"  {output_pdf}")

    plt.close()

    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)

    for name, data in [('1P API O*NET', onet_1p), ('Claude.ai O*NET', onet_claude),
                       ('1P API Requests', request_1p), ('Claude.ai Requests', request_claude)]:
        print(f"\n{name}:")
        print(f"  N = {len(data)}")
        print(f"  Duration: mean={data['duration_mean'].mean():.2f}h, range=[{data['duration_mean'].min():.2f}, {data['duration_mean'].max():.2f}]")
        print(f"  Success: mean={data['success_pct'].mean():.1f}%, range=[{data['success_pct'].min():.1f}, {data['success_pct'].max():.1f}]")


if __name__ == '__main__':
    main()
