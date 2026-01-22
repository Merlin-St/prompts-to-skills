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

# SOC Major Groups mapping (first 2 digits of SOC code -> group name)
SOC_MAJOR_GROUPS = {
    '11': 'Management',
    '13': 'Business and Financial Operations',
    '15': 'Computer and Mathematical',
    '17': 'Architecture and Engineering',
    '19': 'Life, Physical, and Social Science',
    '21': 'Community and Social Service',
    '23': 'Legal',
    '25': 'Educational Instruction and Library',
    '27': 'Arts, Design, Entertainment, Sports, and Media',
    '29': 'Healthcare Practitioners and Technical',
    '31': 'Healthcare Support',
    '33': 'Protective Service',
    '35': 'Food Preparation and Serving Related',
    '37': 'Building and Grounds Cleaning and Maintenance',
    '39': 'Personal Care and Service',
    '41': 'Sales and Related',
    '43': 'Office and Administrative Support',
    '45': 'Farming, Fishing, and Forestry',
    '47': 'Construction and Extraction',
    '49': 'Installation, Maintenance, and Repair',
    '51': 'Production',
    '53': 'Transportation and Material Moving',
}

# Color palette for SOC Major Groups (top 5 + Other)
# Colors will be assigned dynamically based on frequency
CATEGORY_PALETTE = [
    '#4E79A7',  # Blue
    '#F28E2B',  # Orange
    '#59A14F',  # Green
    '#E15759',  # Red
    '#B07AA1',  # Purple
]
OTHER_COLOR = '#BAB0AC'  # Gray for "Other"

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


def load_onet_task_mapping() -> dict:
    """Load O*NET task statements and create task -> SOC major group mapping.

    Returns a dict mapping lowercased task descriptions to their SOC major group name.
    """
    df = pd.read_csv(DATA_DIR / 'cl_onet_taskstatements.csv')

    mapping = {}
    for _, row in df.iterrows():
        task_lower = row['Task'].lower().strip()
        soc_code = str(row['O*NET-SOC Code'])
        major_group_code = soc_code[:2]  # First 2 digits
        major_group_name = SOC_MAJOR_GROUPS.get(major_group_code, 'Other')
        mapping[task_lower] = major_group_name

    return mapping


def extract_onet_task_data(df: pd.DataFrame, task_mapping: dict = None) -> pd.DataFrame:
    """Extract O*NET task level data: success rate, duration, stdev, weight.

    Args:
        df: Raw data DataFrame
        task_mapping: Optional dict mapping task names to SOC major groups
    """

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

    # Add SOC major group column if mapping provided
    if task_mapping is not None:
        result['major_group'] = result['task_name'].str.lower().str.strip().map(task_mapping)
        result['major_group'] = result['major_group'].fillna('Other')

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


def create_onet_subplot(ax, data: pd.DataFrame, data_type: str, top_categories: list, category_colors: dict):
    """Create O*NET subplot with points colored by SOC major group.

    Args:
        ax: Matplotlib axis
        data: DataFrame with task data including 'major_group' column
        data_type: Label for the data type (e.g., 'O*NET Tasks')
        top_categories: List of top N category names to highlight
        category_colors: Dict mapping category names to colors
    """
    x = data['duration_mean'].values
    y = data['success_pct'].values
    weights = data['weight_pct'].values

    # Get standard deviations for error bars
    y_err = data['success_se'].fillna(0).values

    # Fit WLS regression on all data
    slope, intercept, predict, results = fit_wls(x, y, weights)

    # Assign display category (top categories keep their name, others become "Other")
    data = data.copy()
    data['display_category'] = data['major_group'].apply(
        lambda x: x if x in top_categories else 'Other'
    )

    # Plot scatter points by category (Other first so it's in the background)
    categories_to_plot = ['Other'] + top_categories
    for category in categories_to_plot:
        subset = data[data['display_category'] == category]
        if len(subset) > 0:
            cat_x = subset['duration_mean'].values
            cat_y = subset['success_pct'].values
            cat_weights = subset['weight_pct'].values
            sizes = 50 + 500 * (cat_weights / weights.max())
            color = category_colors.get(category, OTHER_COLOR)
            ax.scatter(cat_x, cat_y, s=sizes, c=color, alpha=0.6,
                      edgecolors='white', linewidth=0.5, label=category, zorder=2)

    # Add error bars in gray
    ax.errorbar(x, y, yerr=y_err, fmt='none', ecolor='gray', alpha=0.2, capsize=2, capthick=1, zorder=1)

    # Plot regression line in BLACK at the FRONT
    x_line = np.linspace(x.min() - 0.5, x.max() + 0.5, 100)
    y_line = predict(x_line)
    ax.plot(x_line, y_line, '-', color='black', linewidth=2.5, zorder=100)

    # Add reference line at 50% success
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, linewidth=1, zorder=0)

    # Labels and title
    n_points = len(data)
    ax.set_title(f'{data_type}\n(n={n_points})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Duration (Human-Only Time, hours)', fontsize=10)
    ax.set_ylabel('Task Success (%)', fontsize=10)

    # Set axis limits
    ax.set_xlim(0, min(x.max() + 1, 12))
    ax.set_ylim(max(y.min() - 10, 0), min(y.max() + 10, 100))

    # Add grid
    ax.grid(True, alpha=0.3, zorder=0)

    # Add regression info
    r_squared = results.rsquared
    ax.text(0.95, 0.95, f'WLS R² = {r_squared:.3f}\nslope = {slope:.2f}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return n_points


def create_subplot(ax, data: pd.DataFrame, data_type: str, color: str):
    """Create a single subplot with scatter, WLS regression, and error bars.

    Used for Request Categories panels (single color).
    """
    x = data['duration_mean'].values
    y = data['success_pct'].values
    weights = data['weight_pct'].values

    # Get standard deviations for error bars
    y_err = data['success_se'].fillna(0).values

    # Fit WLS regression
    slope, intercept, predict, results = fit_wls(x, y, weights)

    # Create scatter plot with size proportional to weight
    sizes = 50 + 500 * (weights / weights.max())  # Scale sizes for visibility
    ax.scatter(x, y, s=sizes, c=color, alpha=0.6, edgecolors='white', linewidth=0.5, zorder=2)

    # Add error bars
    ax.errorbar(x, y, yerr=y_err, fmt='none', ecolor=color, alpha=0.3, capsize=2, capthick=1, zorder=1)

    # Plot regression line in BLACK at the FRONT
    x_line = np.linspace(x.min() - 0.5, x.max() + 0.5, 100)
    y_line = predict(x_line)
    ax.plot(x_line, y_line, '-', color='black', linewidth=2.5, zorder=100)

    # Add reference line at 50% success
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, linewidth=1, zorder=0)

    # Labels and title
    n_points = len(data)
    ax.set_title(f'{data_type}\n(n={n_points})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Duration (Human-Only Time, hours)', fontsize=10)
    ax.set_ylabel('Task Success (%)', fontsize=10)

    # Set axis limits
    ax.set_xlim(0, min(x.max() + 1, 12))
    ax.set_ylim(max(y.min() - 10, 0), min(y.max() + 10, 100))

    # Add grid
    ax.grid(True, alpha=0.3, zorder=0)

    # Add regression info
    r_squared = results.rsquared
    ax.text(0.95, 0.95, f'WLS R² = {r_squared:.3f}\nslope = {slope:.2f}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return n_points


def main():
    print("Loading data...")

    # Load O*NET task mapping
    print("Loading O*NET task-to-SOC mapping...")
    task_mapping = load_onet_task_mapping()
    print(f"  Loaded {len(task_mapping)} task mappings")

    # Load data for both platforms
    df_1p = load_data('1P API')
    df_claude = load_data('Claude.ai')

    print(f"1P API data: {len(df_1p)} rows")
    print(f"Claude.ai data: {len(df_claude)} rows")

    # Extract O*NET task data with SOC mapping
    print("\nExtracting O*NET task data...")
    onet_1p = extract_onet_task_data(df_1p, task_mapping)
    onet_claude = extract_onet_task_data(df_claude, task_mapping)
    print(f"  1P API O*NET tasks: {len(onet_1p)}")
    print(f"  Claude.ai O*NET tasks: {len(onet_claude)}")

    # Determine top 5 SOC major groups across both datasets
    combined_groups = pd.concat([onet_1p['major_group'], onet_claude['major_group']])
    group_counts = combined_groups.value_counts()
    top_categories = group_counts.head(5).index.tolist()
    print(f"\nTop 5 SOC Major Groups (by task count):")
    for i, cat in enumerate(top_categories, 1):
        print(f"  {i}. {cat}: {group_counts[cat]} tasks")

    # Build category colors dict
    category_colors = {cat: CATEGORY_PALETTE[i] for i, cat in enumerate(top_categories)}
    category_colors['Other'] = OTHER_COLOR

    # Extract Request data
    print("\nExtracting Request category data...")
    request_1p = extract_request_data(df_1p)
    request_claude = extract_request_data(df_claude)
    print(f"  1P API Requests: {len(request_1p)}")
    print(f"  Claude.ai Requests: {len(request_claude)}")

    # Create figure
    print("\nCreating figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Add big column titles for 1P API and Claude.ai
    fig.text(0.27, 0.97, '1P API', fontsize=16, fontweight='bold', ha='center', va='bottom')
    fig.text(0.73, 0.97, 'Claude.ai', fontsize=16, fontweight='bold', ha='center', va='bottom')

    # Main figure title
    fig.suptitle('Task Horizons in Real-World Usage\nSuccess vs. Task Duration by Category Type',
                 fontsize=14, fontweight='bold', y=1.02)

    # 1P API - O*NET Tasks (top-left)
    create_onet_subplot(axes[0, 0], onet_1p, 'O*NET Tasks', top_categories, category_colors)

    # Claude.ai - O*NET Tasks (top-right)
    create_onet_subplot(axes[0, 1], onet_claude, 'O*NET Tasks', top_categories, category_colors)

    # 1P API - Request Categories (bottom-left) - neutral gray color
    create_subplot(axes[1, 0], request_1p, 'Request Categories', '#7f7f7f')

    # Claude.ai - Request Categories (bottom-right) - neutral gray color
    create_subplot(axes[1, 1], request_claude, 'Request Categories', '#7f7f7f')

    # Add legend for SOC major group colors
    legend_elements = [Patch(facecolor=category_colors[cat], edgecolor='white', label=cat)
                       for cat in top_categories]
    legend_elements.append(Patch(facecolor=OTHER_COLOR, edgecolor='white', label='Other'))
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=9, title='SOC Major Group')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

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

    # Print category breakdown for O*NET panels
    print("\n" + "="*60)
    print("SOC Major Group Breakdown")
    print("="*60)
    for name, data in [('1P API O*NET', onet_1p), ('Claude.ai O*NET', onet_claude)]:
        print(f"\n{name}:")
        group_counts = data['major_group'].value_counts()
        for group, count in group_counts.items():
            pct = 100 * count / len(data)
            print(f"  {group}: {count} ({pct:.1f}%)")


if __name__ == '__main__':
    main()
