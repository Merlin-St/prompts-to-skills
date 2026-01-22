# Real-World Task Horizon Analysis Plan

## Objective
Recreate the "Task Horizons in Real-World Usage" figure (Success vs. task duration by platform) with the following improvements:
1. **Separate 1P API and Claude.ai into distinct panels** (instead of overlaying)
2. **Separate O*NET tasks and Request categories** into their own subplots
3. **Use WLS (Weighted Least Squares) regression** weighted by task/request share percentage
4. **Include standard deviations as error bars** on each data point

## Data Sources
- `aei_raw_1p_api_2025-11-13_to_2025-11-20.csv` - 1P API usage data
- `aei_raw_claude_ai_2025-11-13_to_2025-11-20.csv` - Claude.ai usage data
- Documentation: https://huggingface.co/datasets/Anthropic/EconomicIndex/blob/main/release_2026_01_15/data_documentation.md

## Data Extraction Strategy

### For O*NET Tasks (per platform):
| Metric | Facet | Variable | Description |
|--------|-------|----------|-------------|
| Success Rate | `onet_task::task_success` | `onet_task_task_success_pct` | Filter for `::yes` suffix, gives % successful |
| Duration (mean) | `onet_task::human_only_time` | `onet_task_human_only_time_mean` | Mean human-only time in hours |
| Duration (stdev) | `onet_task::human_only_time` | `onet_task_human_only_time_stdev` | Standard deviation |
| Weight | `onet_task` | `onet_task_pct` | Share of total conversations |

### For Request Categories (per platform):
| Metric | Facet | Variable | Description |
|--------|-------|----------|-------------|
| Success Rate | `request::task_success` | `request_task_success_pct` | Filter for `::yes` suffix |
| Duration (mean) | `request::human_only_time` | `request_human_only_time_mean` | Mean human-only time in hours |
| Duration (stdev) | `request::human_only_time` | `request_human_only_time_stdev` | Standard deviation |
| Weight | `request` | `request_pct` | Share of total conversations |

## Figure Layout

```
+----------------------------------+----------------------------------+
|       1P API - O*NET Tasks       |    Claude.ai - O*NET Tasks       |
|   (n=XXX unique tasks)           |    (n=XXX unique tasks)          |
|   [scatter + WLS + error bars]   |    [scatter + WLS + error bars]  |
+----------------------------------+----------------------------------+
|     1P API - Request Categories  |  Claude.ai - Request Categories  |
|   (n=XXX unique requests)        |    (n=XXX unique requests)       |
|   [scatter + WLS + error bars]   |    [scatter + WLS + error bars]  |
+----------------------------------+----------------------------------+
```

## Axes
- **X-axis**: Duration (Human-Only Time, hours)
- **Y-axis**: Task Success (%)

## Regression Details
- **Method**: Weighted Least Squares (WLS)
- **Weights**: `onet_task_pct` for O*NET tasks, `request_pct` for request categories
- **Interpretation**: Tasks/requests that represent a larger share of usage get more influence on the fitted line

## Error Bars
- Use `*_stdev` values to show variability
- Consider using standard error of the mean (stdev / sqrt(n)) if error bars are too large

## Color Scheme (matching original figure)
- 1P API: Blue (#5B8FB9 or similar)
- Claude.ai: Orange/Coral (#E57373 or similar)

## Implementation Steps

1. **Data Loading**: Read both CSV files into pandas DataFrames
2. **Data Filtering**:
   - Filter for `GLOBAL` geography (1P API is global-only anyway)
   - Extract relevant facets and variables
3. **Data Reshaping**:
   - Pivot from long to wide format
   - Join success rates, durations, stdevs, and weights by cluster_name
4. **Cleaning**:
   - Remove `not_classified` entries
   - Ensure no NaN values in required columns
5. **Visualization**:
   - Create 2x2 subplot figure
   - For each subplot: scatter plot, WLS regression line, error bars
   - Add legends, titles, axis labels
6. **Save**: Export as PNG and PDF

## Output Files
- `task_horizon_4panel.png` - Main figure
- `task_horizon_4panel.pdf` - Vector format for publication
- `analysis_script.py` - Reproducible analysis code
