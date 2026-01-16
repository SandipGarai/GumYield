"""
Agricultural Experiment Statistical Analysis Application v2.2
=============================================================
Comprehensive Streamlit application for analyzing factorial agricultural experiments.

Features:
- Data Summary with extended statistics (Skewness, Kurtosis, CV%)
- Publication-ready visualizations with proper hover labels
- ANOVA with assumption testing and data transformation options
- Welch's ANOVA for heterogeneous variances with Games-Howell post-hoc
- Two-way ANOVA with interaction analysis and bar charts
- Post-hoc Analysis with correct CLD significance letters
- Non-parametric alternatives (Kruskal-Wallis, Friedman, Dunn's test)
- HTML Report Generation

Author: Statistical Analysis Tool
Version: 2.3
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import shapiro, levene, kruskal, f_oneway, friedmanchisquare
from scipy.stats import boxcox, skew, kurtosis, rankdata, norm, t as t_dist
from scipy.stats import studentized_range
from scipy.special import inv_boxcox
from itertools import combinations
from io import BytesIO, StringIO
import base64
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# ========================================
# Page Configuration
# ========================================
st.set_page_config(
    layout="wide",
    page_title="Agricultural Experiment Analysis",
    page_icon="🌳",
    initial_sidebar_state="expanded"
)

# ========================================
# Publication Theme Configuration
# ========================================
THEME = {
    'font_family': 'Arial, Helvetica, sans-serif',
    'title_size': 20,
    'axis_title_size': 15,
    'tick_size': 12,
    'legend_size': 12,
    'annotation_size': 11,
    'line_width': 2.5,
    'marker_size': 8,
    'bgcolor': 'white',
    'grid_color': 'rgba(220, 220, 220, 0.5)',
}

# Professional color palettes
COLOR_PALETTES = {
    'primary': [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf', '#aec7e8', '#ffbb78'
    ],
    'sequential': px.colors.sequential.Viridis,
    'diverging': px.colors.diverging.RdYlGn,
}

MONTH_ORDER = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

# Plotly config for SVG download
PLOTLY_CONFIG = {
    'toImageButtonOptions': {
        'format': 'svg',
        'filename': 'figure',
        'height': 600,
        'width': 900,
        'scale': 2
    },
    'displaylogo': False,
    'modeBarButtonsToAdd': ['downloadImage'],
    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
}


def style_dataframe(df, format_dict=None, highlight_col=None, highlight_condition=None):
    """Apply publication-quality styling to a dataframe"""

    # Create a copy for display
    styled = df.style

    # Apply number formatting
    if format_dict:
        styled = styled.format(format_dict, na_rep='-')

    # Base styling
    styled = styled.set_properties(**{
        'text-align': 'center',
        'font-size': '14px',
        'border': '1px solid #ddd',
        'padding': '8px'
    })

    # Header styling
    styled = styled.set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#1f77b4'),
            ('color', 'white'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('font-size', '14px'),
            ('padding', '10px'),
            ('border', '1px solid #1f77b4')
        ]},
        {'selector': 'td', 'props': [
            ('text-align', 'center'),
            ('padding', '8px')
        ]},
        {'selector': 'tr:nth-child(even)', 'props': [
            ('background-color', '#f9f9f9')
        ]},
        {'selector': 'tr:hover', 'props': [
            ('background-color', '#e8f4f8')
        ]},
        {'selector': 'table', 'props': [
            ('border-collapse', 'collapse'),
            ('width', '100%'),
            ('margin', '10px 0')
        ]}
    ])

    # Highlight significant rows if Significant column exists
    if 'Significant' in df.columns:
        def highlight_significant(row):
            if row.get('Significant', False) == True:
                return ['background-color: #d4edda'] * len(row)
            return [''] * len(row)
        styled = styled.apply(highlight_significant, axis=1)

    # Highlight specific column based on condition
    if highlight_col and highlight_condition and highlight_col in df.columns:
        def highlight_cell(val):
            try:
                if highlight_condition(val):
                    return 'background-color: #d4edda'
            except:
                pass
            return ''
        styled = styled.applymap(highlight_cell, subset=[highlight_col])

    return styled


def display_styled_table(df, title="", format_dict=None, highlight_col=None,
                         highlight_condition=None, key=None):
    """Display a styled table with download option"""

    if title:
        st.markdown(f"**{title}**")

    # Apply styling
    styled = style_dataframe(
        df, format_dict, highlight_col, highlight_condition)

    # Display the table
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Add download button
    csv = df.to_csv(index=False)
    download_key = key if key else f"download_{title.replace(' ', '_')}_{id(df)}"
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{title.replace(' ', '_').lower() if title else 'table'}.csv",
        mime="text/csv",
        key=download_key
    )


# ========================================
# Session State Initialization
# ========================================
if 'transformation_applied' not in st.session_state:
    st.session_state.transformation_applied = None
if 'transformed_data' not in st.session_state:
    st.session_state.transformed_data = None
if 'boxcox_lambda' not in st.session_state:
    st.session_state.boxcox_lambda = None


# ========================================
# Data Loading & Preprocessing
# ========================================
def load_sample_data():
    """Load sample data from data.csv if exists, otherwise return None"""
    if os.path.exists('data.csv'):
        return pd.read_csv('data.csv')
    return None


@st.cache_data
def load_data(file_path_or_buffer):
    """Load and preprocess the dataset"""
    df = pd.read_csv(file_path_or_buffer)

    # Clean column names and values
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

    # Rename columns for clarity
    col_mapping = {
        'TappingTechniques': 'Technique',
        'GumYield.gm': 'GumYield'
    }
    df = df.rename(
        columns={k: v for k, v in col_mapping.items() if k in df.columns})

    # Convert Month to categorical with proper order if exists
    if 'Month' in df.columns:
        available_months = [
            m for m in MONTH_ORDER if m in df['Month'].unique()]
        df['Month'] = pd.Categorical(
            df['Month'], categories=available_months, ordered=True)

    # Sort the dataframe
    sort_cols = [col for col in ['Technique',
                                 'Month', 'Tree'] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def get_factor_columns(df):
    """Dynamically detect factor columns (categorical) and response column (numeric)"""
    factor_cols = []
    response_col = None

    for col in df.columns:
        if df[col].dtype == 'object' or isinstance(df[col].dtype, pd.CategoricalDtype):
            factor_cols.append(col)
        elif np.issubdtype(df[col].dtype, np.number):
            response_col = col

    return factor_cols, response_col


def compute_extended_statistics(df, group_cols, value_col):
    """Compute extended summary statistics including skewness and kurtosis"""

    def safe_cv(x):
        mean_val = x.mean()
        return 100 * x.std() / mean_val if mean_val != 0 else np.nan

    def safe_skew(x):
        if len(x) < 3:
            return np.nan
        return skew(x, nan_policy='omit')

    def safe_kurtosis(x):
        if len(x) < 4:
            return np.nan
        return kurtosis(x, nan_policy='omit')

    stats_df = df.groupby(group_cols)[value_col].agg([
        ('Count', 'count'),
        ('Mean', 'mean'),
        ('Std Dev', 'std'),
        ('CV (%)', safe_cv),
        ('Skewness', safe_skew),
        ('Kurtosis', safe_kurtosis),
        ('Min', 'min'),
        ('25%', lambda x: x.quantile(0.25)),
        ('Median', 'median'),
        ('75%', lambda x: x.quantile(0.75)),
        ('Max', 'max'),
    ]).reset_index()

    return stats_df


# ========================================
# Data Transformation Functions
# ========================================
def apply_transformation(data, transform_type, constant=1):
    """Apply data transformation"""
    data = np.array(data)

    if transform_type == 'log':
        transformed = np.log(data + constant)
    elif transform_type == 'sqrt':
        transformed = np.sqrt(data + constant)
    elif transform_type == 'boxcox':
        shifted_data = data + constant if np.min(data) <= 0 else data
        transformed, lmbda = boxcox(shifted_data)
        st.session_state.boxcox_lambda = lmbda
        return transformed, lmbda
    elif transform_type == 'log10':
        transformed = np.log10(data + constant)
    elif transform_type == 'inverse':
        transformed = 1 / (data + constant)
    else:
        transformed = data

    return transformed, None


# ========================================
# Statistical Tests
# ========================================
def perform_normality_test(data, alpha=0.05):
    """Shapiro-Wilk normality test"""
    data = np.array(data)
    data = data[~np.isnan(data)]

    if len(data) < 3:
        return None, None, "Insufficient data"
    if len(data) > 5000:
        data = np.random.choice(data, 5000, replace=False)

    stat, p_value = shapiro(data)
    is_normal = p_value > alpha
    interpretation = "Normal" if is_normal else "Non-normal"

    return stat, p_value, interpretation


def perform_homogeneity_test(groups, alpha=0.05):
    """Levene's test for homogeneity of variances"""
    groups = [g for g in groups if len(g) > 0]

    if len(groups) < 2:
        return None, None, "Insufficient groups"

    stat, p_value = levene(*groups)
    is_homogeneous = p_value > alpha
    interpretation = "Homogeneous" if is_homogeneous else "Heterogeneous"

    return stat, p_value, interpretation


def perform_anova(df, factor_col, value_col):
    """One-way ANOVA with complete table"""
    groups = [group[value_col].dropna().values for name,
              group in df.groupby(factor_col)]
    groups = [g for g in groups if len(g) > 0]

    if len(groups) < 2:
        return None, None, None, None

    f_stat, p_value = f_oneway(*groups)

    # Calculate ANOVA table components
    grand_mean = df[value_col].mean()
    n_total = len(df[value_col].dropna())
    k = len(groups)  # number of groups

    # Sum of Squares
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = sum((df[value_col].dropna() - grand_mean)**2)
    ss_within = ss_total - ss_between

    # Degrees of freedom
    df_between = k - 1
    df_within = n_total - k
    df_total = n_total - 1

    # Mean Squares
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    # Effect size (eta-squared)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0

    # Omega-squared (less biased estimate)
    omega_squared = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
    omega_squared = max(0, omega_squared)  # Ensure non-negative

    # SEm (Standard Error of Mean) - pooled
    sem = np.sqrt(ms_within / (n_total / k))

    # CD (Critical Difference) at 5% - using t-value
    t_crit = stats.t.ppf(0.975, df_within)
    cd = t_crit * sem * np.sqrt(2)

    anova_table = {
        'Source': ['Between Groups', 'Within Groups', 'Total'],
        'SS': [ss_between, ss_within, ss_total],
        'df': [df_between, df_within, df_total],
        'MS': [ms_between, ms_within, np.nan],
        'F': [f_stat, np.nan, np.nan],
        'p-value': [p_value, np.nan, np.nan]
    }

    additional_stats = {
        'eta_squared': eta_squared,
        'omega_squared': omega_squared,
        'sem': sem,
        'cd': cd,
        'grand_mean': grand_mean,
        'ms_within': ms_within,
        'df_within': df_within,
        'k': k,
        'n_total': n_total
    }

    return f_stat, p_value, anova_table, additional_stats


def perform_welch_anova(df, factor_col, value_col):
    """Welch's ANOVA for heterogeneous variances with additional statistics"""
    from scipy.stats import f as f_dist

    groups_data = {name: group[value_col].dropna().values
                   for name, group in df.groupby(factor_col)}
    groups = list(groups_data.values())
    group_names = list(groups_data.keys())

    if len(groups) < 2:
        return None, None, None, None

    # Calculate group statistics
    n = np.array([len(g) for g in groups])
    means = np.array([np.mean(g) for g in groups])
    variances = np.array([np.var(g, ddof=1) for g in groups])

    # Weights
    w = n / variances

    # Weighted mean
    weighted_mean = np.sum(w * means) / np.sum(w)

    # Welch's F statistic
    k = len(groups)
    numerator = np.sum(w * (means - weighted_mean)**2) / (k - 1)

    # Lambda values for denominator
    lambda_vals = (1 - w / np.sum(w))**2 / (n - 1)
    denominator = 1 + (2 * (k - 2) / (k**2 - 1)) * np.sum(lambda_vals)

    f_stat = numerator / denominator

    # Degrees of freedom
    df1 = k - 1
    df2 = 1 / (3 * np.sum(lambda_vals) / (k**2 - 1))

    # p-value
    p_value = 1 - f_dist.cdf(f_stat, df1, df2)

    # Effect size (omega-squared for Welch's ANOVA)
    # Using adjusted formula for unequal variances
    n_total = sum(n)
    grand_mean = np.mean(df[value_col].dropna())

    # Estimated omega-squared
    ss_between = np.sum(n * (means - grand_mean)**2)
    ss_total = np.sum((df[value_col].dropna() - grand_mean)**2)
    ms_within_pooled = np.mean(variances)  # Approximate

    omega_squared = (ss_between - (k - 1) * ms_within_pooled) / \
        (ss_total + ms_within_pooled)
    omega_squared = max(0, omega_squared)

    eta_squared = ss_between / ss_total if ss_total > 0 else 0

    welch_table = {
        'Source': ['Between Groups (Welch)', 'Residual'],
        'df': [df1, df2],
        'F': [f_stat, np.nan],
        'p-value': [p_value, np.nan]
    }

    additional_stats = {
        'eta_squared': eta_squared,
        'omega_squared': omega_squared,
        'grand_mean': grand_mean,
        'k': k,
        'n_total': n_total,
        'group_means': dict(zip(group_names, means)),
        'group_variances': dict(zip(group_names, variances)),
        'group_ns': dict(zip(group_names, n))
    }

    return f_stat, p_value, welch_table, additional_stats


def perform_games_howell(df, factor_col, value_col):
    """
    Games-Howell post-hoc test for unequal variances.
    Appropriate for use after Welch's ANOVA.
    """
    groups = sorted(df[factor_col].unique())
    n_groups = len(groups)

    # Calculate group statistics
    group_stats = {}
    for g in groups:
        g_data = df[df[factor_col] == g][value_col].dropna()
        group_stats[g] = {
            'mean': g_data.mean(),
            'var': g_data.var(ddof=1),
            'n': len(g_data),
            'se': np.sqrt(g_data.var(ddof=1) / len(g_data))
        }

    results = []

    for g1, g2 in combinations(groups, 2):
        n1 = group_stats[g1]['n']
        n2 = group_stats[g2]['n']
        m1 = group_stats[g1]['mean']
        m2 = group_stats[g2]['mean']
        v1 = group_stats[g1]['var']
        v2 = group_stats[g2]['var']

        # Mean difference
        mean_diff = m1 - m2

        # Standard error of the difference
        se_diff = np.sqrt(v1/n1 + v2/n2)

        # Welch-Satterthwaite degrees of freedom
        num = (v1/n1 + v2/n2)**2
        denom = (v1/n1)**2 / (n1 - 1) + (v2/n2)**2 / (n2 - 1)
        df_gh = num / denom

        # t-statistic
        t_stat = mean_diff / se_diff

        # Critical value from studentized range distribution
        # Using approximation with t-distribution for confidence intervals
        q_crit = stats.t.ppf(0.975, df_gh)

        # p-value (two-tailed)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_gh))

        # Bonferroni-like adjustment for multiple comparisons
        # Using the number of comparisons
        n_comparisons = n_groups * (n_groups - 1) / 2
        p_adj = min(p_value * n_comparisons, 1.0)

        # Confidence interval
        margin = q_crit * se_diff
        lower_ci = mean_diff - margin
        upper_ci = mean_diff + margin

        results.append({
            'Group1': g1,
            'Group2': g2,
            'Mean Diff': mean_diff,
            'SE': se_diff,
            'df': df_gh,
            't-stat': t_stat,
            'p-adj': p_adj,
            'Lower CI': lower_ci,
            'Upper CI': upper_ci,
            'Significant': p_adj < 0.05
        })

    results_df = pd.DataFrame(results)

    # Calculate group means for ordering
    group_means = df.groupby(factor_col)[
        value_col].mean().sort_values(ascending=False)
    sorted_groups = list(group_means.index)

    # Generate significance letters
    letters = assign_cld_letters(results_df, sorted_groups)

    return results_df, letters


def perform_two_way_anova(df, factor1, factor2, value_col):
    """Two-way ANOVA using statsmodels with additional statistics"""
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm

        df_clean = df.copy()
        df_clean['Factor1'] = df_clean[factor1].astype(str)
        df_clean['Factor2'] = df_clean[factor2].astype(str)
        df_clean['Response'] = df_clean[value_col]

        formula = 'Response ~ C(Factor1) + C(Factor2) + C(Factor1):C(Factor2)'
        model = ols(formula, data=df_clean).fit()
        anova_table = anova_lm(model, typ=2)

        anova_table.index = anova_table.index.str.replace(
            'C(Factor1)', factor1)
        anova_table.index = anova_table.index.str.replace(
            'C(Factor2)', factor2)
        anova_table.index = anova_table.index.str.replace(':', ' × ')

        # Calculate additional statistics
        ss_total = anova_table['sum_sq'].sum()
        n_total = len(df_clean['Response'].dropna())
        grand_mean = df_clean['Response'].mean()

        # Effect sizes for each factor
        effect_sizes = {}
        for idx in anova_table.index:
            if idx != 'Residual':
                eta_sq = anova_table.loc[idx, 'sum_sq'] / ss_total
                # Partial eta-squared
                partial_eta_sq = anova_table.loc[idx, 'sum_sq'] / (
                    anova_table.loc[idx, 'sum_sq'] + anova_table.loc['Residual', 'sum_sq'])
                effect_sizes[idx] = {
                    'eta_squared': eta_sq,
                    'partial_eta_squared': partial_eta_sq
                }

        # MSE and related statistics
        ms_error = anova_table.loc['Residual',
                                   'sum_sq'] / anova_table.loc['Residual', 'df']
        df_error = anova_table.loc['Residual', 'df']

        # SEm for main effects
        n_per_cell = n_total / \
            (df_clean['Factor1'].nunique() * df_clean['Factor2'].nunique())
        sem = np.sqrt(ms_error / n_per_cell)

        # CD (Critical Difference) at 5%
        t_crit = stats.t.ppf(0.975, df_error)
        cd = t_crit * sem * np.sqrt(2)

        additional_stats = {
            'effect_sizes': effect_sizes,
            'ms_error': ms_error,
            'df_error': df_error,
            'sem': sem,
            'cd': cd,
            'grand_mean': grand_mean,
            'n_total': n_total,
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj
        }

        return anova_table, model, additional_stats
    except Exception as e:
        return None, str(e), None


def perform_kruskal_wallis(df, factor_col, value_col):
    """Kruskal-Wallis H-test"""
    groups = [group[value_col].dropna().values for name,
              group in df.groupby(factor_col)]
    groups = [g for g in groups if len(g) > 0]

    if len(groups) < 2:
        return None, None

    h_stat, p_value = kruskal(*groups)
    return h_stat, p_value


def perform_friedman_test(df, factor_col, block_col, value_col):
    """Friedman test for repeated measures"""
    try:
        pivot_df = df.pivot_table(
            values=value_col, index=block_col, columns=factor_col, aggfunc='mean')
        pivot_df = pivot_df.dropna()

        if pivot_df.shape[0] < 2 or pivot_df.shape[1] < 2:
            return None, None, "Insufficient data for Friedman test"

        stat, p_value = friedmanchisquare(
            *[pivot_df[col].values for col in pivot_df.columns])
        return stat, p_value, None
    except Exception as e:
        return None, None, str(e)


def perform_nemenyi_test(df, factor_col, block_col, value_col):
    """
    Nemenyi post-hoc test for Friedman test.
    Compares all pairs of groups using rank-based approach.
    """
    try:
        # Create pivot table
        pivot_df = df.pivot_table(
            values=value_col, index=block_col, columns=factor_col, aggfunc='mean')
        pivot_df = pivot_df.dropna()

        n = pivot_df.shape[0]  # number of blocks
        k = pivot_df.shape[1]  # number of treatments
        groups = list(pivot_df.columns)

        # Rank within each block (row)
        ranks_df = pivot_df.rank(axis=1)

        # Calculate mean ranks for each treatment
        mean_ranks = ranks_df.mean()

        # Critical difference for Nemenyi test
        # q_alpha values for alpha=0.05 (studentized range / sqrt(2))
        # Approximation using standard formula
        q_alpha = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)

        # CD = q_alpha * sqrt(k*(k+1)/(6*n))
        cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

        # Pairwise comparisons
        results = []
        for g1, g2 in combinations(groups, 2):
            diff = abs(mean_ranks[g1] - mean_ranks[g2])
            significant = diff > cd

            # Calculate approximate p-value using normal approximation
            se = np.sqrt(k * (k + 1) / (6 * n))
            z = diff / se
            p_value = 2 * (1 - norm.cdf(z))
            p_adj = min(p_value * (k * (k - 1) / 2), 1.0)

            results.append({
                'Group1': g1,
                'Group2': g2,
                'Mean Rank 1': mean_ranks[g1],
                'Mean Rank 2': mean_ranks[g2],
                'Rank Diff': diff,
                'CD': cd,
                'p-adj': p_adj,
                'Significant': significant
            })

        results_df = pd.DataFrame(results)

        # Generate CLD letters
        sorted_groups = list(mean_ranks.sort_values().index)
        letters = assign_cld_letters(results_df, sorted_groups)

        # Additional info
        rank_summary = pd.DataFrame({
            'Group': groups,
            'Mean Rank': [mean_ranks[g] for g in groups]
        }).sort_values('Mean Rank')

        return results_df, letters, cd, rank_summary
    except Exception as e:
        return None, None, None, None


def perform_dunn_test(df, factor_col, value_col):
    """Dunn's post-hoc test for Kruskal-Wallis"""
    try:
        groups = sorted(df[factor_col].unique())
        n_groups = len(groups)

        all_data = df[value_col].values
        ranks = rankdata(all_data)
        df_temp = df.copy()
        df_temp['rank'] = ranks

        group_stats = {}
        for g in groups:
            g_data = df_temp[df_temp[factor_col] == g]
            group_stats[g] = {
                'mean_rank': g_data['rank'].mean(),
                'n': len(g_data)
            }

        N = len(all_data)

        results = {}
        for g1, g2 in combinations(groups, 2):
            n1 = group_stats[g1]['n']
            n2 = group_stats[g2]['n']
            R1 = group_stats[g1]['mean_rank']
            R2 = group_stats[g2]['mean_rank']

            se = np.sqrt((N * (N + 1) / 12) * (1/n1 + 1/n2))
            z = (R1 - R2) / se
            p = 2 * (1 - norm.cdf(abs(z)))
            p_adj = min(p * (n_groups * (n_groups - 1) / 2), 1.0)

            results[(g1, g2)] = p_adj
            results[(g2, g1)] = p_adj

        result_df = pd.DataFrame(index=groups, columns=groups, dtype=float)
        for g in groups:
            result_df.loc[g, g] = 1.0
        for (g1, g2), p in results.items():
            result_df.loc[g1, g2] = p

        return result_df
    except Exception as e:
        return None


def perform_tukey_hsd(df, factor_col, value_col):
    """Tukey's HSD post-hoc test - FIXED version"""
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        df_clean = df[[factor_col, value_col]].dropna()

        tukey_result = pairwise_tukeyhsd(
            endog=df_clean[value_col],
            groups=df_clean[factor_col],
            alpha=0.05
        )

        # Extract results properly - avoid Cell objects
        summary_data = tukey_result.summary().data
        headers = summary_data[0]

        results_list = []
        for row in summary_data[1:]:
            results_list.append({
                'Group1': str(row[0]),
                'Group2': str(row[1]),
                'Mean Diff': float(row[2]),
                'p-adj': float(row[3]),
                'Lower CI': float(row[4]),
                'Upper CI': float(row[5]),
                'Significant': str(row[6]).lower() == 'true'
            })

        results_df = pd.DataFrame(results_list)

        # Get unique groups
        groups = sorted(df_clean[factor_col].unique())

        # Calculate group means for ordering
        group_means = df_clean.groupby(
            factor_col)[value_col].mean().sort_values(ascending=False)
        sorted_groups = list(group_means.index)

        # Generate significance letters using correct CLD algorithm
        letters = assign_cld_letters(results_df, sorted_groups)

        return results_df, tukey_result, letters
    except Exception as e:
        st.error(f"Tukey HSD Error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None


def assign_cld_letters(pairwise_df, sorted_groups):
    """
    Correct Compact Letter Display (CLD) algorithm
    Groups sharing a letter are NOT significantly different
    """
    n_groups = len(sorted_groups)

    # Build significance matrix (True = NOT significantly different)
    sig_matrix = np.ones((n_groups, n_groups), dtype=bool)

    group_to_idx = {g: i for i, g in enumerate(sorted_groups)}

    for _, row in pairwise_df.iterrows():
        g1 = row['Group1']
        g2 = row['Group2']
        i = group_to_idx.get(g1)
        j = group_to_idx.get(g2)
        if i is not None and j is not None:
            if row['Significant']:
                sig_matrix[i, j] = False
                sig_matrix[j, i] = False

    # CLD algorithm: find maximal groups of non-significantly different treatments
    letters = {g: '' for g in sorted_groups}
    current_letter = ord('a')

    # Find all maximal cliques of non-significantly different groups
    assigned = [False] * n_groups

    while not all(assigned):
        # Start with first unassigned or partially assigned group
        # Find groups that can share the current letter
        candidates = list(range(n_groups))

        # Greedy approach: build maximal compatible set
        compatible_set = []

        for i in candidates:
            can_add = True
            for j in compatible_set:
                if not sig_matrix[i, j]:
                    can_add = False
                    break
            if can_add:
                compatible_set.append(i)

        # Check if this set adds new information
        # (covers at least one group that needs distinction)
        useful = False
        for i in compatible_set:
            if not assigned[i]:
                useful = True
                break

        if useful and len(compatible_set) > 0:
            letter = chr(current_letter)
            for idx in compatible_set:
                letters[sorted_groups[idx]] += letter
                assigned[idx] = True
            current_letter += 1
        else:
            # Force assign to remaining
            for i in range(n_groups):
                if not assigned[i]:
                    letters[sorted_groups[i]] += chr(current_letter)
                    assigned[i] = True
                    current_letter += 1

        if current_letter > ord('z'):
            break

    # Verify: groups with same letter should not be significantly different
    # If any issues, refine
    return letters


def perform_two_way_posthoc(df, factor1, factor2, value_col):
    """Post-hoc analysis for two-way ANOVA"""
    results = {}

    # Main effects post-hoc
    for factor in [factor1, factor2]:
        tukey_df, _, letters = perform_tukey_hsd(df, factor, value_col)
        if tukey_df is not None:
            results[factor] = {'tukey': tukey_df, 'letters': letters}

    # Interaction post-hoc (simple effects)
    df_temp = df.copy()
    df_temp['Interaction'] = df_temp[factor1].astype(
        str) + ' × ' + df_temp[factor2].astype(str)
    tukey_df, _, letters = perform_tukey_hsd(df_temp, 'Interaction', value_col)
    if tukey_df is not None:
        results['Interaction'] = {'tukey': tukey_df, 'letters': letters}
        # Store the dataframe for bar chart
        results['Interaction']['df'] = df_temp

    return results


# ========================================
# Publication-Ready Visualization Functions
# ========================================
def create_publication_layout(fig, title, xaxis_title, yaxis_title,
                              show_legend=True, legend_position='right',
                              height=550, width=900, legend_title=None):
    """Apply publication-ready styling to a Plotly figure"""

    if legend_position == 'right':
        legend_config = dict(
            x=1.02, y=0.98, xanchor='left', yanchor='top',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='black', borderwidth=1,
            font=dict(size=THEME['legend_size'],
                      family=THEME['font_family'], color='black'),
            title=dict(text=legend_title, font=dict(size=THEME['legend_size']+1,
                                                    family=THEME['font_family'], color='black')) if legend_title else None
        )
        right_margin = 180
    elif legend_position == 'bottom':
        legend_config = dict(
            x=0.5, y=-0.25, xanchor='center', yanchor='top', orientation='h',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='black', borderwidth=1,
            font=dict(size=THEME['legend_size'],
                      family=THEME['font_family'], color='black'),
            title=dict(text=legend_title, font=dict(size=THEME['legend_size']+1,
                                                    family=THEME['font_family'], color='black')) if legend_title else None
        )
        right_margin = 50
    else:
        legend_config = dict(
            x=0.98, y=0.98, xanchor='right', yanchor='top',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='black', borderwidth=1,
            font=dict(size=THEME['legend_size'],
                      family=THEME['font_family'], color='black'),
            title=dict(text=legend_title, font=dict(size=THEME['legend_size']+1,
                                                    family=THEME['font_family'], color='black')) if legend_title else None
        )
        right_margin = 50

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=THEME['title_size'],
                      family=THEME['font_family'], color='black'),
            x=0.5, xanchor='center', y=0.95
        ),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(size=THEME['axis_title_size'],
                       family=THEME['font_family'], color='black'), standoff=15),
            tickfont=dict(size=THEME['tick_size'],
                          family=THEME['font_family'], color='black'),
            gridcolor=THEME['grid_color'],
            showgrid=True,
            linecolor='black',
            linewidth=1.5,
            mirror=True,
            tickangle=-45 if len(xaxis_title) > 0 else 0,
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(size=THEME['axis_title_size'],
                       family=THEME['font_family'], color='black'), standoff=10),
            tickfont=dict(size=THEME['tick_size'],
                          family=THEME['font_family'], color='black'),
            gridcolor=THEME['grid_color'],
            showgrid=True,
            linecolor='black',
            linewidth=1.5,
            mirror=True,
        ),
        legend=legend_config if show_legend else dict(visible=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family=THEME['font_family'], color='black'),
        margin=dict(l=70, r=right_margin, t=80, b=100),
        hoverlabel=dict(
            bgcolor='white',
            font_size=13,
            font_family=THEME['font_family'],
            font_color='black',
            bordercolor='black',
            namelength=-1
        ),
        height=height,
        width=width,
    )

    return fig


def plot_boxplot(df, factor_col, value_col, show_points=True):
    """Create publication-ready boxplot"""
    fig = go.Figure()

    groups = sorted(df[factor_col].unique())
    colors = COLOR_PALETTES['primary']

    for i, group in enumerate(groups):
        group_data = df[df[factor_col] == group][value_col]
        color = colors[i % len(colors)]

        fig.add_trace(go.Box(
            y=group_data,
            name=str(group),
            marker_color=color,
            boxmean='sd',
            boxpoints='all' if show_points else 'outliers',
            jitter=0.3,
            pointpos=0,
            marker=dict(
                size=6,
                opacity=0.7,
                color=color,
                line=dict(width=1, color='black')
            ),
            line=dict(width=2, color='black'),
            fillcolor=color,
            opacity=0.75,
            hovertemplate=(
                f"<b>{factor_col}: {group}</b><br>"
                f"{value_col}: %{{y:.2f}}<br>"
                "<extra></extra>"
            )
        ))

    fig = create_publication_layout(
        fig,
        title=f"Distribution of {value_col} by {factor_col}",
        xaxis_title=factor_col,
        yaxis_title=value_col,
        show_legend=False
    )

    return fig


def plot_violin(df, factor_col, value_col):
    """Create publication-ready violin plot"""
    fig = go.Figure()

    groups = sorted(df[factor_col].unique())
    colors = COLOR_PALETTES['primary']

    for i, group in enumerate(groups):
        group_data = df[df[factor_col] == group][value_col]
        color = colors[i % len(colors)]

        fig.add_trace(go.Violin(
            y=group_data,
            name=str(group),
            box_visible=True,
            meanline_visible=True,
            fillcolor=color,
            line_color='black',
            opacity=0.7,
            points='all',
            jitter=0.3,
            pointpos=0,
            marker=dict(size=4, color=color, line=dict(
                width=0.5, color='black'), opacity=0.6),
            hovertemplate=(
                f"<b>{factor_col}: {group}</b><br>"
                f"{value_col}: %{{y:.2f}}<br>"
                "<extra></extra>"
            )
        ))

    fig = create_publication_layout(
        fig,
        title=f"Distribution of {value_col} by {factor_col}",
        xaxis_title=factor_col,
        yaxis_title=value_col,
        show_legend=False
    )

    return fig


def plot_bar_with_error(df, factor_col, value_col, letters=None):
    """Create bar plot with error bars and optional significance letters"""
    summary = df.groupby(factor_col)[value_col].agg(
        ['mean', 'std', 'count']).reset_index()
    summary['se'] = summary['std'] / np.sqrt(summary['count'])
    summary = summary.sort_values(factor_col)

    colors = COLOR_PALETTES['primary']

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=summary[factor_col].astype(str),
        y=summary['mean'],
        error_y=dict(
            type='data',
            array=summary['se'],
            visible=True,
            color='black',
            thickness=2,
            width=4,
        ),
        marker=dict(
            color=[colors[i % len(colors)] for i in range(len(summary))],
            line=dict(color='black', width=1.5),
        ),
        hovertemplate=(
            f"<b>{factor_col}: %{{x}}</b><br>"
            f"Mean: %{{y:.2f}} ± %{{error_y.array:.2f}}<br>"
            "<extra></extra>"
        )
    ))

    if letters:
        annotations = []
        for i, row in summary.iterrows():
            group = row[factor_col]
            if group in letters:
                annotations.append(dict(
                    x=str(group),
                    y=row['mean'] + row['se'] + (summary['mean'].max() * 0.05),
                    text=f"<b>{letters[group]}</b>",
                    showarrow=False,
                    font=dict(size=14, color='black', family='Arial'),
                    xanchor='center'
                ))
        fig.update_layout(annotations=annotations)

    fig = create_publication_layout(
        fig,
        title=f"Mean {value_col} by {factor_col}" +
        (" with Significance Letters" if letters else ""),
        xaxis_title=factor_col,
        yaxis_title=f"Mean {value_col} ± SE",
        show_legend=False
    )

    return fig


def plot_interaction_grouped_bar(df, factor1, factor2, value_col, letters=None):
    """Create grouped bar chart for interaction effects"""
    summary = df.groupby([factor1, factor2])[value_col].agg(
        ['mean', 'std', 'count']).reset_index()
    summary['se'] = summary['std'] / np.sqrt(summary['count'])

    # Create interaction label
    summary['Interaction'] = summary[factor1].astype(
        str) + ' × ' + summary[factor2].astype(str)

    fig = go.Figure()

    groups1 = sorted(df[factor1].unique())
    groups2 = sorted(df[factor2].unique())
    colors = COLOR_PALETTES['primary']

    for i, g1 in enumerate(groups1):
        g1_data = summary[summary[factor1] == g1].sort_values(factor2)
        color = colors[i % len(colors)]

        # Get letters for this group if available
        letter_texts = []
        if letters:
            for _, row in g1_data.iterrows():
                interaction_key = f"{row[factor1]} × {row[factor2]}"
                letter_texts.append(letters.get(interaction_key, ''))

        fig.add_trace(go.Bar(
            name=str(g1),
            x=g1_data[factor2].astype(str),
            y=g1_data['mean'],
            error_y=dict(
                type='data',
                array=g1_data['se'],
                visible=True,
                color='black',
                thickness=1.5,
                width=3,
            ),
            marker=dict(color=color, line=dict(color='black', width=1)),
            text=letter_texts if letters else None,
            textposition='outside',
            textfont=dict(size=10, color='black', family='Arial'),
            hovertemplate=(
                f"<b>{factor1}: {g1}</b><br>"
                f"{factor2}: %{{x}}<br>"
                f"Mean: %{{y:.2f}} ± %{{error_y.array:.2f}}<br>"
                "<extra></extra>"
            )
        ))

    fig.update_layout(barmode='group')

    fig = create_publication_layout(
        fig,
        title=f"Interaction: {factor1} × {factor2}",
        xaxis_title=factor2,
        yaxis_title=f"Mean {value_col} ± SE",
        show_legend=True,
        legend_position='right',
        legend_title=factor1
    )

    # Adjust for text labels
    if letters:
        fig.update_layout(margin=dict(t=100))

    return fig


def plot_interaction_dot(df, factor1, factor2, value_col, letters=None, horizontal=False):
    """Create dot plot for interaction effects - cleaner for many groups"""
    summary = df.groupby([factor1, factor2])[value_col].agg(
        ['mean', 'std', 'count']).reset_index()
    summary['se'] = summary['std'] / np.sqrt(summary['count'])
    summary['Interaction'] = summary[factor1].astype(
        str) + ' × ' + summary[factor2].astype(str)

    # Sort by mean for better visualization
    summary = summary.sort_values('mean', ascending=True)

    fig = go.Figure()

    groups1 = sorted(df[factor1].unique())
    colors = COLOR_PALETTES['primary']
    color_map = {g: colors[i % len(colors)] for i, g in enumerate(groups1)}

    for _, row in summary.iterrows():
        color = color_map[row[factor1]]
        letter = letters.get(row['Interaction'], '') if letters else ''

        if horizontal:
            fig.add_trace(go.Scatter(
                x=[row['mean']],
                y=[row['Interaction']],
                mode='markers',
                marker=dict(size=12, color=color, line=dict(
                    width=1.5, color='black')),
                error_x=dict(type='data', array=[row['se']], visible=True,
                             color=color, thickness=2, width=6),
                name=str(row[factor1]),
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['Interaction']}</b><br>"
                    f"Mean: {row['mean']:.2f} ± {row['se']:.2f}<br>"
                    "<extra></extra>"
                )
            ))
            if letter:
                fig.add_annotation(
                    x=row['mean'] + row['se'] + (summary['mean'].max() * 0.03),
                    y=row['Interaction'],
                    text=f"<b>{letter}</b>",
                    showarrow=False,
                    font=dict(size=11, color='black'),
                    xanchor='left'
                )
        else:
            fig.add_trace(go.Scatter(
                x=[row['Interaction']],
                y=[row['mean']],
                mode='markers',
                marker=dict(size=12, color=color, line=dict(
                    width=1.5, color='black')),
                error_y=dict(type='data', array=[row['se']], visible=True,
                             color=color, thickness=2, width=6),
                name=str(row[factor1]),
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['Interaction']}</b><br>"
                    f"Mean: {row['mean']:.2f} ± {row['se']:.2f}<br>"
                    "<extra></extra>"
                )
            ))

    # Add legend entries
    for g1 in groups1:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=12, color=color_map[g1]),
            name=str(g1),
            showlegend=True
        ))

    if horizontal:
        fig = create_publication_layout(
            fig,
            title=f"Interaction: {factor1} × {factor2}",
            xaxis_title=f"Mean {value_col} ± SE",
            yaxis_title="",
            show_legend=True,
            legend_position='right',
            legend_title=factor1,
            height=max(400, 50 + len(summary) * 25)
        )
        fig.update_yaxes(tickfont=dict(size=10))
    else:
        fig = create_publication_layout(
            fig,
            title=f"Interaction: {factor1} × {factor2}",
            xaxis_title="",
            yaxis_title=f"Mean {value_col} ± SE",
            show_legend=True,
            legend_position='right',
            legend_title=factor1
        )
        fig.update_xaxes(tickangle=-45, tickfont=dict(size=9))

    return fig


def plot_interaction_facet(df, factor1, factor2, value_col, letters=None):
    """Create faceted bar chart - one subplot per level of factor1"""
    groups1 = sorted(df[factor1].unique())
    groups2 = sorted(df[factor2].unique())
    n_cols = min(3, len(groups1))
    n_rows = (len(groups1) + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[str(g) for g in groups1],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )

    colors = COLOR_PALETTES['primary']

    for idx, g1 in enumerate(groups1):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        g1_data = df[df[factor1] == g1].groupby(factor2)[value_col].agg(
            ['mean', 'std', 'count']).reset_index()
        g1_data['se'] = g1_data['std'] / np.sqrt(g1_data['count'])
        g1_data = g1_data.sort_values(factor2)

        # Get letters if available
        letter_texts = []
        if letters:
            for _, row_data in g1_data.iterrows():
                interaction_key = f"{g1} × {row_data[factor2]}"
                letter_texts.append(letters.get(interaction_key, ''))

        fig.add_trace(
            go.Bar(
                x=g1_data[factor2].astype(str),
                y=g1_data['mean'],
                error_y=dict(type='data', array=g1_data['se'], visible=True,
                             color='black', thickness=1.5, width=3),
                marker=dict(color=colors[idx % len(colors)],
                            line=dict(color='black', width=1)),
                text=letter_texts if letters else None,
                textposition='outside',
                textfont=dict(size=9),
                showlegend=False,
                hovertemplate=(
                    f"<b>{factor1}: {g1}</b><br>"
                    f"{factor2}: %{{x}}<br>"
                    f"Mean: %{{y:.2f}}<br>"
                    "<extra></extra>"
                )
            ),
            row=row, col=col
        )

    fig.update_layout(
        title=dict(
            text=f"<b>Interaction: {factor1} × {factor2}</b>",
            font=dict(size=THEME['title_size'],
                      family=THEME['font_family'], color='black'),
            x=0.5, xanchor='center'
        ),
        height=300 * n_rows,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family=THEME['font_family'], color='black'),
        margin=dict(t=80)
    )

    # Update all axes
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=10), linecolor='black',
                     linewidth=1, mirror=True, gridcolor=THEME['grid_color'])
    fig.update_yaxes(title_text=f"Mean {value_col}", tickfont=dict(size=10),
                     linecolor='black', linewidth=1, mirror=True, gridcolor=THEME['grid_color'])

    return fig


def plot_interaction_heatmap_enhanced(df, factor1, factor2, value_col, letters=None):
    """Enhanced heatmap with optional significance letters inside cells"""
    pivot_table = df.pivot_table(
        values=value_col, index=factor1, columns=factor2, aggfunc='mean')

    if factor2 == 'Month':
        available_months = [m for m in MONTH_ORDER if m in pivot_table.columns]
        pivot_table = pivot_table[available_months]

    # Create text annotations with letters
    text_matrix = []
    for i, row_name in enumerate(pivot_table.index):
        row_text = []
        for j, col_name in enumerate(pivot_table.columns):
            val = pivot_table.iloc[i, j]
            if letters:
                interaction_key = f"{row_name} × {col_name}"
                letter = letters.get(interaction_key, '')
                row_text.append(f"{val:.1f}<br><b>{letter}</b>")
            else:
                row_text.append(f"{val:.1f}")
        text_matrix.append(row_text)

    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=[str(c)[:3] if factor2 == 'Month' else str(c)
           for c in pivot_table.columns],
        y=pivot_table.index.astype(str),
        colorscale='RdYlGn',
        colorbar=dict(
            title=dict(text=f'Mean {value_col}',
                       font=dict(size=13, color='black')),
            tickfont=dict(size=11, color='black'),
            thickness=18,
            len=0.9,
        ),
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=10, color='black'),
        hovertemplate=(
            f"<b>{factor1}: %{{y}}</b><br>"
            f"<b>{factor2}: %{{x}}</b><br>"
            f"Mean: %{{z:.2f}}<br>"
            "<extra></extra>"
        ),
        showscale=True,
    ))

    fig = create_publication_layout(
        fig,
        title=f"Interaction Heatmap: {factor1} × {factor2}" +
        (" with CLD" if letters else ""),
        xaxis_title=factor2,
        yaxis_title=factor1,
        show_legend=False
    )

    fig.update_xaxes(tickangle=0)

    return fig


def plot_interaction(df, factor1, factor2, value_col):
    """Create interaction plot"""
    summary = df.groupby([factor1, factor2])[value_col].agg(
        ['mean', 'std', 'count']).reset_index()
    summary['se'] = summary['std'] / np.sqrt(summary['count'])

    fig = go.Figure()

    groups1 = sorted(df[factor1].unique())
    colors = COLOR_PALETTES['primary']

    for i, g1 in enumerate(groups1):
        g1_data = summary[summary[factor1] == g1].sort_values(factor2)
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=g1_data[factor2].astype(str),
            y=g1_data['mean'],
            mode='lines+markers',
            name=str(g1),
            line=dict(color=color, width=THEME['line_width']),
            marker=dict(size=THEME['marker_size'], color=color,
                        line=dict(width=1.5, color='black')),
            error_y=dict(
                type='data',
                array=g1_data['se'],
                visible=True,
                color=color,
                thickness=1.5,
                width=4,
            ),
            hovertemplate=(
                f"<b>{factor1}: {g1}</b><br>"
                f"{factor2}: %{{x}}<br>"
                f"Mean: %{{y:.2f}} ± %{{error_y.array:.2f}}<br>"
                "<extra></extra>"
            )
        ))

    fig = create_publication_layout(
        fig,
        title=f"Interaction Plot: {factor1} × {factor2}",
        xaxis_title=factor2,
        yaxis_title=f"Mean {value_col} ± SE",
        show_legend=True,
        legend_position='right',
        legend_title=factor1
    )

    return fig


def plot_heatmap(df, factor1, factor2, value_col):
    """Create heatmap without text inside cells"""
    pivot_table = df.pivot_table(
        values=value_col, index=factor1, columns=factor2, aggfunc='mean')

    if factor2 == 'Month':
        available_months = [m for m in MONTH_ORDER if m in pivot_table.columns]
        pivot_table = pivot_table[available_months]

    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=[str(c)[:3] if factor2 == 'Month' else str(c)
           for c in pivot_table.columns],
        y=pivot_table.index.astype(str),
        colorscale='RdYlGn',
        colorbar=dict(
            title=dict(text=f'Mean {value_col}',
                       font=dict(size=13, color='black')),
            tickfont=dict(size=11, color='black'),
            thickness=18,
            len=0.9,
        ),
        hovertemplate=(
            f"<b>{factor1}: %{{y}}</b><br>"
            f"<b>{factor2}: %{{x}}</b><br>"
            f"Mean {value_col}: %{{z:.2f}}<br>"
            "<extra></extra>"
        ),
        showscale=True,
    ))

    fig = create_publication_layout(
        fig,
        title=f"Heatmap: Mean {value_col} by {factor1} × {factor2}",
        xaxis_title=factor2,
        yaxis_title=factor1,
        show_legend=False
    )

    fig.update_xaxes(tickangle=0)

    return fig


def plot_tukey_heatmap(tukey_df, groups, title="Tukey HSD Pairwise Comparisons", lower_triangular=True):
    """Create heatmap for post-hoc results - color shows mean diff, stars show significance"""
    n = len(groups)

    # Create matrices for p-values and mean differences
    p_matrix = np.ones((n, n))
    diff_matrix = np.zeros((n, n))
    lower_ci = np.zeros((n, n))
    upper_ci = np.zeros((n, n))

    group_to_idx = {g: i for i, g in enumerate(groups)}

    for _, row in tukey_df.iterrows():
        i = group_to_idx.get(row['Group1'])
        j = group_to_idx.get(row['Group2'])
        if i is not None and j is not None:
            p_matrix[i, j] = row['p-adj']
            p_matrix[j, i] = row['p-adj']
            diff_matrix[i, j] = row['Mean Diff']
            diff_matrix[j, i] = -row['Mean Diff']
            lower_ci[i, j] = row['Lower CI']
            lower_ci[j, i] = -row['Upper CI']
            upper_ci[i, j] = row['Upper CI']
            upper_ci[j, i] = -row['Lower CI']

    # Create lower triangular mask for display
    if lower_triangular:
        mask = np.triu(np.ones_like(diff_matrix, dtype=bool), k=0)
        diff_matrix_display = np.where(mask, np.nan, diff_matrix)
    else:
        diff_matrix_display = diff_matrix.copy()
        for i in range(n):
            diff_matrix_display[i, i] = np.nan

    # Get color range (symmetric around 0)
    max_abs_diff = np.nanmax(np.abs(diff_matrix_display))
    if max_abs_diff == 0:
        max_abs_diff = 1

    # Create significance annotations (only for lower triangle)
    annotations = []
    for i in range(n):
        for j in range(n):
            if lower_triangular:
                if i <= j:
                    continue
            else:
                if i == j:
                    continue

            p = p_matrix[i, j]
            diff_val = diff_matrix[i, j]
            if p < 0.001:
                star = "***"
            elif p < 0.01:
                star = "**"
            elif p < 0.05:
                star = "*"
            else:
                star = "ns"

            # Choose text color based on background intensity
            text_color = 'white' if abs(
                diff_val) > max_abs_diff * 0.5 else 'black'

            annotations.append(dict(
                x=groups[j],
                y=groups[i],
                text=star,
                showarrow=False,
                font=dict(size=12, color=text_color,
                          family='Arial', weight='bold'),
                xanchor='center',
                yanchor='middle'
            ))

    # Create custom hover text
    hover_text = []
    for i in range(n):
        row_text = []
        for j in range(n):
            if lower_triangular and i <= j:
                row_text.append("")
            elif i == j:
                row_text.append(f"{groups[i]}")
            else:
                text = (f"<b>{groups[i]} vs {groups[j]}</b><br>"
                        f"Mean Diff: {diff_matrix[i, j]:.3f}<br>"
                        f"p-adj: {p_matrix[i, j]:.4f}<br>"
                        f"95% CI: [{lower_ci[i, j]:.3f}, {upper_ci[i, j]:.3f}]")
                row_text.append(text)
        hover_text.append(row_text)

    # Use diverging colorscale (blue-white-red) for mean differences
    fig = go.Figure(data=go.Heatmap(
        z=diff_matrix_display,
        x=groups,
        y=groups,
        colorscale='RdBu_r',  # Red for positive, Blue for negative
        zmid=0,  # Center at 0
        zmin=-max_abs_diff,
        zmax=max_abs_diff,
        colorbar=dict(
            title=dict(text='Mean<br>Difference',
                       font=dict(size=12, color='black')),
            tickfont=dict(size=11, color='black'),
            thickness=18,
            len=0.9,
            x=1.02,
        ),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_text,
        showscale=True,
    ))

    fig.update_layout(annotations=annotations)

    fig = create_publication_layout(
        fig,
        title=title,
        xaxis_title="Group",
        yaxis_title="Group",
        show_legend=False,
        height=550,
        width=750
    )

    fig.update_xaxes(tickangle=0, side='bottom')
    fig.update_yaxes(autorange='reversed')

    # Add legend for significance stars
    fig.add_annotation(
        x=1.18, y=0.5,
        xref='paper', yref='paper',
        text="<b>Significance:</b><br>*** p<0.001<br>** p<0.01<br>* p<0.05<br>ns: not sig.",
        showarrow=False,
        font=dict(size=11, color='black'),
        align='left',
        bgcolor='rgba(255,255,255,0.95)',
        bordercolor='black',
        borderwidth=1,
        borderpad=8
    )

    fig.update_layout(margin=dict(r=200))

    return fig


def plot_temporal_trend(df, time_col, value_col):
    """Create temporal trend plot with confidence bands"""
    summary = df.groupby(time_col)[value_col].agg(
        ['mean', 'std', 'count']).reset_index()
    summary['se'] = summary['std'] / np.sqrt(summary['count'])
    summary['ci95'] = 1.96 * summary['se']

    if time_col == 'Month':
        summary[time_col] = pd.Categorical(
            summary[time_col], categories=MONTH_ORDER, ordered=True)
        summary = summary.sort_values(time_col)

    fig = go.Figure()

    x_vals = summary[time_col].astype(str)

    fig.add_trace(go.Scatter(
        x=list(x_vals) + list(x_vals)[::-1],
        y=list(summary['mean'] + summary['ci95']) +
        list(summary['mean'] - summary['ci95'])[::-1],
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% CI',
        showlegend=True,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=summary['mean'],
        mode='lines+markers',
        name='Mean',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10, color='#1f77b4',
                    line=dict(width=2, color='black')),
        hovertemplate=(
            f"<b>{time_col}: %{{x}}</b><br>"
            f"Mean: %{{y:.2f}}<br>"
            "<extra></extra>"
        )
    ))

    fig = create_publication_layout(
        fig,
        title=f"Temporal Trend of {value_col}",
        xaxis_title=time_col,
        yaxis_title=f"Mean {value_col}",
        show_legend=True,
        legend_position='right'
    )

    return fig


def plot_distribution_histogram(df, value_col, factor_col=None):
    """Create distribution histogram with proper legend"""
    fig = go.Figure()

    if factor_col:
        groups = sorted(df[factor_col].unique())
        colors = COLOR_PALETTES['primary']

        for i, group in enumerate(groups):
            group_data = df[df[factor_col] == group][value_col].dropna()
            fig.add_trace(go.Histogram(
                x=group_data,
                name=str(group),
                marker_color=colors[i % len(colors)],
                opacity=0.7,
                nbinsx=30,
                hovertemplate=f"<b>{factor_col}: {group}</b><br>{value_col}: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>"
            ))

        fig.update_layout(barmode='overlay')

        fig = create_publication_layout(
            fig,
            title=f"Distribution of {value_col} by {factor_col}",
            xaxis_title=value_col,
            yaxis_title="Frequency",
            show_legend=True,
            legend_position='right',
            legend_title=factor_col
        )
    else:
        fig.add_trace(go.Histogram(
            x=df[value_col].dropna(),
            marker_color='#1f77b4',
            opacity=0.8,
            nbinsx=30,
            hovertemplate=f"{value_col}: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>"
        ))

        fig = create_publication_layout(
            fig,
            title=f"Distribution of {value_col}",
            xaxis_title=value_col,
            yaxis_title="Frequency",
            show_legend=False
        )

    return fig


def style_dataframe(df, format_dict=None, highlight_col=None, highlight_condition=None):
    """Apply publication-quality styling to a pandas DataFrame"""

    # Create a copy for styling
    styled = df.style

    # Apply number formatting
    if format_dict:
        styled = styled.format(format_dict, na_rep='-')

    # Base styling
    styled = styled.set_properties(**{
        'text-align': 'center',
        'font-size': '14px',
        'border': '1px solid #ddd',
        'padding': '8px'
    })

    # Header styling
    styled = styled.set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#1f77b4'),
            ('color', 'white'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('font-size', '14px'),
            ('padding', '10px'),
            ('border', '1px solid #1f77b4')
        ]},
        {'selector': 'td', 'props': [
            ('text-align', 'center'),
            ('padding', '8px')
        ]},
        {'selector': 'tr:nth-child(even)', 'props': [
            ('background-color', '#f9f9f9')
        ]},
        {'selector': 'tr:hover', 'props': [
            ('background-color', '#e8f4fc')
        ]},
        {'selector': '', 'props': [
            ('border-collapse', 'collapse'),
            ('width', '100%')
        ]}
    ])

    # Row highlighting based on condition
    if highlight_col and highlight_condition:
        def highlight_row(row):
            try:
                if highlight_condition(row[highlight_col]):
                    return ['background-color: #d4edda'] * len(row)
            except:
                pass
            return [''] * len(row)
        styled = styled.apply(highlight_row, axis=1)

    # Highlight significant rows
    if 'Significant' in df.columns:
        def highlight_sig(row):
            try:
                if row['Significant'] == True:
                    return ['background-color: #d4edda'] * len(row)
            except:
                pass
            return [''] * len(row)
        styled = styled.apply(highlight_sig, axis=1)

    return styled


def display_styled_table(df, title="", format_dict=None, highlight_col=None,
                         highlight_condition=None, key=None):
    """Display a styled table with download option"""

    if title:
        st.markdown(f"**{title}**")

    # Apply styling
    styled = style_dataframe(
        df, format_dict, highlight_col, highlight_condition)

    # Display table
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Download button
    csv = df.to_csv(index=False)
    download_key = key or f"download_{title.replace(' ', '_')}_{id(df)}"
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{title.replace(' ', '_').lower() or 'table'}.csv",
        mime="text/csv",
        key=download_key
    )


def download_figure_as_svg(fig, filename, key):
    """Add SVG download button for a Plotly figure"""
    # Convert to SVG
    svg_bytes = fig.to_image(format="svg")

    st.download_button(
        label="Download SVG",
        data=svg_bytes,
        file_name=f"{filename}.svg",
        mime="image/svg+xml",
        key=key
    )


def display_figure_with_download(fig, filename, key_prefix):
    """Display a Plotly figure with SVG download option"""
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # Try to add SVG download (requires kaleido)
    try:
        svg_bytes = fig.to_image(format="svg")
        st.download_button(
            label="📥 Download SVG",
            data=svg_bytes,
            file_name=f"{filename}.svg",
            mime="image/svg+xml",
            key=f"{key_prefix}_{filename}"
        )
    except Exception as e:
        # Fallback to HTML download if kaleido not available
        html_str = fig.to_html(include_plotlyjs='cdn')
        st.download_button(
            label="📥 Download HTML",
            data=html_str,
            file_name=f"{filename}.html",
            mime="text/html",
            key=f"{key_prefix}_{filename}"
        )


def create_styled_table(df, title="", highlight_col=None, highlight_condition=None,
                        format_dict=None, height=400):
    """Create a publication-quality styled table using Plotly - DEPRECATED, use display_styled_table"""
    # Keep for backward compatibility but redirect to new function
    return _create_plotly_table(df, title, highlight_col, highlight_condition, format_dict, height)


def _create_plotly_table(df, title="", highlight_col=None, highlight_condition=None,
                         format_dict=None, height=400):
    """Create a publication-quality styled table using Plotly"""

    # Prepare data with formatting
    display_df = df.copy()
    if format_dict:
        for col, fmt in format_dict.items():
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: fmt.format(x) if pd.notna(x) and isinstance(x, (int, float)) else (
                        '-' if pd.isna(x) else str(x)
                    )
                )

    # Convert all columns to string for display
    for col in display_df.columns:
        display_df[col] = display_df[col].apply(
            lambda x: '-' if pd.isna(x) else str(x))

    # Define colors
    header_color = '#1f77b4'
    row_even_color = '#f9f9f9'
    row_odd_color = 'white'
    highlight_color = '#d4edda'
    sig_highlight_color = '#d4edda'

    # Create cell colors based on highlighting
    n_rows = len(display_df)
    n_cols = len(display_df.columns)

    cell_colors = []
    for j in range(n_cols):
        col_colors = []
        col_name = display_df.columns[j]
        for i in range(n_rows):
            base_color = row_even_color if i % 2 == 0 else row_odd_color

            # Apply highlight if condition met for specific column
            if highlight_col and highlight_condition and col_name == highlight_col:
                try:
                    val = df.iloc[i][highlight_col]
                    if pd.notna(val) and highlight_condition(val):
                        base_color = highlight_color
                except:
                    pass

            # Highlight entire row if 'Significant' column is True
            if 'Significant' in df.columns:
                try:
                    if df.iloc[i]['Significant'] == True:
                        base_color = sig_highlight_color
                except:
                    pass

            # Highlight row for Result column conditions
            if 'Result' in df.columns and highlight_col == 'Result':
                try:
                    val = df.iloc[i]['Result']
                    if highlight_condition and highlight_condition(val):
                        base_color = highlight_color
                except:
                    pass

            col_colors.append(base_color)
        cell_colors.append(col_colors)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{col}</b>' for col in display_df.columns],
            fill_color=header_color,
            align='center',
            font=dict(color='white', size=13, family='Arial'),
            height=40,
            line=dict(color='white', width=1)
        ),
        cells=dict(
            values=[display_df[col].tolist() for col in display_df.columns],
            fill_color=cell_colors,
            align='center',
            font=dict(color='black', size=12, family='Arial'),
            height=32,
            line=dict(color='#e0e0e0', width=1)
        )
    )])

    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b>' if title else '',
            font=dict(size=16, family='Arial', color='black'),
            x=0.5,
            xanchor='center',
            y=0.98
        ),
        margin=dict(l=20, r=20, t=60 if title else 20, b=20),
        height=height,
        paper_bgcolor='white'
    )

    return fig


def plot_nemenyi_heatmap(results_df, groups, cd, title="Nemenyi Post-hoc Test"):
    """Create heatmap for Nemenyi test results"""
    n = len(groups)

    # Create matrices
    diff_matrix = np.zeros((n, n))
    p_matrix = np.ones((n, n))

    group_to_idx = {g: i for i, g in enumerate(groups)}

    for _, row in results_df.iterrows():
        i = group_to_idx.get(row['Group1'])
        j = group_to_idx.get(row['Group2'])
        if i is not None and j is not None:
            diff_matrix[i, j] = row['Rank Diff']
            diff_matrix[j, i] = row['Rank Diff']
            p_matrix[i, j] = row['p-adj']
            p_matrix[j, i] = row['p-adj']

    # Lower triangular
    mask = np.triu(np.ones_like(diff_matrix, dtype=bool), k=0)
    diff_matrix_display = np.where(mask, np.nan, diff_matrix)

    max_diff = np.nanmax(diff_matrix_display) if np.nanmax(
        diff_matrix_display) > 0 else 1

    # Annotations
    annotations = []
    for i in range(n):
        for j in range(n):
            if i <= j:
                continue

            diff = diff_matrix[i, j]
            sig = diff > cd
            star = "**" if sig else "ns"
            text_color = 'white' if diff > max_diff * 0.5 else 'black'

            annotations.append(dict(
                x=groups[j], y=groups[i], text=star,
                showarrow=False,
                font=dict(size=12, color=text_color,
                          family='Arial', weight='bold'),
                xanchor='center', yanchor='middle'
            ))

    # Hover text
    hover_text = []
    for i in range(n):
        row_text = []
        for j in range(n):
            if i <= j:
                row_text.append("")
            else:
                text = (f"<b>{groups[i]} vs {groups[j]}</b><br>"
                        f"Rank Diff: {diff_matrix[i, j]:.3f}<br>"
                        f"CD: {cd:.3f}<br>"
                        f"Significant: {'Yes' if diff_matrix[i, j] > cd else 'No'}")
                row_text.append(text)
        hover_text.append(row_text)

    fig = go.Figure(data=go.Heatmap(
        z=diff_matrix_display,
        x=groups,
        y=groups,
        colorscale='YlOrRd',
        zmin=0,
        zmax=max_diff,
        colorbar=dict(
            title=dict(text='Rank<br>Difference',
                       font=dict(size=12, color='black')),
            tickfont=dict(size=11, color='black'),
            thickness=18,
            len=0.9,
            x=1.02,
        ),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_text,
        showscale=True,
    ))

    # Add CD line annotation
    fig.add_annotation(
        x=1.18, y=0.5,
        xref='paper', yref='paper',
        text=f"<b>CD = {cd:.3f}</b><br>** sig. diff.<br>ns: not sig.",
        showarrow=False,
        font=dict(size=11, color='black'),
        align='left',
        bgcolor='rgba(255,255,255,0.95)',
        bordercolor='black',
        borderwidth=1,
        borderpad=8
    )

    fig.update_layout(annotations=annotations)

    fig = create_publication_layout(
        fig, title=title, xaxis_title="Group", yaxis_title="Group",
        show_legend=False, height=550, width=750
    )
    fig.update_xaxes(tickangle=0, side='bottom')
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(margin=dict(r=200))

    return fig


# ========================================
# HTML Report Generation
# ========================================
def generate_html_report(df, factor_cols, value_col, analyses_results):
    """Generate comprehensive HTML report"""

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Statistical Analysis Report</title>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #1f77b4, #2ca02c);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .header h1 {{
                margin: 0;
                font-size: 28px;
            }}
            .header p {{
                margin: 10px 0 0 0;
                opacity: 0.9;
            }}
            .section {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .section h2 {{
                color: #1f77b4;
                border-bottom: 2px solid #1f77b4;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            .section h3 {{
                color: #2ca02c;
                margin-top: 20px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: center;
            }}
            th {{
                background-color: #1f77b4;
                color: white;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            tr:hover {{
                background-color: #e8f4f8;
            }}
            .highlight {{
                background-color: #d4edda;
                border-left: 4px solid #28a745;
                padding: 15px;
                margin: 15px 0;
            }}
            .warning {{
                background-color: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px;
                margin: 15px 0;
            }}
            .metric-box {{
                display: inline-block;
                background: #e8f4f8;
                padding: 15px 25px;
                border-radius: 8px;
                margin: 10px;
                text-align: center;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #1f77b4;
            }}
            .metric-label {{
                font-size: 12px;
                color: #666;
                margin-top: 5px;
            }}
            .footer {{
                text-align: center;
                color: #666;
                margin-top: 30px;
                padding: 20px;
                border-top: 1px solid #ddd;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Statistical Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>1. Dataset Overview</h2>
            <div class="metric-box">
                <div class="metric-value">{len(df)}</div>
                <div class="metric-label">Total Observations</div>
            </div>
    """

    for col in factor_cols:
        html_content += f"""
            <div class="metric-box">
                <div class="metric-value">{df[col].nunique()}</div>
                <div class="metric-label">{col} Levels</div>
            </div>
        """

    stats = df[value_col].describe()
    html_content += f"""
        </div>
        
        <div class="section">
            <h2>2. Response Variable: {value_col}</h2>
            <table>
                <tr>
                    <th>Statistic</th>
                    <th>Value</th>
                </tr>
                <tr><td>Count</td><td>{stats['count']:.0f}</td></tr>
                <tr><td>Mean</td><td>{stats['mean']:.3f}</td></tr>
                <tr><td>Std Dev</td><td>{stats['std']:.3f}</td></tr>
                <tr><td>CV (%)</td><td>{100*stats['std']/stats['mean']:.1f}</td></tr>
                <tr><td>Skewness</td><td>{skew(df[value_col].dropna()):.3f}</td></tr>
                <tr><td>Kurtosis</td><td>{kurtosis(df[value_col].dropna()):.3f}</td></tr>
                <tr><td>Min</td><td>{stats['min']:.3f}</td></tr>
                <tr><td>25%</td><td>{stats['25%']:.3f}</td></tr>
                <tr><td>Median</td><td>{stats['50%']:.3f}</td></tr>
                <tr><td>75%</td><td>{stats['75%']:.3f}</td></tr>
                <tr><td>Max</td><td>{stats['max']:.3f}</td></tr>
            </table>
        </div>
    """

    if analyses_results:
        html_content += """
        <div class="section">
            <h2>3. Statistical Analysis Results</h2>
        """

        for analysis_name, result in analyses_results.items():
            html_content += f"""
            <h3>{analysis_name}</h3>
            <div class="highlight">
                {result}
            </div>
            """

        html_content += "</div>"

    html_content += """
        <div class="section">
            <h2>4. Recommendations for Report</h2>
            <p>Include the following in your final report:</p>
            <ul>
                <li><strong>Figures:</strong> Box plots, bar charts with error bars, interaction plots</li>
                <li><strong>Tables:</strong> Summary statistics, ANOVA tables, post-hoc results</li>
                <li><strong>Statistical Tests:</strong> Assumption tests, ANOVA/Kruskal-Wallis, post-hoc comparisons</li>
            </ul>
        </div>
        
        <div class="footer">
            <p>Generated by Agricultural Experiment Statistical Analysis Tool v2.2</p>
        </div>
    </body>
    </html>
    """

    return html_content


# ========================================
# Main Application
# ========================================
def main():
    st.title("Agricultural Experiment Statistical Analysis")

    # Sidebar
    st.sidebar.title("Navigation")

    with st.sidebar.expander("Data Preparation Guide", expanded=False):
        st.markdown("""
        **Required Data Format:**
        
        Your CSV file should have:
        1. **Factor columns** (categorical):
           - Treatment/Technique (e.g., T1, T2...)
           - Time period (e.g., Month)
           - Blocking factor (e.g., Tree, Block)
        
        2. **Response column** (numeric):
           - Your measurement variable
        
        **Example Structure:**
        ```
        Technique,Month,Block,Yield
        T1,January,B1,10.5
        T2,January,B1,15.2
        ...
        ```
        """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="Upload your experimental data in CSV format"
    )

    # Load data
    df = None
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
    else:
        try:
            df = load_data('data.csv')
            st.sidebar.info("Using data.csv from project folder")
        except:
            st.sidebar.warning(
                "⚠️ No data loaded. Upload a file or place data.csv in project folder.")

    if df is None:
        st.warning(
            "Please upload a CSV file or ensure data.csv exists in the project folder.")
        st.stop()

    factor_cols, value_col = get_factor_columns(df)

    if not factor_cols or not value_col:
        st.error(
            "Could not detect factor and response columns. Please check your data format.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Info")
    st.sidebar.markdown(f"**Observations:** {len(df)}")
    st.sidebar.markdown(f"**Factors:** {', '.join(factor_cols)}")
    st.sidebar.markdown(f"**Response:** {value_col}")

    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Select Analysis",
        ["Data Summary", "Visualization",
            "Statistical Analysis", "Report Generation"]
    )

    # ========================================
    # PAGE 1: Data Summary
    # ========================================
    if page == "Data Summary":
        st.header("Data Summary")

        tab1, tab2, tab3 = st.tabs(
            ["Overview", "Summary Statistics", "Raw Data"])

        with tab1:
            st.subheader("Dataset Overview")

            cols = st.columns(len(factor_cols) + 2)
            cols[0].metric("Observations", len(df))
            for i, col in enumerate(factor_cols):
                cols[i+1].metric(f"{col} Levels", df[col].nunique())

            st.markdown("---")

            st.subheader(f"Response Variable: {value_col}")

            data = df[value_col].dropna()
            stats_data = {
                'Statistic': ['Count', 'Mean', 'Std Dev', 'CV (%)', 'Skewness', 'Kurtosis',
                              'Min', '25%', 'Median', '75%', 'Max'],
                'Value': [
                    f"{len(data):.0f}",
                    f"{data.mean():.3f}",
                    f"{data.std():.3f}",
                    f"{100*data.std()/data.mean():.1f}" if data.mean() != 0 else "N/A",
                    f"{skew(data):.3f}",
                    f"{kurtosis(data):.3f}",
                    f"{data.min():.3f}",
                    f"{data.quantile(0.25):.3f}",
                    f"{data.median():.3f}",
                    f"{data.quantile(0.75):.3f}",
                    f"{data.max():.3f}"
                ]
            }
            display_styled_table(
                pd.DataFrame(stats_data),
                title=f"Response Variable: {value_col}",
                key="stats_overview"
            )

        with tab2:
            st.subheader("Summary Statistics by Group")

            group_by = st.selectbox("Group by:", factor_cols + [
                                    f"{factor_cols[0]} × {factor_cols[1]}"] if len(factor_cols) > 1 else factor_cols)

            if " × " in group_by:
                parts = group_by.split(" × ")
                summary = compute_extended_statistics(df, parts, value_col)
            else:
                summary = compute_extended_statistics(
                    df, [group_by], value_col)

            format_dict = {col: '{:.3f}' for col in summary.select_dtypes(
                include=[np.number]).columns}
            display_styled_table(
                summary,
                title="Summary Statistics by Group",
                format_dict=format_dict,
                key="summary_stats"
            )

        with tab3:
            st.subheader("Raw Data")

            filter_cols = st.columns(len(factor_cols))
            filters = {}
            for i, col in enumerate(factor_cols):
                filters[col] = filter_cols[i].multiselect(
                    f"Filter {col}:", sorted(df[col].unique()))

            filtered_df = df.copy()
            for col, vals in filters.items():
                if vals:
                    filtered_df = filtered_df[filtered_df[col].isin(vals)]

            display_styled_table(
                filtered_df,
                title="Raw Data",
                key="raw_data"
            )
            st.caption(f"Showing {len(filtered_df)} of {len(df)} observations")

    # ========================================
    # PAGE 2: Visualization
    # ========================================
    elif page == "Visualization":
        st.header("Statistical Visualization")

        viz_options = [
            "Box Plot",
            "Violin Plot",
            "Bar Chart with Error Bars",
            "Interaction Plot",
            "Heatmap",
            "Temporal Trend",
            "Distribution Histogram"
        ]

        viz_type = st.selectbox("Select Visualization Type:", viz_options)

        st.markdown("---")

        if viz_type in ["Box Plot", "Violin Plot", "Bar Chart with Error Bars", "Distribution Histogram"]:
            selected_factor = st.selectbox("Select Factor:", factor_cols)

            if viz_type == "Box Plot":
                show_points = st.checkbox("Show all data points", value=True)
                fig = plot_boxplot(df, selected_factor, value_col, show_points)
            elif viz_type == "Violin Plot":
                fig = plot_violin(df, selected_factor, value_col)
            elif viz_type == "Bar Chart with Error Bars":
                fig = plot_bar_with_error(df, selected_factor, value_col)
            else:
                group_option = st.radio(
                    "Group by:", ["None"] + factor_cols, horizontal=True)
                factor = None if group_option == "None" else group_option
                fig = plot_distribution_histogram(df, value_col, factor)

        elif viz_type in ["Interaction Plot", "Heatmap"]:
            if len(factor_cols) >= 2:
                col1, col2 = st.columns(2)
                factor1 = col1.selectbox(
                    "Factor 1 (Lines/Rows):", factor_cols, index=0)
                remaining = [f for f in factor_cols if f != factor1]
                factor2 = col2.selectbox(
                    "Factor 2 (X-axis/Columns):", remaining, index=0)

                if viz_type == "Interaction Plot":
                    fig = plot_interaction(df, factor1, factor2, value_col)
                else:
                    fig = plot_heatmap(df, factor1, factor2, value_col)
            else:
                st.warning("Need at least 2 factors for this visualization.")
                st.stop()

        elif viz_type == "Temporal Trend":
            time_col = st.selectbox("Select Time Factor:", [f for f in factor_cols if 'month' in f.lower(
            ) or 'time' in f.lower() or 'date' in f.lower()] or factor_cols)
            fig = plot_temporal_trend(df, time_col, value_col)

        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # ========================================
    # PAGE 3: Statistical Analysis
    # ========================================
    elif page == "Statistical Analysis":
        st.header("Statistical Analysis")

        analysis_type = st.selectbox(
            "Select Analysis Type:",
            ["Assumption Testing", "ANOVA", "Non-parametric Tests"]
        )

        st.markdown("---")

        # ========================================
        # ASSUMPTION TESTING
        # ========================================
        if analysis_type == "Assumption Testing":
            st.subheader("ANOVA Assumptions")

            test_factor = st.selectbox(
                "Select Factor for Testing:", factor_cols)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 1. Normality (Shapiro-Wilk)")
                st.markdown("**H₀:** Data is normally distributed (p > 0.05)")

                overall_stat, overall_p, overall_interp = perform_normality_test(
                    df[value_col])

                results = [{'Group': 'Overall', 'W-stat': overall_stat,
                            'p-value': overall_p, 'Result': overall_interp}]

                for group in sorted(df[test_factor].unique()):
                    group_data = df[df[test_factor] == group][value_col]
                    stat, p, interp = perform_normality_test(group_data)
                    results.append(
                        {'Group': str(group), 'W-stat': stat, 'p-value': p, 'Result': interp})

                results_df = pd.DataFrame(results)

                def highlight_result(val):
                    if val == 'Normal':
                        return 'background-color: #d4edda'
                    elif val == 'Non-normal':
                        return 'background-color: #f8d7da'
                    return ''

                display_styled_table(
                    results_df,
                    title="Shapiro-Wilk Normality Test",
                    format_dict={'W-stat': '{:.4f}', 'p-value': '{:.4f}'},
                    highlight_col='Result',
                    highlight_condition=lambda x: x == 'Normal',
                    key="normality_test"
                )

                normality_violated = any(
                    r['Result'] == 'Non-normal' for r in results if r['Group'] != 'Overall')

            with col2:
                st.markdown("### 2. Homogeneity (Levene's)")
                st.markdown("**H₀:** Variances are equal (p > 0.05)")

                levene_results = []
                for factor in factor_cols:
                    groups = [group[value_col].dropna().values for _,
                              group in df.groupby(factor)]
                    stat, p, interp = perform_homogeneity_test(groups)
                    levene_results.append(
                        {'Factor': factor, 'Statistic': stat, 'p-value': p, 'Result': interp})

                levene_df = pd.DataFrame(levene_results)
                display_styled_table(
                    levene_df,
                    title="Levene's Homogeneity Test",
                    format_dict={'Statistic': '{:.4f}', 'p-value': '{:.4f}'},
                    highlight_col='Result',
                    highlight_condition=lambda x: x == 'Homogeneous',
                    key="levene_test"
                )

                homogeneity_violated = any(
                    r['Result'] == 'Heterogeneous' for r in levene_results)

            st.markdown("---")
            st.markdown("### Recommendations")

            if normality_violated or homogeneity_violated:
                st.warning("⚠️ One or more assumptions are violated!")

                st.markdown("**Suggested Actions:**")
                st.markdown(
                    "1. **Try Data Transformation** (see ANOVA section)")
                st.markdown(
                    "2. **Use Non-parametric Tests** (Kruskal-Wallis, Friedman)")

                if normality_violated:
                    st.markdown(
                        "   - Normality violated: Consider log, sqrt, or Box-Cox transformation")
                if homogeneity_violated:
                    st.markdown(
                        "   - Homogeneity violated: Use **Welch's ANOVA** (available in ANOVA section) or non-parametric tests")
            else:
                st.success(
                    "All assumptions are satisfied. Proceed with parametric ANOVA.")

        # ========================================
        # ANOVA
        # ========================================
        elif analysis_type == "ANOVA":
            st.subheader("Analysis of Variance (ANOVA)")

            data_type = st.radio(
                "Data Type:", ["Original Data", "Transformed Data"], horizontal=True)

            working_df = df.copy()
            transform_label = ""

            if data_type == "Transformed Data":
                st.markdown("### Data Transformation")

                transform_type = st.selectbox(
                    "Select Transformation:",
                    ["log", "sqrt", "log10", "boxcox"],
                    help="Choose a transformation to normalize the data"
                )

                min_val = df[value_col].min()
                if min_val <= 0:
                    constant = abs(min_val) + 1
                    st.info(
                        f"ℹ️ Adding constant ({constant}) to handle zeros/negative values")
                else:
                    constant = 0

                try:
                    transformed_vals, lmbda = apply_transformation(
                        df[value_col].values, transform_type, constant)
                    working_df['Transformed_' + value_col] = transformed_vals
                    working_value_col = 'Transformed_' + value_col
                    transform_label = f" ({transform_type} transformed)"

                    if transform_type == 'boxcox' and lmbda is not None:
                        st.info(f"Box-Cox λ = {lmbda:.4f}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original Data:**")
                        stat, p, interp = perform_normality_test(df[value_col])
                        st.markdown(f"Shapiro-Wilk p = {p:.4f} ({interp})")
                    with col2:
                        st.markdown("**Transformed Data:**")
                        stat, p, interp = perform_normality_test(
                            transformed_vals)
                        st.markdown(f"Shapiro-Wilk p = {p:.4f} ({interp})")

                    st.success("Transformation applied successfully!")
                except Exception as e:
                    st.error(f"Transformation failed: {str(e)}")
                    working_value_col = value_col
            else:
                working_value_col = value_col

            st.markdown("---")

            anova_type = st.radio(
                "ANOVA Type:", ["One-way ANOVA", "Welch's ANOVA", "Two-way ANOVA"], horizontal=True)

            if anova_type == "One-way ANOVA":
                factor = st.selectbox("Select Factor:", factor_cols)

                f_stat, p_value, anova_table, add_stats = perform_anova(
                    working_df, factor, working_value_col)

                if f_stat is not None:
                    # Display ANOVA Table
                    st.markdown("### ANOVA Table")
                    anova_df = pd.DataFrame(anova_table)

                    display_styled_table(
                        anova_df,
                        title="One-way ANOVA Table",
                        format_dict={
                            'SS': '{:.4f}', 'df': '{:.0f}', 'MS': '{:.4f}',
                            'F': '{:.4f}', 'p-value': '{:.6f}'
                        },
                        key="anova_table"
                    )

                    # Additional statistics
                    st.markdown("### Additional Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("η² (Eta-squared)",
                                f"{add_stats['eta_squared']:.4f}")
                    col2.metric("ω² (Omega-squared)",
                                f"{add_stats['omega_squared']:.4f}")
                    col3.metric("SEm", f"{add_stats['sem']:.4f}")
                    col4.metric("CD (5%)", f"{add_stats['cd']:.4f}")

                    col5, col6, col7, col8 = st.columns(4)
                    col5.metric("Grand Mean", f"{add_stats['grand_mean']:.4f}")
                    col6.metric("MSE", f"{add_stats['ms_within']:.4f}")
                    col7.metric("df (Error)", f"{add_stats['df_within']:.0f}")
                    col8.metric("k (Groups)", f"{add_stats['k']:.0f}")

                    if p_value < 0.05:
                        st.success(
                            f"**Significant effect** of {factor} on {value_col}{transform_label} (p < 0.05)")

                        st.markdown("---")
                        st.markdown("### Post-hoc Analysis (Tukey HSD)")

                        tukey_df, tukey_result, letters = perform_tukey_hsd(
                            working_df, factor, working_value_col)

                        if tukey_df is not None:
                            # Heatmap visualization (lower triangular)
                            st.markdown("#### Pairwise Comparison Heatmap")
                            groups = sorted(working_df[factor].unique())
                            fig_heatmap = plot_tukey_heatmap(
                                tukey_df, groups, lower_triangular=True)
                            st.plotly_chart(
                                fig_heatmap, use_container_width=True, config=PLOTLY_CONFIG)

                            # Significance letters
                            if letters:
                                st.markdown("### Compact Letter Display (CLD)")
                                st.info("""
                                **Interpretation:** Groups sharing the same letter are NOT significantly different from each other.
                                Groups with different letters ARE significantly different (p < 0.05).
                                """)

                                # Show letters with means
                                group_means = working_df.groupby(
                                    factor)[working_value_col].mean().sort_values(ascending=False)
                                letters_df = pd.DataFrame([
                                    {'Group': g, 'Mean': group_means[g], 'Letter': letters.get(
                                        g, '')}
                                    for g in group_means.index
                                ])
                                display_styled_table(
                                    letters_df,
                                    title="Compact Letter Display",
                                    format_dict={'Mean': '{:.3f}'},
                                    key="cld_oneway"
                                )

                                st.markdown(
                                    "### Bar Chart with Significance Letters")
                                fig = plot_bar_with_error(
                                    working_df, factor, working_value_col, letters)
                                st.plotly_chart(
                                    fig, use_container_width=True, config=PLOTLY_CONFIG)
                    else:
                        st.warning(
                            f"⚠️ **No significant effect** of {factor} on {value_col}{transform_label} (p ≥ 0.05)")
                        st.info(
                            "Post-hoc analysis is not needed when ANOVA is not significant.")

            elif anova_type == "Welch's ANOVA":
                st.markdown("### Welch's ANOVA")
                st.info("Use when homogeneity of variances assumption is violated.")

                factor = st.selectbox("Select Factor:", factor_cols)

                f_stat, p_value, welch_table, add_stats = perform_welch_anova(
                    working_df, factor, working_value_col)

                if f_stat is not None:
                    st.markdown("### Welch's ANOVA Results")
                    welch_df = pd.DataFrame(welch_table)
                    display_styled_table(
                        welch_df,
                        title="Welch's ANOVA Table",
                        format_dict={'df': '{:.2f}',
                                     'F': '{:.4f}', 'p-value': '{:.6f}'},
                        key="welch_table"
                    )

                    col1, col2 = st.columns(2)
                    col1.metric("Welch's F", f"{f_stat:.4f}")
                    col2.metric("p-value", f"{p_value:.6f}")

                    # Additional Statistics for Welch's ANOVA
                    st.markdown("### Additional Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("η² (Eta-squared)",
                                f"{add_stats['eta_squared']:.4f}")
                    col2.metric("ω² (Omega-squared)",
                                f"{add_stats['omega_squared']:.4f}")
                    col3.metric("Grand Mean", f"{add_stats['grand_mean']:.4f}")
                    col4.metric("k (Groups)", f"{add_stats['k']:.0f}")

                    # Group-wise statistics
                    st.markdown("#### Group Statistics")
                    group_stats_df = pd.DataFrame({
                        'Group': list(add_stats['group_means'].keys()),
                        'Mean': list(add_stats['group_means'].values()),
                        'Variance': list(add_stats['group_variances'].values()),
                        'n': list(add_stats['group_ns'].values())
                    })
                    display_styled_table(
                        group_stats_df,
                        title="Group Statistics",
                        format_dict={'Mean': '{:.4f}',
                                     'Variance': '{:.4f}', 'n': '{:.0f}'},
                        key="group_stats"
                    )

                    if p_value < 0.05:
                        st.success(
                            f"**Significant effect** of {factor} (p < 0.05)")

                        st.markdown("---")
                        st.markdown("### Post-hoc Analysis (Games-Howell)")
                        st.info(
                            "Games-Howell test is the appropriate post-hoc test for Welch's ANOVA as it does not assume equal variances.")

                        gh_df, letters = perform_games_howell(
                            working_df, factor, working_value_col)

                        if gh_df is not None:
                            # Display Games-Howell results table
                            st.markdown("#### Pairwise Comparisons")
                            display_gh_df = gh_df[[
                                'Group1', 'Group2', 'Mean Diff', 'SE', 'df', 't-stat', 'p-adj', 'Significant']].copy()

                            display_styled_table(
                                display_gh_df,
                                title="Games-Howell Pairwise Comparisons",
                                format_dict={
                                    'Mean Diff': '{:.4f}', 'SE': '{:.4f}', 'df': '{:.2f}',
                                    't-stat': '{:.4f}', 'p-adj': '{:.6f}'
                                },
                                key="gh_pairwise"
                            )

                            # Heatmap
                            groups = sorted(working_df[factor].unique())
                            fig_heatmap = plot_tukey_heatmap(
                                gh_df, groups, "Games-Howell Pairwise Comparisons", lower_triangular=True)
                            st.plotly_chart(
                                fig_heatmap, use_container_width=True, config=PLOTLY_CONFIG)

                            # Bar chart with letters
                            if letters:
                                st.markdown("### Compact Letter Display (CLD)")
                                group_means = working_df.groupby(
                                    factor)[working_value_col].mean().sort_values(ascending=False)
                                letters_df = pd.DataFrame([
                                    {'Group': g, 'Mean': group_means[g], 'Letter': letters.get(
                                        g, '')}
                                    for g in group_means.index
                                ])
                                display_styled_table(
                                    letters_df,
                                    title="Compact Letter Display",
                                    format_dict={'Mean': '{:.3f}'},
                                    key="cld_welch"
                                )

                                st.markdown(
                                    "### Bar Chart with Significance Letters")
                                fig = plot_bar_with_error(
                                    working_df, factor, working_value_col, letters)
                                st.plotly_chart(
                                    fig, use_container_width=True, config=PLOTLY_CONFIG)
                    else:
                        st.warning(
                            f"**No significant effect** of {factor} (p ≥ 0.05)")

            else:  # Two-way ANOVA
                if len(factor_cols) < 2:
                    st.warning("Need at least 2 factors for two-way ANOVA.")
                else:
                    col1, col2 = st.columns(2)
                    factor1 = col1.selectbox("Factor 1:", factor_cols, index=0)
                    remaining = [f for f in factor_cols if f != factor1]
                    factor2 = col2.selectbox("Factor 2:", remaining, index=0)

                    anova_table, model, add_stats = perform_two_way_anova(
                        working_df, factor1, factor2, working_value_col)

                    if anova_table is not None and add_stats is not None:
                        st.markdown("### ANOVA Table")

                        display_table = anova_table.reset_index()
                        display_table.columns = [
                            'Source', 'Sum Sq', 'df', 'F', 'p-value']

                        def highlight_sig_p(val):
                            if isinstance(val, (int, float)) and val < 0.05:
                                return 'background-color: #d4edda; font-weight: bold'
                            return ''

                        display_styled_table(
                            display_table,
                            title="Two-way ANOVA Table",
                            format_dict={
                                'Sum Sq': '{:.3f}', 'df': '{:.0f}', 'F': '{:.3f}', 'p-value': '{:.6f}'
                            },
                            highlight_col='p-value',
                            highlight_condition=lambda x: isinstance(
                                x, (int, float)) and x < 0.05,
                            key="twoway_anova"
                        )

                        # Additional Statistics for Two-way ANOVA
                        st.markdown("### Additional Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("R²", f"{add_stats['r_squared']:.4f}")
                        col2.metric(
                            "Adj. R²", f"{add_stats['adj_r_squared']:.4f}")
                        col3.metric("MSE", f"{add_stats['ms_error']:.4f}")
                        col4.metric(
                            "df (Error)", f"{add_stats['df_error']:.0f}")

                        col5, col6, col7, col8 = st.columns(4)
                        col5.metric(
                            "Grand Mean", f"{add_stats['grand_mean']:.4f}")
                        col6.metric("SEm", f"{add_stats['sem']:.4f}")
                        col7.metric("CD (5%)", f"{add_stats['cd']:.4f}")
                        col8.metric("N (Total)", f"{add_stats['n_total']:.0f}")

                        # Effect sizes table
                        st.markdown("#### Effect Sizes by Factor")
                        effect_data = []
                        for source, sizes in add_stats['effect_sizes'].items():
                            effect_data.append({
                                'Source': source,
                                'η² (Eta-squared)': sizes['eta_squared'],
                                'Partial η²': sizes['partial_eta_squared']
                            })
                        effect_df = pd.DataFrame(effect_data)
                        display_styled_table(
                            effect_df,
                            title="Effect Sizes by Factor",
                            format_dict={
                                'η² (Eta-squared)': '{:.4f}',
                                'Partial η²': '{:.4f}'
                            },
                            key="effect_sizes"
                        )

                        st.markdown("### Interpretation")
                        significant_factors = []
                        for idx, row in display_table.iterrows():
                            source = row['Source']
                            if source != 'Residual' and pd.notna(row['p-value']):
                                p = row['p-value']
                                if p < 0.001:
                                    sig = "highly significant (p < 0.001) ⭐⭐⭐"
                                    significant_factors.append(source)
                                elif p < 0.01:
                                    sig = "very significant (p < 0.01) ⭐⭐"
                                    significant_factors.append(source)
                                elif p < 0.05:
                                    sig = "significant (p < 0.05) ⭐"
                                    significant_factors.append(source)
                                else:
                                    sig = "not significant (p ≥ 0.05)"
                                st.markdown(f"- **{source}:** {sig}")

                        # Post-hoc for significant factors
                        if significant_factors:
                            st.markdown("---")
                            st.markdown(
                                "### Post-hoc Analysis for Significant Effects")

                            # Add selection for which factors to show
                            default_selection = [
                                f for f in significant_factors if '×' not in f]
                            selected_posthoc = st.multiselect(
                                "Select effects to analyze:",
                                significant_factors,
                                default=default_selection,
                                help="Choose which significant effects to show post-hoc analysis for"
                            )

                            posthoc_results = perform_two_way_posthoc(
                                working_df, factor1, factor2, working_value_col)

                            for source in selected_posthoc:
                                # Map source name to key in posthoc_results
                                if source == factor1:
                                    result_key = factor1
                                elif source == factor2:
                                    result_key = factor2
                                elif '×' in source:
                                    result_key = 'Interaction'
                                else:
                                    continue

                                if result_key in posthoc_results:
                                    st.markdown(f"#### {source}")
                                    result = posthoc_results[result_key]
                                    tukey_df = result['tukey']
                                    letters = result['letters']

                                    if tukey_df is not None:
                                        # Determine groups for heatmap
                                        if result_key == factor1:
                                            groups = sorted(
                                                working_df[factor1].unique())
                                            plot_df = working_df
                                            plot_factor = factor1
                                        elif result_key == factor2:
                                            groups = sorted(
                                                working_df[factor2].unique())
                                            plot_df = working_df
                                            plot_factor = factor2
                                        else:  # Interaction
                                            all_groups = set(
                                                tukey_df['Group1'].tolist() + tukey_df['Group2'].tolist())
                                            groups = sorted(list(all_groups))
                                            # Use the interaction dataframe
                                            plot_df = result.get(
                                                'df', working_df)
                                            plot_factor = 'Interaction'

                                        # Heatmap (lower triangular)
                                        fig = plot_tukey_heatmap(tukey_df, groups,
                                                                 f"Post-hoc: {source}", lower_triangular=True)
                                        st.plotly_chart(
                                            fig, use_container_width=True, config=PLOTLY_CONFIG)

                                        if letters:
                                            st.markdown(
                                                "**Compact Letter Display (CLD):**")
                                            st.info(
                                                "Groups sharing the same letter are NOT significantly different.")

                                            # Get means for proper ordering
                                            if result_key == 'Interaction':
                                                group_means = plot_df.groupby('Interaction')[
                                                    working_value_col].mean().sort_values(ascending=False)
                                            else:
                                                group_means = plot_df.groupby(
                                                    plot_factor)[working_value_col].mean().sort_values(ascending=False)

                                            letters_df = pd.DataFrame([
                                                {'Group': k, 'Mean': group_means.get(
                                                    k, 0), 'Letter': v}
                                                for k, v in sorted(letters.items(), key=lambda x: group_means.get(x[0], 0), reverse=True)
                                            ])
                                            display_styled_table(
                                                letters_df,
                                                title=f"CLD - {source}",
                                                format_dict={'Mean': '{:.3f}'},
                                                key=f"cld_twoway_{source.replace(' ', '_')}"
                                            )

                                            # Bar chart with significance letters
                                            st.markdown(
                                                f"**Visualization: {source}**")

                                            if result_key == 'Interaction':
                                                # Offer multiple visualization options for interactions
                                                viz_type = st.radio(
                                                    f"Select visualization type for {source}:",
                                                    ["Grouped Bar Chart", "Horizontal Dot Plot", "Vertical Dot Plot",
                                                     "Faceted Bar Chart", "Heatmap with CLD"],
                                                    horizontal=True,
                                                    key=f"viz_{source}"
                                                )

                                                if viz_type == "Grouped Bar Chart":
                                                    fig_bar = plot_interaction_grouped_bar(
                                                        working_df, factor1, factor2, working_value_col, letters)
                                                elif viz_type == "Horizontal Dot Plot":
                                                    fig_bar = plot_interaction_dot(
                                                        plot_df, factor1, factor2, working_value_col, letters, horizontal=True)
                                                elif viz_type == "Vertical Dot Plot":
                                                    fig_bar = plot_interaction_dot(
                                                        plot_df, factor1, factor2, working_value_col, letters, horizontal=False)
                                                elif viz_type == "Faceted Bar Chart":
                                                    fig_bar = plot_interaction_facet(
                                                        working_df, factor1, factor2, working_value_col, letters)
                                                else:  # Heatmap with CLD
                                                    fig_bar = plot_interaction_heatmap_enhanced(
                                                        working_df, factor1, factor2, working_value_col, letters)
                                            else:
                                                fig_bar = plot_bar_with_error(
                                                    plot_df, plot_factor, working_value_col, letters)
                                            st.plotly_chart(
                                                fig_bar, use_container_width=True, config=PLOTLY_CONFIG)
                    else:
                        st.error(f"Error: {model}")

        # ========================================
        # NON-PARAMETRIC TESTS
        # ========================================
        elif analysis_type == "Non-parametric Tests":
            st.subheader("Non-parametric Tests")
            st.info("Use when ANOVA assumptions are violated")

            test_type = st.radio("Select Test:", [
                                 "Kruskal-Wallis (One-way)", "Friedman (Repeated Measures)"], horizontal=True)

            if test_type == "Kruskal-Wallis (One-way)":
                st.markdown("""
                **Kruskal-Wallis H-test** is a non-parametric alternative to one-way ANOVA.
                It tests whether samples originate from the same distribution.
                """)

                factor = st.selectbox("Select Factor:", factor_cols)

                h_stat, p_value = perform_kruskal_wallis(df, factor, value_col)

                if h_stat is not None:
                    col1, col2 = st.columns(2)
                    col1.metric("H-statistic", f"{h_stat:.4f}")
                    col2.metric("p-value", f"{p_value:.6f}")

                    if p_value < 0.05:
                        st.success(
                            f"**Significant difference** among {factor} groups (p < 0.05)")

                        st.markdown("---")
                        st.markdown("### Post-hoc Analysis (Dunn's Test)")
                        st.info(
                            "Dunn's test is the appropriate post-hoc test for Kruskal-Wallis.")

                        dunn_result = perform_dunn_test(df, factor, value_col)

                        if dunn_result is not None:
                            st.markdown(
                                "**Pairwise p-values (Bonferroni corrected):**")

                            # Create lower triangular heatmap for Dunn's test
                            groups = list(dunn_result.columns)
                            n = len(groups)

                            # Create lower triangular mask
                            z_values = dunn_result.values.astype(float).copy()
                            mask = np.triu(np.ones_like(
                                z_values, dtype=bool), k=0)
                            z_lower = np.where(mask, np.nan, z_values)

                            fig_dunn = go.Figure(data=go.Heatmap(
                                z=z_lower,
                                x=[str(c) for c in groups],
                                y=[str(c) for c in groups],
                                colorscale=[[0, '#d62728'], [0.05, '#ff7f0e'], [
                                    0.1, '#ffbb78'], [1, '#2ca02c']],
                                zmin=0, zmax=1,
                                colorbar=dict(
                                    title=dict(
                                        text='p-value', font=dict(size=12, color='black')),
                                    tickfont=dict(size=11, color='black'),
                                    tickvals=[0, 0.01, 0.05, 0.1, 0.5, 1],
                                    ticktext=['0', '0.01', '0.05',
                                              '0.10', '0.50', '1.00'],
                                    x=1.02,
                                ),
                                hovertemplate='%{y} vs %{x}<br>p-value: %{z:.4f}<extra></extra>',
                            ))

                            # Add significance annotations (only lower triangle)
                            annotations = []
                            for i in range(n):
                                for j in range(n):
                                    if i > j:  # Only lower triangle
                                        val = z_values[i, j]
                                        star = "***" if val < 0.001 else "**" if val < 0.01 else "*" if val < 0.05 else "ns"
                                        text_color = 'white' if val < 0.05 else 'black'
                                        annotations.append(dict(
                                            x=str(groups[j]), y=str(groups[i]), text=star,
                                            showarrow=False,
                                            font=dict(
                                                size=11, color=text_color, weight='bold'),
                                            xanchor='center', yanchor='middle'
                                        ))

                            # Add legend annotation
                            annotations.append(dict(
                                x=1.18, y=0.5,
                                xref='paper', yref='paper',
                                text="<b>Significance:</b><br>*** p<0.001<br>** p<0.01<br>* p<0.05<br>ns: not sig.",
                                showarrow=False,
                                font=dict(size=10, color='black'),
                                align='left',
                                bgcolor='rgba(255,255,255,0.95)',
                                bordercolor='black',
                                borderwidth=1,
                                borderpad=6
                            ))

                            fig_dunn.update_layout(
                                annotations=annotations,
                                title=dict(text="<b>Dunn's Test Pairwise Comparisons</b>",
                                           font=dict(size=16, color='black', family='Arial'), x=0.5),
                                xaxis=dict(title=dict(text='Group', font=dict(size=14, color='black')),
                                           tickfont=dict(size=12, color='black'), linecolor='black', mirror=True),
                                yaxis=dict(title=dict(text='Group', font=dict(size=14, color='black')),
                                           tickfont=dict(size=12, color='black'), autorange='reversed',
                                           linecolor='black', mirror=True),
                                height=500,
                                margin=dict(r=180),
                                paper_bgcolor='white', plot_bgcolor='white'
                            )
                            st.plotly_chart(
                                fig_dunn, use_container_width=True, config=PLOTLY_CONFIG)

                            st.markdown("### Recommended Visualization")
                            fig = plot_boxplot(
                                df, factor, value_col, show_points=True)
                            st.plotly_chart(
                                fig, use_container_width=True, config=PLOTLY_CONFIG)
                    else:
                        st.warning(
                            f"⚠️ **No significant difference** among {factor} groups (p ≥ 0.05)")

            else:  # Friedman test
                st.markdown("""
                **Friedman test** is a non-parametric alternative to two-way ANOVA for repeated measures.
                """)

                if len(factor_cols) < 2:
                    st.warning(
                        "Need at least 2 factors (treatment and blocking factor)")
                else:
                    col1, col2 = st.columns(2)
                    treatment_factor = col1.selectbox(
                        "Treatment Factor:", factor_cols, index=0)
                    remaining = [
                        f for f in factor_cols if f != treatment_factor]
                    block_factor = col2.selectbox(
                        "Blocking Factor (Subject):", remaining, index=0)

                    stat, p_value, error = perform_friedman_test(
                        df, treatment_factor, block_factor, value_col)

                    if error:
                        st.error(f"Error: {error}")
                    elif stat is not None:
                        col1, col2 = st.columns(2)
                        col1.metric("χ² statistic", f"{stat:.4f}")
                        col2.metric("p-value", f"{p_value:.6f}")

                        if p_value < 0.05:
                            st.success(
                                f"**Significant effect** of {treatment_factor} (p < 0.05)")

                            st.markdown("---")
                            st.markdown("### Post-hoc Analysis (Nemenyi Test)")

                            nemenyi_df, letters, cd, rank_summary = perform_nemenyi_test(
                                df, treatment_factor, block_factor, value_col)

                            if nemenyi_df is not None:
                                st.info(
                                    f"**Critical Difference (CD) = {cd:.4f}** - Rank differences greater than CD are significant.")

                                # Rank summary table
                                st.markdown("#### Mean Ranks by Treatment")
                                display_styled_table(
                                    rank_summary,
                                    title="Mean Ranks",
                                    format_dict={'Mean Rank': '{:.3f}'},
                                    key="nemenyi_ranks"
                                )

                                # Pairwise comparison table
                                st.markdown("#### Pairwise Comparisons")
                                display_nemenyi = nemenyi_df[[
                                    'Group1', 'Group2', 'Rank Diff', 'CD', 'p-adj', 'Significant']].copy()
                                display_styled_table(
                                    display_nemenyi,
                                    title="Nemenyi Pairwise Comparisons",
                                    format_dict={'Rank Diff': '{:.4f}',
                                                 'CD': '{:.4f}', 'p-adj': '{:.6f}'},
                                    key="nemenyi_pairwise"
                                )

                                # Heatmap visualization
                                st.markdown("#### Rank Difference Heatmap")
                                groups = list(rank_summary['Group'])
                                fig_heatmap = plot_nemenyi_heatmap(
                                    nemenyi_df, groups, cd)
                                st.plotly_chart(
                                    fig_heatmap, use_container_width=True, config=PLOTLY_CONFIG)

                                # CLD letters
                                if letters:
                                    st.markdown(
                                        "#### Compact Letter Display (CLD)")
                                    st.info(
                                        "Groups sharing the same letter are NOT significantly different.")

                                    letters_df = pd.DataFrame([
                                        {'Group': g, 'Mean Rank': rank_summary[rank_summary['Group'] == g]['Mean Rank'].values[0],
                                         'Letter': letters.get(g, '')}
                                        for g in rank_summary['Group']
                                    ])
                                    display_styled_table(
                                        letters_df,
                                        title="Significance Groups",
                                        format_dict={'Mean Rank': '{:.3f}'},
                                        key="nemenyi_cld"
                                    )

                                # Bar chart with letters
                                st.markdown(
                                    "#### Bar Chart with Significance Letters")
                                # Create summary for bar chart using mean values
                                group_means = df.groupby(treatment_factor)[value_col].agg(
                                    ['mean', 'std', 'count']).reset_index()
                                group_means['se'] = group_means['std'] / \
                                    np.sqrt(group_means['count'])
                                group_means = group_means.sort_values(
                                    treatment_factor)

                                fig_bar = plot_bar_with_error(
                                    df, treatment_factor, value_col, letters)
                                st.plotly_chart(
                                    fig_bar, use_container_width=True, config=PLOTLY_CONFIG)
                            else:
                                st.warning("Could not perform Nemenyi test.")

                            st.markdown("### Interaction Visualization")
                            fig = plot_interaction(
                                df, treatment_factor, block_factor, value_col)
                            st.plotly_chart(
                                fig, use_container_width=True, config=PLOTLY_CONFIG)
                        else:
                            st.warning(
                                f"⚠️ **No significant effect** of {treatment_factor} (p ≥ 0.05)")

    # ========================================
    # PAGE 4: Report Generation
    # ========================================
    elif page == "Report Generation":
        st.header("Report Generation")

        st.markdown("""
        Generate a comprehensive HTML report of your analysis.
        
        ### Report Contents Guide
        
        **Your report should include:**
        
        1. **Introduction** - Experimental design description, Objectives
        2. **Data Summary** - Sample sizes, Summary statistics table (Mean, SD, SE)
        3. **Visualizations** - Box plots, Bar charts with error bars, Interaction plots, Heatmaps
        4. **Statistical Analysis** - Assumption tests, ANOVA table with SEm & CD, Post-hoc comparisons
        5. **Conclusions** - Key findings, Recommendations
        """)

        st.markdown("---")

        analyses_results = {}

        if st.button("Generate HTML Report"):
            with st.spinner("Generating report..."):
                for factor in factor_cols[:2]:
                    f_stat, p_val, anova_table, add_stats = perform_anova(
                        df, factor, value_col)
                    if f_stat:
                        sig = "Significant" if p_val < 0.05 else "Not significant"
                        analyses_results[f"One-way ANOVA ({factor})"] = (
                            f"F = {f_stat:.3f}, p = {p_val:.6f}, η² = {add_stats['eta_squared']:.3f}, ω² = {add_stats['omega_squared']:.3f} ({sig})<br>"
                            f"SEm = {add_stats['sem']:.4f}, CD (5%) = {add_stats['cd']:.4f}"
                        )

                html_report = generate_html_report(
                    df, factor_cols, value_col, analyses_results)

                st.download_button(
                    label="Download HTML Report",
                    data=html_report,
                    file_name=f"statistical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )

                st.success("Report generated successfully!")

                st.markdown("### Preview")
                st.components.v1.html(html_report, height=600, scrolling=True)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
    <small>
    <b>Download Options:</b><br>
    • Tables: Click "Download CSV" below each table<br>
    • Figures: Click camera icon (📷) in figure toolbar → SVG<br>
    </small>
    """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
    <small>
    <b>Agricultural Experiment Analysis v2.3</b><br>
    Developed by  
    <a href="https://scholar.google.com/citations?user=Es-kJk4AAAAJ&hl=en" target="_blank">
        Dr. Sandip Garai
    </a>
    &nbsp;·&nbsp;
    <a href="https://scholar.google.com/citations?user=0dQ7Sf8AAAAJ&hl=en&oi=ao" target="_blank">
        Dr. Kanaka K K
    </a><br>
    📧 <a href="mailto:drgaraislab@gmail.com">Contact</a>
    </small>
    """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
