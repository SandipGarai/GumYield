# Agricultural Experiment Statistical Analysis App (v2.1)

A comprehensive **Streamlit-based statistical analysis application** designed for **factorial agricultural experiments**.  
This tool enables researchers, students, and analysts to perform **publication-ready statistical analyses and visualizations** with minimal coding.

---

## Features

### Data Summary

- Automatic detection of **factor (categorical)** and **response (numeric)** variables
- Extended descriptive statistics:
  - Mean, Standard Deviation
  - Coefficient of Variation (CV%)
  - Skewness & Kurtosis
  - Quartiles and Range
- Group-wise and interaction summaries
- Downloadable summary tables (CSV)

---

### Visualization (Publication-Ready)

- Box plots (with jittered raw data points)
- Violin plots
- Bar charts with **SE bars** and **significance letters**
- Interaction plots
- Heatmaps (factor × factor)
- Temporal trend plots with **95% confidence intervals**
- Distribution histograms (grouped or ungrouped)

All plots are styled for **journal-quality output**.

---

### Statistical Analysis

#### Assumption Testing

- **Normality**: Shapiro–Wilk test (overall & by group)
- **Homogeneity of variance**: Levene’s test
- Automated recommendations when assumptions are violated

#### Parametric Tests

- **One-way ANOVA**
  - Effect size (η²)
  - SEm (Standard Error of Mean)
  - CD (Critical Difference at 5%)
- **Welch’s ANOVA** (heterogeneous variances)
- **Two-way ANOVA** with interaction effects

#### Post-hoc Analysis

- Tukey HSD
- Correct **Compact Letter Display (CLD)** algorithm
- Pairwise comparison heatmaps with significance stars

#### Non-Parametric Alternatives

- Kruskal–Wallis test + Dunn’s post-hoc (Bonferroni adjusted)
- Friedman test (repeated measures designs)

---

### Data Transformation

- Log
- Log10
- Square root
- Box–Cox (automatic λ estimation)
- Side-by-side normality comparison (before vs after)

---

### Report Generation

- Fully styled **HTML statistical report**
- Includes:
  - Dataset overview
  - Descriptive statistics
  - ANOVA summaries
  - Effect sizes
  - Recommendations for publication
- One-click download

---

## Data Format

Your CSV file should include:

### Required

- **One numeric response variable**
- **One or more categorical factors**

### Example

```csv
Technique,Month,Tree,GumYield
T1,January,T1,12.4
T2,January,T1,15.7
T1,February,T2,11.9
```
