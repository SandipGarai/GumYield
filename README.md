# Agricultural Experiment Statistical Analysis Application

## Version 2.2

A comprehensive Streamlit application for analyzing factorial agricultural experiments with publication-ready visualizations and rigorous statistical testing.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Data Format](#data-format)
5. [Statistical Methods](#statistical-methods)
   - [Descriptive Statistics](#descriptive-statistics)
   - [Normality Testing](#normality-testing)
   - [Homogeneity of Variances](#homogeneity-of-variances)
   - [One-way ANOVA](#one-way-anova)
   - [Welch's ANOVA](#welchs-anova)
   - [Two-way ANOVA](#two-way-anova)
   - [Post-hoc Tests](#post-hoc-tests)
   - [Non-parametric Tests](#non-parametric-tests)
   - [Effect Sizes](#effect-sizes)
6. [Visualizations](#visualizations)
7. [Usage Guide](#usage-guide)
8. [Authors](#authors)

---

## Overview

This application provides a user-friendly interface for performing comprehensive statistical analysis of agricultural experiments. It supports various experimental designs including completely randomized designs (CRD), randomized complete block designs (RCBD), and factorial experiments.

---

## Features

- **Data Summary**: Extended statistics including skewness, kurtosis, and coefficient of variation
- **Publication-ready Visualizations**: Box plots, violin plots, bar charts, heatmaps, interaction plots
- **ANOVA Analysis**: One-way, Two-way, and Welch's ANOVA with assumption testing
- **Post-hoc Tests**: Tukey HSD, Games-Howell, Dunn's test with Compact Letter Display (CLD)
- **Non-parametric Alternatives**: Kruskal-Wallis and Friedman tests
- **Data Transformations**: Log, sqrt, Box-Cox transformations
- **HTML Report Generation**: Comprehensive downloadable reports

---

## Installation

```bash
# Install required packages
pip install streamlit pandas numpy plotly scipy statsmodels

# Run the application
streamlit run agri_stats_app.py
```

---

## Data Format

Your CSV file should contain:

| Column Type                  | Description                            | Example                 |
| ---------------------------- | -------------------------------------- | ----------------------- |
| Factor columns (categorical) | Treatment groups, blocks, time periods | Technique, Month, Block |
| Response column (numeric)    | Measurement variable                   | Yield, GumYield         |

**Example structure:**

```
Technique,Month,Block,Yield
T1,January,B1,10.5
T2,January,B1,15.2
T1,February,B1,12.3
...
```

---

## Statistical Methods

### Descriptive Statistics

#### Mean (Arithmetic Average)

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

#### Standard Deviation (Sample)

$$s = \sqrt{\frac{\sum_{i=1}^{n}(x_i - \bar{x})^2}{n-1}}$$

#### Standard Error of Mean (SEM)

$$SE_{\bar{x}} = \frac{s}{\sqrt{n}}$$

#### Coefficient of Variation

$$CV\% = \frac{s}{\bar{x}} \times 100$$

#### Skewness (Fisher's)

$$g_1 = \frac{m_3}{m_2^{3/2}} = \frac{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^3}{\left[\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2\right]^{3/2}}$$

#### Kurtosis (Excess)

$$g_2 = \frac{m_4}{m_2^2} - 3 = \frac{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^4}{\left[\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2\right]^2} - 3$$

---

### Normality Testing

#### Shapiro-Wilk Test

The Shapiro-Wilk test statistic is:

$$W = \frac{\left(\sum_{i=1}^{n} a_i x_{(i)}\right)^2}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

Where:

- $x_{(i)}$ is the $i$-th order statistic
- $a_i$ are constants generated from the means, variances, and covariances of the order statistics
- $\bar{x}$ is the sample mean

**Hypotheses:**

- $H_0$: Data is normally distributed
- $H_1$: Data is not normally distributed

**Decision Rule:** Reject $H_0$ if $p < \alpha$ (typically $\alpha = 0.05$)

---

### Homogeneity of Variances

#### Levene's Test

Levene's test statistic is based on the absolute deviations from group means:

$$W = \frac{(N-k)}{(k-1)} \cdot \frac{\sum_{i=1}^{k} n_i (\bar{Z}_{i\cdot} - \bar{Z}_{\cdot\cdot})^2}{\sum_{i=1}^{k}\sum_{j=1}^{n_i}(Z_{ij} - \bar{Z}_{i\cdot})^2}$$

Where:

- $Z_{ij} = |Y_{ij} - \bar{Y}_{i\cdot}|$ (using mean) or $Z_{ij} = |Y_{ij} - \tilde{Y}_{i\cdot}|$ (using median)
- $\bar{Z}_{i\cdot}$ is the mean of group $i$'s $Z$ values
- $\bar{Z}_{\cdot\cdot}$ is the overall mean of all $Z$ values
- $N$ is total sample size, $k$ is number of groups

**Hypotheses:**

- $H_0$: $\sigma_1^2 = \sigma_2^2 = \cdots = \sigma_k^2$ (homogeneous variances)
- $H_1$: At least one $\sigma_i^2 \neq \sigma_j^2$

---

### One-way ANOVA

Analysis of Variance partitions total variability into between-group and within-group components.

#### ANOVA Model

$$Y_{ij} = \mu + \tau_i + \epsilon_{ij}$$

Where:

- $Y_{ij}$ is the $j$-th observation in group $i$
- $\mu$ is the overall mean
- $\tau_i$ is the effect of treatment $i$
- $\epsilon_{ij} \sim N(0, \sigma^2)$ is the random error

#### Sum of Squares

**Total Sum of Squares:**
$$SS_{Total} = \sum_{i=1}^{k}\sum_{j=1}^{n_i}(Y_{ij} - \bar{Y}_{\cdot\cdot})^2$$

**Between-Groups Sum of Squares:**
$$SS_{Between} = \sum_{i=1}^{k} n_i (\bar{Y}_{i\cdot} - \bar{Y}_{\cdot\cdot})^2$$

**Within-Groups Sum of Squares:**
$$SS_{Within} = \sum_{i=1}^{k}\sum_{j=1}^{n_i}(Y_{ij} - \bar{Y}_{i\cdot})^2 = SS_{Total} - SS_{Between}$$

#### Degrees of Freedom

| Source         | df      |
| -------------- | ------- |
| Between Groups | $k - 1$ |
| Within Groups  | $N - k$ |
| Total          | $N - 1$ |

#### Mean Squares

$$MS_{Between} = \frac{SS_{Between}}{k-1}$$

$$MS_{Within} = \frac{SS_{Within}}{N-k}$$

#### F-Statistic

$$F = \frac{MS_{Between}}{MS_{Within}}$$

Under $H_0$: $F \sim F_{k-1, N-k}$

#### Standard Error of Mean (Pooled)

$$SE_m = \sqrt{\frac{MS_{Within}}{\bar{n}}}$$

Where $\bar{n}$ is the harmonic mean of group sizes.

#### Critical Difference (CD) at 5%

$$CD = t_{\alpha/2, df_{within}} \times SE_m \times \sqrt{2}$$

---

### Welch's ANOVA

Welch's ANOVA is robust to heterogeneous variances (does not assume equal variances).

#### Weights

$$w_i = \frac{n_i}{s_i^2}$$

#### Weighted Grand Mean

$$\bar{Y}_w = \frac{\sum_{i=1}^{k} w_i \bar{Y}_i}{\sum_{i=1}^{k} w_i}$$

#### Welch's F-Statistic

$$F_W = \frac{\frac{1}{k-1}\sum_{i=1}^{k} w_i (\bar{Y}_i - \bar{Y}_w)^2}{1 + \frac{2(k-2)}{k^2-1}\sum_{i=1}^{k}\frac{(1-w_i/\sum w_j)^2}{n_i-1}}$$

#### Degrees of Freedom

$$df_1 = k - 1$$

$$df_2 = \frac{1}{3\sum_{i=1}^{k}\frac{(1-w_i/\sum w_j)^2}{n_i-1}/(k^2-1)}$$

---

### Two-way ANOVA

For factorial experiments with two factors.

#### Model

$$Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk}$$

Where:

- $\alpha_i$ is the effect of factor A at level $i$
- $\beta_j$ is the effect of factor B at level $j$
- $(\alpha\beta)_{ij}$ is the interaction effect
- $\epsilon_{ijk} \sim N(0, \sigma^2)$

#### Sum of Squares Decomposition

$$SS_{Total} = SS_A + SS_B + SS_{A \times B} + SS_{Error}$$

**Factor A:**
$$SS_A = bn\sum_{i=1}^{a}(\bar{Y}_{i\cdot\cdot} - \bar{Y}_{\cdot\cdot\cdot})^2$$

**Factor B:**
$$SS_B = an\sum_{j=1}^{b}(\bar{Y}_{\cdot j\cdot} - \bar{Y}_{\cdot\cdot\cdot})^2$$

**Interaction:**
$$SS_{A \times B} = n\sum_{i=1}^{a}\sum_{j=1}^{b}(\bar{Y}_{ij\cdot} - \bar{Y}_{i\cdot\cdot} - \bar{Y}_{\cdot j\cdot} + \bar{Y}_{\cdot\cdot\cdot})^2$$

#### ANOVA Table

| Source   | SS        | df           | MS        | F              |
| -------- | --------- | ------------ | --------- | -------------- |
| Factor A | $SS_A$    | $a-1$        | $MS_A$    | $MS_A/MS_E$    |
| Factor B | $SS_B$    | $b-1$        | $MS_B$    | $MS_B/MS_E$    |
| A × B    | $SS_{AB}$ | $(a-1)(b-1)$ | $MS_{AB}$ | $MS_{AB}/MS_E$ |
| Error    | $SS_E$    | $ab(n-1)$    | $MS_E$    |                |
| Total    | $SS_T$    | $abn-1$      |           |                |

---

### Post-hoc Tests

#### Tukey's Honestly Significant Difference (HSD)

For comparing all pairs of means after significant ANOVA.

$$HSD = q_{\alpha, k, df_E} \times \sqrt{\frac{MS_E}{n}}$$

Where $q_{\alpha, k, df_E}$ is the studentized range statistic.

**Pairwise Comparison:**
$$|\bar{Y}_i - \bar{Y}_j| > HSD \Rightarrow \text{Significant difference}$$

**95% Confidence Interval:**
$$(\bar{Y}_i - \bar{Y}_j) \pm q_{\alpha, k, df_E} \times \sqrt{\frac{MS_E}{2}\left(\frac{1}{n_i} + \frac{1}{n_j}\right)}$$

#### Games-Howell Test

For post-hoc comparisons when variances are unequal (use after Welch's ANOVA).

**Standard Error of Difference:**
$$SE_{ij} = \sqrt{\frac{s_i^2}{n_i} + \frac{s_j^2}{n_j}}$$

**Welch-Satterthwaite Degrees of Freedom:**
$$df_{ij} = \frac{\left(\frac{s_i^2}{n_i} + \frac{s_j^2}{n_j}\right)^2}{\frac{(s_i^2/n_i)^2}{n_i-1} + \frac{(s_j^2/n_j)^2}{n_j-1}}$$

**Test Statistic:**
$$t = \frac{\bar{Y}_i - \bar{Y}_j}{SE_{ij}}$$

**p-value with Bonferroni adjustment:**
$$p_{adj} = \min\left(p \times \frac{k(k-1)}{2}, 1\right)$$

#### Compact Letter Display (CLD)

Groups sharing the same letter are NOT significantly different from each other.

**Algorithm:**

1. Order groups by mean (descending)
2. Build significance matrix based on pairwise comparisons
3. Assign letters to maximal sets of non-significantly different groups

---

### Non-parametric Tests

#### Kruskal-Wallis H-Test

Non-parametric alternative to one-way ANOVA (does not assume normality).

**H-Statistic:**
$$H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)$$

Where:

- $R_i$ is the sum of ranks in group $i$
- $n_i$ is the sample size of group $i$
- $N = \sum n_i$ is total sample size

Under $H_0$: $H \sim \chi^2_{k-1}$ (approximately)

#### Dunn's Test

Post-hoc test for Kruskal-Wallis.

**Z-statistic for groups $i$ and $j$:**
$$z_{ij} = \frac{\bar{R}_i - \bar{R}_j}{\sqrt{\frac{N(N+1)}{12}\left(\frac{1}{n_i} + \frac{1}{n_j}\right)}}$$

Where $\bar{R}_i$ is the mean rank of group $i$.

**Bonferroni-adjusted p-value:**
$$p_{adj} = \min\left(2(1-\Phi(|z|)) \times \frac{k(k-1)}{2}, 1\right)$$

#### Friedman Test

Non-parametric alternative to two-way ANOVA for repeated measures.

**Friedman Statistic:**
$$\chi^2_r = \frac{12}{nk(k+1)} \sum_{j=1}^{k} R_j^2 - 3n(k+1)$$

Where:

- $n$ is the number of blocks (subjects)
- $k$ is the number of treatments
- $R_j$ is the sum of ranks for treatment $j$

Under $H_0$: $\chi^2_r \sim \chi^2_{k-1}$

---

### Effect Sizes

#### Eta-squared ($\eta^2$)

Proportion of total variance explained by the factor.

$$\eta^2 = \frac{SS_{Between}}{SS_{Total}}$$

**Interpretation:**

- Small: $\eta^2 \approx 0.01$
- Medium: $\eta^2 \approx 0.06$
- Large: $\eta^2 \approx 0.14$

#### Omega-squared ($\omega^2$)

Less biased estimate of effect size (corrects for sample size).

$$\omega^2 = \frac{SS_{Between} - (k-1) \times MS_{Within}}{SS_{Total} + MS_{Within}}$$

#### Partial Eta-squared ($\eta_p^2$)

For factorial designs, proportion of variance explained by a factor controlling for other factors.

$$\eta_p^2 = \frac{SS_{Effect}}{SS_{Effect} + SS_{Error}}$$

#### R-squared ($R^2$)

For regression/ANOVA models:

$$R^2 = 1 - \frac{SS_{Error}}{SS_{Total}}$$

#### Adjusted R-squared

$$R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

Where $p$ is the number of predictors.

---

### Data Transformations

When assumptions are violated, transformations can help normalize data.

#### Log Transformation

$$Y' = \ln(Y + c)$$
Use for right-skewed data, count data, or when variance increases with mean.

#### Square Root Transformation

$$Y' = \sqrt{Y + c}$$
Use for count data or Poisson-distributed data.

#### Box-Cox Transformation

$$Y' = \begin{cases} \frac{Y^\lambda - 1}{\lambda} & \lambda \neq 0 \\ \ln(Y) & \lambda = 0 \end{cases}$$

The optimal $\lambda$ is estimated by maximum likelihood.

---

## Visualizations

### Box Plot

Shows distribution through quartiles with outliers.

### Violin Plot

Combines box plot with kernel density estimation.

### Bar Chart with Error Bars

Displays means with standard error bars and significance letters.

### Interaction Plot

Shows factor interactions with connected means.

### Heatmap

Color-coded matrix showing means or p-values.

- **Lower triangular only**: Avoids redundant information in symmetric matrices.

### Temporal Trend Plot

Time series with confidence bands.

---

## Usage Guide

### Step 1: Data Upload

Upload your CSV file or place `data.csv` in the application directory.

### Step 2: Assumption Testing

Navigate to "Statistical Analysis" → "Assumption Testing"

- Check normality with Shapiro-Wilk test
- Check homogeneity with Levene's test

### Step 3: Choose Analysis Type

Based on assumption results:

| Assumptions Met      | Recommended Analysis            |
| -------------------- | ------------------------------- |
| Both satisfied       | One-way or Two-way ANOVA        |
| Homogeneity violated | Welch's ANOVA with Games-Howell |
| Normality violated   | Kruskal-Wallis with Dunn's test |
| Both violated        | Non-parametric tests            |

### Step 4: Interpret Results

- Check p-values (significant if p < 0.05)
- Review effect sizes
- Examine post-hoc comparisons with CLD letters

### Step 5: Generate Report

Navigate to "Report Generation" for downloadable HTML report.

---

## Formulas Quick Reference

| Statistic | Formula                                           |
| --------- | ------------------------------------------------- |
| Mean      | $\bar{x} = \frac{\sum x_i}{n}$                    |
| Variance  | $s^2 = \frac{\sum(x_i - \bar{x})^2}{n-1}$         |
| SE        | $SE = \frac{s}{\sqrt{n}}$                         |
| F-ratio   | $F = \frac{MS_{Between}}{MS_{Within}}$            |
| η²        | $\eta^2 = \frac{SS_B}{SS_T}$                      |
| ω²        | $\omega^2 = \frac{SS_B - (k-1)MS_W}{SS_T + MS_W}$ |
| CD        | $CD = t_{\alpha/2} \times SE_m \times \sqrt{2}$   |

---

## Authors

**Dr. Sandip Garai**  
[Google Scholar](https://scholar.google.com/citations?user=Es-kJk4AAAAJ&hl=en)

**Dr. Kanaka K K**  
[Google Scholar](https://scholar.google.com/citations?user=0dQ7Sf8AAAAJ&hl=en&oi=ao)

📧 Contact: [drgaraislab@gmail.com](mailto:drgaraislab@gmail.com)

---

## License

This software is provided for academic and research purposes.

---

## Version History

| Version | Date | Changes                                                                                           |
| ------- | ---- | ------------------------------------------------------------------------------------------------- |
| 2.2     | 2025 | Added Games-Howell test, lower triangular heatmaps, additional statistics, interaction bar charts |
| 2.1     | 2025 | Fixed CLD algorithm, improved visualizations                                                      |
| 2.0     | 2025 | Added two-way ANOVA, non-parametric tests                                                         |
| 1.0     | 2024 | Initial release                                                                                   |

---

## References

1. Welch, B. L. (1951). On the comparison of several mean values: an alternative approach. _Biometrika_, 38(3/4), 330-336.

2. Games, P. A., & Howell, J. F. (1976). Pairwise multiple comparison procedures with unequal N's and/or variances. _Journal of Educational Statistics_, 1(2), 113-125.

3. Tukey, J. W. (1949). Comparing individual means in the analysis of variance. _Biometrics_, 5(2), 99-114.

4. Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. _Journal of the American Statistical Association_, 47(260), 583-621.

5. Dunn, O. J. (1964). Multiple comparisons using rank sums. _Technometrics_, 6(3), 241-252.

6. Friedman, M. (1937). The use of ranks to avoid the assumption of normality implicit in the analysis of variance. _Journal of the American Statistical Association_, 32(200), 675-701.
