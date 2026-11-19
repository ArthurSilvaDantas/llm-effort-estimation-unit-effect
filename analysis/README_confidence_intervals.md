# Confidence intervals for the geometric mean ratio

This addition reports a pointwise, two-sided 95% confidence interval for each
main-analysis model's geometric mean ratio R. Confidence intervals supplement
the original analysis and were added after collection; they must not be described
as part of the original analysis plan unless a corresponding amendment exists.

## Estimand and calculation

For each valid matched pair, retain the implemented log difference:

`L_im = ln(Y_WH,im) - ln(Y_WDh,im)`

Here, `Y_WDh = Y_WD * C_im`. The reported estimate is `R = exp(mean(L))`.
Its target is `exp(mu_L)`, the population geometric mean ratio under the
analysis model. It is not the arithmetic mean of the individual ratios.

For a model with `n` valid pairs, compute:

```text
standard_error = sample_sd(L) / sqrt(n)
critical_value = t.ppf(0.975, df=n - 1)
lower_mean_L = mean(L) - critical_value * standard_error
upper_mean_L = mean(L) + critical_value * standard_error
R_ci_lower = exp(lower_mean_L)
R_ci_upper = exp(upper_mean_L)
```

The sample standard deviation uses `ddof=1`. Exponentiating both limits is
appropriate because the exponential function is strictly increasing. The result
is generally asymmetric on the R scale. Each model uses its own valid-pair count.

## Outputs and execution

To regenerate the complete analysis pipeline from the repository root, run:

```bash
python reproduce.py
```

To run the per-model stage alone, use `python scripts/analyze_results.py`.
That script regenerates its existing CSVs and figures from saved records.
It does not collect new model responses. The effect table at
`analysis/outputs/effects.csv` retains its existing columns and appends:

| Column | Meaning |
|---|---|
| `R_confidence_level` | Confidence coefficient, currently 0.95. |
| `R_ci_lower` | Lower confidence limit for the geometric mean ratio. |
| `R_ci_upper` | Upper confidence limit for the geometric mean ratio. |

The existing `analysis/outputs/figures/ratio_by_model.png` and `.pdf` now show
horizontal confidence intervals. Points and numeric labels still represent R.

## Interpretation and scope

An R of 0.70 corresponds to estimates in work-hours being 30% lower on the
geometric comparison. For a hypothetical interval of [0.65, 0.76], the equivalent
reduction interval is [24%, 35%]: reverse the bounds when applying `100 * (1 - R)`.
This is a proportional difference between estimates, not an accuracy measure.
It is also distinct from Cohen's standardized paired effect size, d_z.

The existing left-tailed hypothesis test and its p-values remain unchanged.
A two-sided 95% interval is used to describe precision; it is not the confidence
bound dual to a one-sided test at alpha=0.05. Do not replace the original
rejection rule with an interval-based rule. The intervals are pointwise per
model, not simultaneous 95% coverage across all ten models.

The Student t interval assumes independent paired log differences and normality
of those differences for exact coverage; otherwise its coverage is approximate.
It does not automatically correct dependence from shared project content,
selection caused by invalid pairs, or variability between repeated model runs.
The original complete-pair selection and hypothesis-testing procedure are retained.

Confidence intervals are reported separately for each included model.
Exploratory subgroup outputs retain their existing descriptive scope.

## Method references

- NIST/SEMATECH, [Confidence Limits for the Mean](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm).
- SciPy, [Student t distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html).
