"""
Análise conjunta dos pares válidos, com os mesmos critérios da análise por modelo.

Cada linha mantém o par do mesmo modelo e da mesma especificação. Os testes
são aplicados a todos os pares, sem calcular médias prévias por modelo ou
especificação. O teste t trata os diferentes pares como independentes;
modelos com mais pares válidos contribuem mais para o resultado agrupado.

Gera valid_pairs.csv, normality.csv, hypothesis_tests.csv e effects.csv
em analysis/outputs/pooled/.

Uso:
    python scripts/analyze_pooled_results.py
"""

import numpy as np
import pandas as pd
from scipy import stats

from analyze_results import (
    ALPHA,
    OUTPUT_DIR as MODEL_OUTPUT_DIR,
    classify_effect_size,
    load_valid_pairs,
)


OUTPUT_DIR = MODEL_OUTPUT_DIR / "pooled"


def build_pooled_summaries(
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calcula normalidade, teste t e medidas de efeito do conjunto completo."""
    if len(pairs) < 3:
        raise ValueError("A análise conjunta com Shapiro-Wilk exige pelo menos 3 pares.")
    if pairs.duplicated(["model_id", "spec_id"]).any():
        raise ValueError("Há pares duplicados para o mesmo modelo e especificação.")

    values = pairs[["Y_WH", "Y_WDh"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("As estimativas devem ser finitas e estritamente positivas.")

    # Cada posição nas duas colunas corresponde ao mesmo par original.
    log_wh = np.log(values[:, 0])
    log_wdh = np.log(values[:, 1])
    differences = log_wh - log_wdh
    mean_l = float(np.mean(differences))
    sd_l = float(np.std(differences, ddof=1))
    if sd_l == 0:
        raise ValueError("As diferenças logarítmicas não apresentam variabilidade.")

    normality_result = stats.shapiro(differences, nan_policy="raise")
    # H0: média de L >= 0; H1: média de L < 0, como na análise por modelo.
    hypothesis_result = stats.ttest_rel(
        log_wh, log_wdh, alternative="less", nan_policy="raise"
    )

    common = {
        "analysis": "pooled",
        "n_models": int(pairs["model_id"].nunique()),
        "n_specifications": int(pairs["spec_id"].nunique()),
        "n_pairs": len(pairs),
    }
    normality = pd.DataFrame([{
        **common,
        "shapiro_W": float(normality_result.statistic),
        "shapiro_p": float(normality_result.pvalue),
        "normality_rejected": bool(normality_result.pvalue < ALPHA),
    }])
    hypotheses = pd.DataFrame([{
        **common,
        "mean_L": mean_l,
        "sd_L": sd_l,
        "t_statistic": float(hypothesis_result.statistic),
        "df": int(hypothesis_result.df),
        "t_p": float(hypothesis_result.pvalue),
        "reject_H0": bool(hypothesis_result.pvalue < ALPHA),
    }])

    dz = mean_l / sd_l
    ratio = float(np.exp(mean_l))
    if ratio < 1:
        direction = "menores"
        percent_difference = (1 - ratio) * 100
    elif ratio > 1:
        direction = "maiores"
        percent_difference = (ratio - 1) * 100
    else:
        direction = "sem diferença"
        percent_difference = 0.0

    effects = pd.DataFrame([{
        **common,
        "cohen_dz": dz,
        "effect_magnitude": classify_effect_size(dz),
        "R": ratio,
        "percent_difference": percent_difference,
        "wh_direction": direction,
    }])
    return normality, hypotheses, effects



def write_pooled_plots(pairs, out_dir):
    from matplotlib.figure import Figure

    out_dir.mkdir(parents=True, exist_ok=True)
    differences = np.log(pairs["Y_WH"]) - np.log(pairs["Y_WDh"])
    context = (
        f"{len(pairs)} pares | {pairs['model_id'].nunique()} modelos | "
        f"{pairs['spec_id'].nunique()} especificações"
    )

    scatter = Figure(figsize=(6.4, 6.4), layout="constrained")
    ax = scatter.subplots()
    values = pairs[["Y_WH", "Y_WDh"]].to_numpy(dtype=float)
    lower, upper = float(values.min()) / 1.2, float(values.max()) * 1.2
    ax.scatter(pairs["Y_WDh"], pairs["Y_WH"], s=22, alpha=0.45,
               color="#2864A6", edgecolors="none")
    ax.plot([lower, upper], [lower, upper], "--", color="#555555",
            label="Estimativas iguais")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Dias convertidos para horas ($Y_{WDh}$)")
    ax.set_ylabel(r"Horas ($Y_{WH}$)")
    ax.set_title(f"Comparação conjunta das estimativas\n{context}", fontsize=10)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)

    distribution = Figure(figsize=(7.2, 4.8), layout="constrained")
    ax = distribution.subplots()
    edges = np.histogram_bin_edges(differences, bins="fd")
    ax.hist(differences, bins=edges, density=True, color="#2864A6",
            alpha=0.75, edgecolor="white")
    ax.axvline(0, linestyle="--", color="#555555", label="L = 0")
    ax.axvline(float(differences.mean()), color="#B65D19",
               label=f"Média de L = {differences.mean():.3f}")
    lower, upper = min(float(edges[0]), 0.0), max(float(edges[-1]), 0.0)
    margin = (upper - lower) * 0.05
    ax.set_xlim(lower - margin, upper + margin)
    ax.set_xlabel(r"$L = \ln(Y_{WH}) - \ln(Y_{WDh})$")
    ax.set_ylabel("Densidade")
    ax.set_title(f"Distribuição conjunta das diferenças logarítmicas\n{context}", fontsize=10)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.2)
    ax.set_axisbelow(True)


    for name, figure in (("scatter", scatter), ("distribution", distribution)):
        figure.savefig(out_dir / f"{name}.png", dpi=300)
        figure.savefig(out_dir / f"{name}.pdf")
    print(f"Gráficos salvos em: {out_dir}")


def main() -> None:
    pairs = load_valid_pairs()
    normality, hypotheses, effects = build_pooled_summaries(pairs)
    write_pooled_plots(pairs, OUTPUT_DIR / "figures")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, table in (
        ("valid_pairs.csv", pairs),
        ("normality.csv", normality),
        ("hypothesis_tests.csv", hypotheses),
        ("effects.csv", effects),
    ):
        table.to_csv(OUTPUT_DIR / filename, index=False)

    print("Análise conjunta dos pares válidos")
    print(normality.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(hypotheses.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(effects.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"\nResultados salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
