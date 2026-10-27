"""
Análise estatística dos resultados (scripts/analyze_results.py)

Lê os JSONs de experiment/results/ e reutiliza a validação e o pareamento
de export_results_xlsx.py. Seleciona os pares válidos dos modelos incluídos
e aplica Shapiro-Wilk e o teste t pareado unilateral à esquerda por modelo.
Calcula também Cohen's d_z, R e a diferença percentual.

Gera em analysis/outputs/:
  - valid_pairs.csv : pares válidos dos modelos incluídos
  - normality.csv : avaliação da normalidade por modelo
  - hypothesis_tests.csv : teste t pareado por modelo
  - effects.csv : tamanho de efeito e diferença percentual por modelo
  - figures/scatter/ : comparação WH × WDh (PNG e PDF)
  - figures/distribution/ : distribuição de L (PNG e PDF)
  - figures/ratio_by_model.png e .pdf : comparação descritiva de R

Uso:
    python scripts/analyze_results.py
"""

from pathlib import Path

import pandas as pd
from scipy import stats
import numpy as np
from matplotlib.figure import Figure

from export_results_xlsx import iter_results, build_pairs_rows


ROOT        = Path(__file__).resolve().parent.parent
OUTPUT_DIR  = ROOT / "analysis" / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
ALPHA = 0.05

# Laguna foi excluído pelo critério de elegibilidade: propósito geral.
EXCLUDED_MODELS = {"poolside/laguna-s-2.1:free"}


# ---------------------------------------------------------------------------
# Leitura e seleção dos pares
# ---------------------------------------------------------------------------

def load_valid_pairs() -> pd.DataFrame:
    """Retorna os pares válidos dos modelos incluídos na análise."""
    headers, rows = build_pairs_rows(list(iter_results()))
    pairs = pd.DataFrame(rows, columns=headers)

    included = ~pairs["model_id"].isin(EXCLUDED_MODELS)
    valid = pairs["pair_complete"].eq(True)

    return pairs.loc[included & valid].copy()

# ---------------------------------------------------------------------------
# Avaliação da normalidade
# ---------------------------------------------------------------------------

def build_normality_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """Aplica Shapiro-Wilk às diferenças intrapar, separadamente por modelo."""
    rows = []
    for model_id, model_pairs in pairs.groupby("model_id", sort=True):
        differences = model_pairs["L_im"].to_numpy(dtype=float)

        if len(differences) < 3:
            raise ValueError(f"{model_id}: Shapiro-Wilk exige pelo menos 3 pares.")
        if model_pairs["L_im"].nunique() < 2:
            raise ValueError(f"{model_id}: L_im constante; normalidade não avaliável.")

        # L_im já é a diferença entre os logaritmos das estimativas.
        result = stats.shapiro(differences, nan_policy="raise")
        rows.append({
            "model_id": model_id,
            "n_pairs": len(differences),
            "shapiro_W": float(result.statistic),
            "shapiro_p": float(result.pvalue),
            "normality_rejected": bool(result.pvalue < ALPHA),
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Teste das hipóteses
# ---------------------------------------------------------------------------

def build_hypothesis_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """Aplica o teste t pareado unilateral à esquerda aos logaritmos."""
    rows = []
    for model_id, model_pairs in pairs.groupby("model_id", sort=True):
        # As duas colunas mantêm a mesma ordem de especificações.
        log_wh = np.log(model_pairs["Y_WH"].to_numpy(dtype=float))
        log_wdh = np.log(model_pairs["Y_WDh"].to_numpy(dtype=float))
        differences = log_wh - log_wdh
        n_pairs = len(differences)

        if n_pairs < 2:
            raise ValueError(f"{model_id}: teste t exige pelo menos 2 pares.")

        mean_l = float(np.mean(differences))
        sd_l = float(np.std(differences, ddof=1))
        if sd_l == 0:
            raise ValueError(f"{model_id}: diferenças sem variabilidade.")

        # H0: média de L >= 0; H1: média de L < 0.
        result = stats.ttest_rel(
            log_wh, log_wdh, alternative="less", nan_policy="raise"
        )
        rows.append({
            "model_id": model_id,
            "n_pairs": n_pairs,
            "mean_L": mean_l,
            "sd_L": sd_l,
            "t_statistic": float(result.statistic),
            "df": int(result.df),
            "t_p": float(result.pvalue),
            "reject_H0": bool(result.pvalue < ALPHA),
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Tamanho de efeito e diferença percentual
# ---------------------------------------------------------------------------

def classify_effect_size(dz: float) -> str:
    """Classifica |d_z| usando as referências de Cohen previstas no protocolo."""
    magnitude = abs(dz)
    if magnitude < 0.2:
        return "abaixo de 0,2"
    if magnitude < 0.5:
        return "pequeno"
    if magnitude < 0.8:
        return "médio"
    return "grande"


def build_effect_summary(hypotheses: pd.DataFrame) -> pd.DataFrame:
    """Calcula d_z, R e a diferença percentual a partir da média e do desvio de L."""
    rows = []
    for row in hypotheses.itertuples(index=False):
        dz = row.mean_L / row.sd_L
        ratio = float(np.exp(row.mean_L))

        # R é o antilogaritmo da média de L, conforme o protocolo.
        if ratio < 1:
            direction = "menores"
            percent_difference = (1 - ratio) * 100
        elif ratio > 1:
            direction = "maiores"
            percent_difference = (ratio - 1) * 100
        else:
            direction = "sem diferença"
            percent_difference = 0.0

        rows.append({
            "model_id": row.model_id,
            "n_pairs": row.n_pairs,
            "cohen_dz": dz,
            "effect_magnitude": classify_effect_size(dz),
            "R": ratio,
            "percent_difference": percent_difference,
            "wh_direction": direction,
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Visualizações
# ---------------------------------------------------------------------------

def write_scatter_plots(pairs: pd.DataFrame, out_dir: Path) -> None:
    """Gera a comparação WH × WDh em escala logarítmica, por modelo."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mesmos limites nos dois eixos e em todos os modelos.
    values = pairs[["Y_WH", "Y_WDh"]].to_numpy(dtype=float)
    lower = float(values.min()) / 1.2
    upper = float(values.max()) * 1.2

    for model_id, model_pairs in pairs.groupby("model_id", sort=True):
        fig = Figure(figsize=(6.4, 6.4), layout="constrained")
        ax = fig.subplots()
        ax.scatter(
            model_pairs["Y_WDh"], model_pairs["Y_WH"],
            s=30, alpha=0.7, color="#2864A6", edgecolors="none",
        )
        ax.plot(
            [lower, upper], [lower, upper],
            color="#555555", linestyle="--", linewidth=1.2,
            label="Estimativas iguais",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"Dias convertidos para horas ($Y_{WDh}$)")
        ax.set_ylabel(r"Horas ($Y_{WH}$)")
        ax.set_title(f"{model_id}\n{len(model_pairs)} pares válidos", fontsize=11)
        ax.set_axisbelow(True)
        ax.grid(True, which="major", alpha=0.2)
        ax.legend(loc="upper left", frameon=False)

        # Nomes das figuras portáveis entre sistemas operacionais.
        filename = model_id.replace("/", "__").replace(":", "__")
        fig.savefig(out_dir / f"{filename}.png", dpi=300)
        fig.savefig(out_dir / f"{filename}.pdf")
        
        
        
def write_distribution_plots(pairs: pd.DataFrame, out_dir: Path) -> None:
    """Gera histogramas de L por modelo, com zero como referência."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Intervalos comuns pela regra de Freedman-Diaconis.
    values = pairs["L_im"].to_numpy(dtype=float)
    bin_edges = np.histogram_bin_edges(values, bins="fd")
    lower = min(float(bin_edges[0]), 0.0)
    upper = max(float(bin_edges[-1]), 0.0)
    margin = (upper - lower) * 0.05

    groups = list(pairs.groupby("model_id", sort=True))
    max_density = max(
        float(np.histogram(group["L_im"], bins=bin_edges, density=True)[0].max())
        for _, group in groups
    )

    for model_id, model_pairs in groups:
        fig = Figure(figsize=(7, 4.8), layout="constrained")
        ax = fig.subplots()
        ax.hist(
            model_pairs["L_im"], bins=bin_edges, density=True,
            color="#2864A6", alpha=0.75, edgecolor="white", linewidth=0.8,
        )
        ax.axvline(
            0, color="#555555", linestyle="--", linewidth=1.2,
            label="L = 0 (estimativas iguais)",
        )

        # Mesmas escalas em todos os modelos; L já está na escala logarítmica.
        ax.set_xlim(lower - margin, upper + margin)
        ax.set_ylim(0, max_density * 1.2)
        ax.set_xlabel(r"$L = \ln(Y_{WH}) - \ln(Y_{WDh})$")
        ax.set_ylabel("Densidade")
        ax.set_title(f"{model_id}\n{len(model_pairs)} pares válidos", fontsize=11)
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", alpha=0.2)
        ax.legend(loc="upper right", frameon=False, fontsize=9)

        filename = model_id.replace("/", "__").replace(":", "__")
        fig.savefig(out_dir / f"{filename}.png", dpi=300)
        fig.savefig(out_dir / f"{filename}.pdf")
        
        
def write_ratio_plot(effects: pd.DataFrame, out_dir: Path) -> None:
    """Gera a comparação descritiva de R por modelo, com referência em 1."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # A ordenação é apenas visual e não altera a tabela de efeitos.
    ordered = effects.sort_values(["R", "model_id"])
    positions = np.arange(len(ordered))
    labels = [
        f"{row.model_id} (n={row.n_pairs})"
        for row in ordered.itertuples(index=False)
    ]

    fig = Figure(figsize=(11, 6), layout="constrained")
    ax = fig.subplots()
    ax.scatter(ordered["R"], positions, s=45, color="#2864A6", zorder=3)
    ax.axvline(
        1, color="#555555", linestyle="--", linewidth=1.2, label="R = 1",
    )

    for position, ratio in zip(positions, ordered["R"]):
        ax.annotate(
            f"{ratio:.3f}", (ratio, position), xytext=(7, 0),
            textcoords="offset points", va="center", fontsize=9,
        )

    ax.set_yticks(positions, labels=labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.0, float(ordered["R"].max())) * 1.15)
    ax.set_xlabel(r"$R = \exp(\overline{L})$")
    ax.set_title("Razão geométrica por modelo", fontsize=12)
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(loc="upper right", frameon=False)

    fig.savefig(out_dir / "ratio_by_model.png", dpi=300)
    fig.savefig(out_dir / "ratio_by_model.pdf")

# ---------------------------------------------------------------------------
# Escrita CSV
# ---------------------------------------------------------------------------

def write_valid_pairs(pairs: pd.DataFrame, out_path: Path) -> None:
    """Salva os pares selecionados em CSV, sem arredondamento explícito."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

def main() -> None:
    pairs = load_valid_pairs()
    normality = build_normality_summary(pairs)
    hypotheses = build_hypothesis_summary(pairs)
    effects = build_effect_summary(hypotheses)

    out_path = OUTPUT_DIR / "valid_pairs.csv"
    write_valid_pairs(pairs, out_path)

    normality_path = OUTPUT_DIR / "normality.csv"
    normality.to_csv(normality_path, index=False)

    hypotheses_path = OUTPUT_DIR / "hypothesis_tests.csv"
    hypotheses.to_csv(hypotheses_path, index=False)

    effects_path = OUTPUT_DIR / "effects.csv"
    effects.to_csv(effects_path, index=False)
    
    write_scatter_plots(pairs, FIGURES_DIR / "scatter")
    print(f"Gráficos salvos em: {FIGURES_DIR / 'scatter'}")
    write_distribution_plots(pairs, FIGURES_DIR / "distribution")
    print(f"Distribuições salvas em: {FIGURES_DIR / 'distribution'}")
    write_ratio_plot(effects, FIGURES_DIR)
    print(f"Comparação de R salva em: {FIGURES_DIR / 'ratio_by_model.png'}")

    print(effects.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"\nTotal: {len(pairs)} pares válidos.")
    print(f"CSV salvo em: {out_path}")
    print(f"Normalidade salva em: {normality_path}")
    print(f"Teste t salvo em: {hypotheses_path}")
    print(f"Efeitos salvos em: {effects_path}")


if __name__ == "__main__":
    main()