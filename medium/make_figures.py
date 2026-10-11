"""
Figures and summary tables for the RBF-SVM results (regenerated from results/*.csv,
no retraining).

Error bars: per (repeat, k) the metric is first averaged over failure subsets;
bars are the std of those 10 repeat means (outer-split variance). Where shown,
the shaded band is the mean within-repeat std across failure patterns
(sensitivity to WHICH sensors fail). This matches the team protocol in
TEAM_README.txt; both sources are tabulated in results/medium_summary.csv.

    python -m medium.make_figures [--smoke]
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from medium.common import FIGURES_DIR, GAS_NAMES, results_path
from medium.model import GRID
from shared_protocol import N_CLASSES
from shared_robustness import EXPECTED_FAILURE_LEVELS as K_LEVELS

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e4e3df"
SEQ_BLUES = ["#b7d3f3", "#7fb0ea", "#4a8fe0", "#2a78d6", "#174f94"]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRIDC, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
    "lines.markersize": 4, "legend.frameon": False, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def save(fig, name: str, smoke: bool) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    stem = f"medium_{name}" + ("_smoke" if smoke else "")
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  wrote figures/{stem}.pdf/.png")


def read(path) -> pd.DataFrame:
    """Read a results CSV and map the shared column names to short internal ones."""
    df = pd.read_csv(path, dtype={"test_batch": str})
    return df.rename(columns={"outer_repeat": "repeat", "n_failed_sensors": "k",
                              "pattern_id": "subset_id"})


def two_level_summary(df: pd.DataFrame, by: list[str], metric: str) -> pd.DataFrame:
    """mean, std across repeats (of subset means), mean within-repeat std across subsets."""
    per_repeat = df.groupby(by + ["repeat"])[metric].agg(["mean", "std"]).reset_index()
    out = per_repeat.groupby(by).agg(mean=("mean", "mean"), std_repeat=("mean", "std"),
                                     std_subset=("std", "mean")).reset_index()
    return out.fillna({"std_subset": 0.0})


def plot_curve(ax, s: pd.DataFrame, x: str, color: str, label: str, marker: str = "o",
               linestyle: str = "-", band: bool = True) -> None:
    if band:
        ax.fill_between(s[x], s["mean"] - s["std_subset"], s["mean"] + s["std_subset"],
                        color=color, alpha=0.15, linewidth=0)
    ax.errorbar(s[x], s["mean"], yerr=s["std_repeat"], color=color, label=label,
                marker=marker, linestyle=linestyle, capsize=2, elinewidth=0.9)


def fig_degradation(smoke: bool) -> None:
    sweeps = {"RBF-SVM": (results_path("medium_failure_sweep", smoke), BLUE, "o", "-"),
              "RBF-SVM + sensor dropout": (results_path("medium_dropout_failure_sweep", smoke),
                                           ORANGE, "s", "--")}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
    tables = []
    for label, (path, color, marker, ls) in sweeps.items():
        if not path.exists():
            continue
        df = read(path)
        df = df[df.test_batch == "all"]
        for ax, metric in zip(axes, ("accuracy", "macro_f1")):
            s = two_level_summary(df, ["k"], metric)
            plot_curve(ax, s, "k", color, label, marker, ls)
            tables.append(s.assign(model=label, metric=metric))
    chance = 1 / N_CLASSES
    for ax, title in zip(axes, ("Accuracy", "Macro-F1")):
        ax.axhline(chance, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_xlabel("Number of failed sensors k (of 16)")
        ax.set_title(title, loc="left")
        ax.set_xticks(K_LEVELS)
    axes[0].text(K_LEVELS[-1], chance + 0.01, "chance", color=MUTED, fontsize=7, ha="right", va="bottom")
    axes[0].set_ylabel("Score on batches 8-10")
    axes[0].set_ylim(0, 1)
    axes[1].legend(loc="upper right")
    save(fig, "degradation", smoke)
    if tables:
        pd.concat(tables).round(4).to_csv(results_path("medium_summary", smoke), index=False)
        print("  wrote results/medium_summary" + ("_smoke" if smoke else "") + ".csv")


def fig_by_batch(smoke: bool) -> None:
    path = results_path("medium_failure_sweep", smoke)
    if not path.exists():
        return
    df = read(path)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for b, color, marker in (("8", BLUE, "o"), ("9", ORANGE, "s"), ("10", AQUA, "^")):
        s = two_level_summary(df[df.test_batch == b], ["k"], "macro_f1")
        plot_curve(ax, s, "k", color, f"Batch {b}", marker, band=False)
    ax.set_xlabel("Number of failed sensors k")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1)
    ax.set_xticks(K_LEVELS)
    ax.set_title("RBF-SVM by test batch (drift x failure)", loc="left")
    ax.legend(loc="upper right")
    save(fig, "by_batch", smoke)


def fig_confusion(smoke: bool, k_fail: int = 8) -> None:
    path = results_path("medium_confusion", smoke)
    if not path.exists():
        return
    df = read(path)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    short = [g[:5] for g in GAS_NAMES]
    for ax, k in zip(axes, (0, k_fail)):
        sub = df[df.k == k]
        # Row-normalise each (repeat, subset) matrix, then average.
        sub = sub.assign(rate=sub["count"] / sub.groupby(["repeat", "subset_id", "true"])["count"]
                         .transform("sum"))
        cm = sub.groupby(["true", "pred"]).rate.mean().unstack().reindex(
            index=range(N_CLASSES), columns=range(N_CLASSES)).to_numpy()
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for t in range(N_CLASSES):
            for p in range(N_CLASSES):
                ax.text(p, t, f"{cm[t, p]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if cm[t, p] > 0.55 else INK)
        ax.set_xticks(range(N_CLASSES), short, rotation=45, ha="right")
        ax.set_yticks(range(N_CLASSES), short)
        ax.set_xlabel("Predicted")
        ax.grid(False)
        ax.set_title(f"k = {k} failed sensors (row-normalised)", loc="left")
    axes[0].set_ylabel("True gas")
    save(fig, "confusion", smoke)


def fig_class_f1(smoke: bool) -> None:
    path = results_path("medium_failure_sweep", smoke)
    if not path.exists():
        return
    df = read(path)
    df = df[df.test_batch == "all"]
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 3.6), sharex=True, sharey=True)
    for c, ax in enumerate(axes.flat):
        s = two_level_summary(df, ["k"], f"f1_c{c}")
        plot_curve(ax, s, "k", BLUE, "RBF-SVM")
        ax.set_title(GAS_NAMES[c], loc="left")
        ax.set_ylim(0, 1)
        ax.set_xticks(K_LEVELS)
    for ax in axes[1]:
        ax.set_xlabel("Failed sensors k")
    for ax in axes[:, 0]:
        ax.set_ylabel("Per-class F1")
    save(fig, "class_f1", smoke)


def fig_learning_curve(smoke: bool) -> None:
    path = results_path("medium_learning_curve", smoke)
    if not path.exists():
        return
    df = read(path)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for (k, color, marker) in zip(sorted(df.k.unique()), (BLUE, ORANGE, AQUA), "os^"):
        sub = df[df.k == k]
        s = two_level_summary(sub, ["train_fraction"], "macro_f1")
        s["n_train"] = sub.groupby("train_fraction").n_train.mean().to_numpy()
        plot_curve(ax, s, "n_train", color, f"k = {k}", marker)
    ax.set_xlabel("Training samples (batches 1-7)")
    ax.set_ylabel("Macro-F1 on batches 8-10")
    ax.set_ylim(0, 1)
    ax.set_title("RBF-SVM learning curve", loc="left")
    ax.legend(loc="center right")
    save(fig, "learning_curve", smoke)


def fig_gamma(smoke: bool) -> None:
    path = results_path("medium_gamma_ablation", smoke)
    if not path.exists():
        return
    df = read(path)
    gammas = sorted(df.gamma_mult.unique())
    colors = SEQ_BLUES[-len(gammas):] if len(gammas) <= len(SEQ_BLUES) else None
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for g, color, marker in zip(gammas, colors, "os^Dv"):
        sub = df[df.gamma_mult == g]
        tuned = " (tuned grid mid)" if g == GRID["gamma_mult"][1] else ""
        for ax, metric in zip(axes, ("accuracy", "max_kernel")):
            s = two_level_summary(sub, ["k"], metric)
            plot_curve(ax, s, "k", color, f"{g:g} x scale{tuned}", marker, band=False)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Accuracy vs failures, by gamma", loc="left")
    axes[1].set_ylabel("Mean similarity to nearest SV")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Similarity to nearest support vector", loc="left")
    for ax in axes:
        ax.set_xlabel("Number of failed sensors k")
        ax.set_xticks(K_LEVELS)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title=r"$\gamma$ multiplier", title_fontsize=7,
               loc="lower center", ncol=len(gammas), bbox_to_anchor=(0.5, -0.2))
    save(fig, "gamma_ablation", smoke)


def fig_tuning(smoke: bool) -> None:
    scores_path = results_path("medium_inner_scores", smoke)
    log_path = results_path("medium_tuning_log", smoke)
    if not scores_path.exists():
        return
    scores = pd.read_csv(scores_path)
    per_outer = scores.groupby(["outer_repeat", "C", "gamma_mult"]).score.mean().reset_index()
    surf = per_outer.groupby(["C", "gamma_mult"]).score.mean().unstack()
    log = pd.read_csv(log_path)
    chosen = log.pivot(index="outer_repeat", columns="hparam", values="chosen_value")
    freq = chosen.groupby(["C", "gamma_mult"]).size()

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.imshow(surf.to_numpy(), cmap="Blues", aspect="auto",
              vmin=np.nanmin(surf.to_numpy()) - 0.02, vmax=np.nanmax(surf.to_numpy()))
    for i, c in enumerate(surf.index):
        for j, g in enumerate(surf.columns):
            n = int(freq.get((c, g), 0))
            ax.text(j, i, f"{surf.loc[c, g]:.3f}" + (f"\nchosen {n}/{len(chosen)}" if n else ""),
                    ha="center", va="center", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(surf.columns)), [f"{g:g}" for g in surf.columns])
    ax.set_yticks(range(len(surf.index)), [f"{c:g}" for c in surf.index])
    ax.set_xlabel(r"$\gamma$ multiplier (x $\gamma_{scale}$)")
    ax.set_ylabel("C")
    ax.grid(False)
    ax.set_title("Inner-validation macro-F1 (batch 7)", loc="left")
    save(fig, "tuning", smoke)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for fig in (fig_degradation, fig_by_batch, fig_confusion, fig_class_f1,
                fig_learning_curve, fig_gamma, fig_tuning):
        fig(args.smoke)


if __name__ == "__main__":
    main()
