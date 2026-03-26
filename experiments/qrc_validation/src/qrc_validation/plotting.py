from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", palette="colorblind", context="talk")


def plot_cf1_rank_advantage(df: pd.DataFrame, out_path: Path) -> None:
    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.8), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        for key, label in [("delta_vs_esn", "QRC-Ent - ESN"), ("delta_vs_rbf", "QRC-Ent - RBF")]:
            grp = sub.groupby("rank")[key].agg(["mean", "std", "count"]).reset_index()
            sem = (grp["std"] / grp["count"].clip(lower=1) ** 0.5).fillna(0.0)
            ax.plot(grp["rank"], grp["mean"], marker="o", label=label)
            ax.fill_between(grp["rank"], grp["mean"] - 1.96 * sem, grp["mean"] + 1.96 * sem, alpha=0.2)
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_title(f"{ds}")
        ax.set_xlabel("PCA Components (count)")
        ax.set_ylabel("Delta Accuracy (absolute)")
        ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_cf2_did_eta(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    did = df.groupby("eta")["did_interaction_delta"].agg(["mean", "std", "count"]).reset_index()
    sem = (did["std"] / did["count"].clip(lower=1) ** 0.5).fillna(0.0)
    axes[0].plot(did["eta"], did["mean"], marker="o", label="DiD interaction")
    axes[0].fill_between(did["eta"], did["mean"] - 1.96 * sem, did["mean"] + 1.96 * sem, alpha=0.2)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("Entanglement Strength eta")
    axes[0].set_ylabel("DiD Delta Accuracy")
    axes[0].set_title("Interaction Strength")
    axes[0].legend(frameon=True)

    util = df.groupby("eta")["pareto_utility_J_eta"].agg(["mean", "std", "count"]).reset_index()
    usem = (util["std"] / util["count"].clip(lower=1) ** 0.5).fillna(0.0)
    axes[1].plot(util["eta"], util["mean"], marker="s", label="Utility J(eta)")
    axes[1].fill_between(util["eta"], util["mean"] - 1.96 * usem, util["mean"] + 1.96 * usem, alpha=0.2)
    axes[1].set_xlabel("Entanglement Strength eta")
    axes[1].set_ylabel("Utility (accuracy-runtime tradeoff)")
    axes[1].set_title("Runtime-Normalized Utility")
    axes[1].legend(frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_cf3_null_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), gridspec_kw={"wspace": 0.55})
    piv = df.pivot_table(index="dataset", columns="rank", values="bound_ratio_delta_over_Aalpha_eK", aggfunc="mean")
    sns.heatmap(
        piv,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        annot_kws={"size": 9},
        cbar_kws={"label": "Bound Ratio", "fraction": 0.08, "pad": 0.03},
        ax=axes[0],
    )
    axes[0].set_xlabel("PCA Components (count)")
    axes[0].set_ylabel("Dataset")
    axes[0].set_title("Bound Ratio by Dataset/Rank")
    axes[0].tick_params(axis="x", labelrotation=0, labelsize=10)
    axes[0].tick_params(axis="y", labelsize=10)

    piv2 = df.pivot_table(index="dataset", columns="rank", values="delta_accuracy_nonent_vs_emulator", aggfunc="mean")
    sns.heatmap(
        piv2,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0.0,
        annot_kws={"size": 9},
        cbar_kws={"label": "Delta Accuracy", "fraction": 0.08, "pad": 0.03},
        ax=axes[1],
    )
    axes[1].set_xlabel("PCA Components (count)")
    axes[1].set_ylabel("")
    axes[1].set_title("Non-Entangling Delta vs Emulator")
    axes[1].tick_params(axis="x", labelrotation=0, labelsize=10)
    axes[1].tick_params(axis="y", labelleft=False, left=False)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_channel_stability(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for fam, sub in df.groupby("noise_family"):
        grp = sub.groupby("noise_strength")["relative_rank_stability"].mean().reset_index()
        axes[0].plot(grp["noise_strength"], grp["relative_rank_stability"], marker="o", label=fam)
    axes[0].set_xlabel("Noise Strength (probability)")
    axes[0].set_ylabel("Relative Rank Stability")
    axes[0].set_title("Ranking Stability Across Channels")
    axes[0].legend(frameon=True, fontsize=9)

    for fam, sub in df.groupby("noise_family"):
        grp = sub.groupby("noise_strength")["error_rate_shift"].mean().reset_index()
        axes[1].plot(grp["noise_strength"], grp["error_rate_shift"], marker="s", label=fam)
    axes[1].set_xlabel("Noise Strength (probability)")
    axes[1].set_ylabel("Error Rate Shift")
    axes[1].set_title("Error Shift Under Channel Stress")
    axes[1].legend(frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
