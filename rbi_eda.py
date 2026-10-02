"""
rbi_eda.py
Exploratory Data Analysis figures for the corrected dataset representation
(20-dimensional window-level features extracted from raw 256-sample RBI
signal windows), replacing the EDA built around the six algebraically
related scalar features (Current, SNR, SNRdB, LoD, LoQ, Capacity) in an
earlier version of this manuscript.

Generates, all at 600 dpi, Times New Roman:
  Fig_eda_distributions.png    histograms of all 20 features, split by class
  Fig_eda_boxplots.png         boxplots of all 20 features, split by class
  Fig_eda_correlation.png      Pearson correlation heatmap across features
  Fig_eda_pca.png              PCA scatter (PC1 vs PC2), coloured by class
  Fig_eda_pairplot.png         pairwise scatter for a selected feature subset

Also prints:
  descriptive statistics table (mean, std, min, 25%, 50%, 75%, max) per feature
  class balance summary
  top correlated feature pairs

Usage
-----
  python rbi_eda.py --n 6000 --out paper_assets --seed 42
"""

import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from rbi_all import generate_corpus, window_features

FEATURE_NAMES = ["mean", "std", "min", "max", "ptp", "median", "skew", "kurt",
                 "q25", "q75", "d_mean", "d_std", "d_absmax", "slope",
                 "spec_entropy", "log_lowband", "log_highband", "log_hl_ratio",
                 "acf1", "cv"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.linewidth": 0.7,
})
COLOR_NORMAL, COLOR_ANOM = "#a3a3a3", "#c0392b"


def build_dataframe(n_windows, seed):
    X, y, cls, params = generate_corpus(n_windows=n_windows, seed=seed)
    F = window_features(X)
    df = pd.DataFrame(F, columns=FEATURE_NAMES)
    df["Label"] = y
    df["class"] = cls
    return df


def print_descriptive_stats(df):
    print("\n" + "=" * 100)
    print("DESCRIPTIVE STATISTICS (window-level feature representation, n = {:,})"
          .format(len(df)))
    print("=" * 100)
    desc = df[FEATURE_NAMES].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    print(desc.round(4).to_string())

    print("\nClass balance:")
    vc = df["Label"].value_counts().sort_index()
    for lab, cnt in vc.items():
        name = "Normal (0)" if lab == 0 else "Anomalous (1)"
        print(f"  {name}: {cnt:,} ({cnt / len(df) * 100:.1f}%)")

    print("\nGenerative class balance:")
    vc2 = df["class"].value_counts()
    for cls, cnt in vc2.items():
        print(f"  {cls:>8s}: {cnt:,} ({cnt / len(df) * 100:.1f}%)")
    return desc


def print_top_correlations(df, k=10):
    corr = df[FEATURE_NAMES].corr()
    pairs = []
    for i, a in enumerate(FEATURE_NAMES):
        for b in FEATURE_NAMES[i + 1:]:
            pairs.append((abs(corr.loc[a, b]), a, b, corr.loc[a, b]))
    pairs.sort(reverse=True)
    print(f"\nTop {k} most correlated feature pairs:")
    for _, a, b, v in pairs[:k]:
        print(f"  {a:<14s} <-> {b:<14s}   r = {v:+.3f}")
    return corr


def fig_distributions(df, outdir, ncols=4):
    nrows = int(np.ceil(len(FEATURE_NAMES) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.0))
    for ax, feat in zip(axes.ravel(), FEATURE_NAMES):
        for lab, color, name in [(0, COLOR_NORMAL, "Normal"), (1, COLOR_ANOM, "Anomalous")]:
            v = df.loc[df.Label == lab, feat]
            ax.hist(v, bins=40, alpha=0.6, color=color, label=name, density=True)
        ax.set_title(feat, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_yticks([])
    for ax in axes.ravel()[len(FEATURE_NAMES):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9,
              bbox_to_anchor=(0.5, 1.02), frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{outdir}/Fig_eda_distributions.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"\n[saved] {outdir}/Fig_eda_distributions.png")


def fig_boxplots(df, outdir, ncols=4):
    nrows = int(np.ceil(len(FEATURE_NAMES) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.4, nrows * 2.2))
    for ax, feat in zip(axes.ravel(), FEATURE_NAMES):
        data = [df.loc[df.Label == 0, feat], df.loc[df.Label == 1, feat]]
        bp = ax.boxplot(data, labels=["N", "A"], patch_artist=True,
                        widths=0.6, showfliers=True,
                        flierprops=dict(marker=".", markersize=2, alpha=0.4))
        for patch, color in zip(bp["boxes"], [COLOR_NORMAL, COLOR_ANOM]):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_title(feat, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(FEATURE_NAMES):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{outdir}/Fig_eda_boxplots.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[saved] {outdir}/Fig_eda_boxplots.png")


def fig_correlation(corr, outdir):
    cmap = LinearSegmentedColormap.from_list(
        "rb", ["#2b5d8a", "#ffffff", "#c0392b"])
    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=7)
    ax.set_yticklabels(FEATURE_NAMES, fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Pearson r", fontsize=8)
    ax.set_title("Feature correlation, window-level representation", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{outdir}/Fig_eda_correlation.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[saved] {outdir}/Fig_eda_correlation.png")


def fig_pca(df, outdir):
    X = StandardScaler().fit_transform(df[FEATURE_NAMES].values)
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    for lab, color, name in [(0, COLOR_NORMAL, "Normal"), (1, COLOR_ANOM, "Anomalous")]:
        m = df.Label.values == lab
        ax.scatter(pcs[m, 0], pcs[m, 1], s=6, alpha=0.35, color=color,
                  label=name, linewidths=0)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
    ax.set_title("PCA of window-level feature representation", fontsize=10)
    ax.legend(fontsize=8, frameon=False, markerscale=3)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/Fig_eda_pca.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[saved] {outdir}/Fig_eda_pca.png  "
          f"(PC1+PC2 explain {pca.explained_variance_ratio_[:2].sum()*100:.1f}% of variance)")


def fig_pairplot(df, outdir, subset=("mean", "std", "spec_entropy", "acf1", "slope")):
    n = len(subset)
    fig, axes = plt.subplots(n, n, figsize=(n * 1.9, n * 1.9))
    for i, fi in enumerate(subset):
        for j, fj in enumerate(subset):
            ax = axes[i, j]
            if i == j:
                for lab, color in [(0, COLOR_NORMAL), (1, COLOR_ANOM)]:
                    v = df.loc[df.Label == lab, fi]
                    ax.hist(v, bins=25, color=color, alpha=0.6, density=True)
            else:
                for lab, color in [(0, COLOR_NORMAL), (1, COLOR_ANOM)]:
                    m = df.Label.values == lab
                    ax.scatter(df.loc[m, fj], df.loc[m, fi], s=3, alpha=0.25,
                              color=color, linewidths=0)
            if i == n - 1:
                ax.set_xlabel(fj, fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(fi, fontsize=7)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=6)
    plt.tight_layout()
    plt.savefig(f"{outdir}/Fig_eda_pairplot.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[saved] {outdir}/Fig_eda_pairplot.png  (subset: {', '.join(subset)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="paper_assets")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = build_dataframe(a.n, a.seed)
    print_descriptive_stats(df)
    corr = print_top_correlations(df)

    fig_distributions(df, a.out)
    fig_boxplots(df, a.out)
    fig_correlation(corr, a.out)
    fig_pca(df, a.out)
    fig_pairplot(df, a.out)

    df.drop(columns="class").to_csv(f"{a.out}/eda_features.csv", index=False)
    print(f"\nFeature table written to {a.out}/eda_features.csv")
    print(f"All EDA assets written to: {a.out}/")
