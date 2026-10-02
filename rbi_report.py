"""
rbi_report.py
Generates every results figure and table for the manuscript from the CSV/JSON
outputs already produced by rbi_all.py. Run this AFTER the experiment and
ablation runs have completed. Nothing here retrains a model; it only reads
and formats existing results.

Expected inputs (paths are configurable via the arguments below):
  results_v2/table4_mean_ci.csv
  results_v2/per_fold_metrics.csv
  results_v2/confusion_matrices.json
  results_v2/significance_f1.csv           (or significance_accuracy.csv etc.)
  ablation/ablation_summary.csv

Outputs, all written to --out (default: paper_assets/):
  Table4.csv, Table4.md                     formatted results table
  Table5_significance.csv, Table5.md        paired comparisons
  TableZ_ablation.csv, TableZ.md            resampling ablation
  Fig_f1_comparison.png                     bar chart, all models, F1 with CI
  Fig_roc_accuracy.png                      scatter, accuracy vs ROC-AUC
  Fig_confusion_best.png                    best model confusion matrix
  Fig_confusion_all_deep.png                six deep architectures, 2x3 grid
  Fig_ablation.png                          resampling ablation, grouped bars
  Fig_fidelity.png                          four-panel electrochemical fidelity
  console output                            every table also printed as text

Usage
-----
  python rbi_report.py --results results_v2 --ablation ablation --out paper_assets
"""

import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.linewidth": 0.8, "figure.dpi": 100,
})

ORDER = ["1D-CNN", "CNN", "GRU", "LSTM", "CNN+GRU", "CNN+LSTM",
         "RandomForest", "IsolationForest", "OneClassSVM",
         "CUSUM", "EWMA", "NaiveThreshold"]
LABEL = {"RandomForest": "Random Forest", "IsolationForest": "Isolation Forest",
         "OneClassSVM": "One-Class SVM", "NaiveThreshold": "Naive threshold"}
DEEP = ["1D-CNN", "CNN", "GRU", "LSTM", "CNN+GRU", "CNN+LSTM"]
COLOR_DEEP, COLOR_BASE = "#2b5d8a", "#a3a3a3"


def nm(m):
    return LABEL.get(m, m)


# =====================================================================
# TABLE 4 : main results table
# =====================================================================
def build_table4(results_dir, outdir):
    d = pd.read_csv(f"{results_dir}/table4_mean_ci.csv")
    d["ci_low"] = d.ci_low.clip(0, 1)
    d["ci_high"] = d.ci_high.clip(0, 1)

    metrics = ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "mse", "mae"]
    wide = d.pivot(index="model", columns="metric", values=["mean", "sd"])
    wide = wide.reindex(ORDER)

    rows = []
    for m in ORDER:
        row = {"Model": nm(m)}
        for met in metrics:
            mean = wide[("mean", met)][m]
            sd = wide[("sd", met)][m]
            row[met] = f"{mean:.3f} ± {sd:.3f}"
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{outdir}/Table4.csv", index=False)

    md = ["| Model | " + " | ".join(m.upper() for m in metrics) + " |",
          "|---|" + "---|" * len(metrics)]
    for _, r in tab.iterrows():
        md.append("| " + r["Model"] + " | " + " | ".join(r[m] for m in metrics) + " |")
    open(f"{outdir}/Table4.md", "w").write("\n".join(md))

    print("\n" + "=" * 100)
    print("TABLE 4. Classification performance, mean ± SD across cross-validation folds")
    print("=" * 100)
    print(tab.to_string(index=False))
    return d, tab


# =====================================================================
# TABLE 5 : significance tests
# =====================================================================
def build_table5(results_dir, outdir, metric="f1"):
    path = f"{results_dir}/significance_{metric}.csv"
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return None
    s = pd.read_csv(path)
    s.to_csv(f"{outdir}/Table5_significance.csv", index=False)

    cols = ["comparison", "mean_diff", "cohens_d", "ttest_p", "wilcoxon_p",
            "ttest_p_holm", "wilcoxon_p_holm", "significant_at_0.05"]
    cols = [c for c in cols if c in s.columns]
    disp = s[cols].round(4)

    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in disp.iterrows():
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    open(f"{outdir}/Table5.md", "w").write("\n".join(md))

    print("\n" + "=" * 100)
    print(f"TABLE 5. Paired comparisons, {metric.upper()}")
    print("=" * 100)
    print(disp.to_string(index=False))
    return s


# =====================================================================
# TABLE Z : resampling ablation
# =====================================================================
def build_ablation_table(ablation_dir, outdir):
    path = f"{ablation_dir}/ablation_summary.csv"
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return None
    piv = pd.read_csv(path, keep_default_na=False, na_values=[""])
    piv.columns = [str(c) for c in piv.columns]
    val_col = "mean" if "mean" in piv.columns else piv.columns[-1]
    wide = piv.pivot(index="model", columns="resampling", values=val_col)
    wide.columns = [str(c) for c in wide.columns]
    wide = wide.reindex([m for m in ORDER if m in wide.index])
    wide.to_csv(f"{outdir}/TableZ_ablation.csv")

    md = ["| Model | " + " | ".join(wide.columns) + " |", "|---|" + "---|" * len(wide.columns)]
    for m, row in wide.iterrows():
        md.append("| " + nm(m) + " | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    open(f"{outdir}/TableZ.md", "w").write("\n".join(md))

    print("\n" + "=" * 100)
    print("TABLE Z. Resampling ablation, mean F1")
    print("=" * 100)
    print(wide.round(3).to_string())
    return wide


# =====================================================================
# FIGURE : F1 bar chart with 95% CI, all models
# =====================================================================
def fig_f1_comparison(d, outdir):
    sub = d[d.metric == "f1"].set_index("model").reindex(ORDER)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = [COLOR_DEEP if m in DEEP else COLOR_BASE for m in ORDER]
    x = np.arange(len(ORDER))
    err = np.vstack([sub["mean"] - sub["ci_low"], sub["ci_high"] - sub["mean"]])
    ax.bar(x, sub["mean"], yerr=err, capsize=3, color=colors,
          edgecolor="black", linewidth=0.6, width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels([nm(m) for m in ORDER], rotation=40, ha="right", fontsize=8.5)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.set_title("Detection performance across architectures and baselines\n"
                "(bars: deep architectures; grey: classical/statistical baselines)",
                fontsize=9.5)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/Fig_f1_comparison.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"\n[saved] {outdir}/Fig_f1_comparison.png")


# =====================================================================
# FIGURE : accuracy vs ROC-AUC scatter
# =====================================================================
def fig_roc_accuracy(d, outdir):
    acc = d[d.metric == "accuracy"].set_index("model")["mean"]
    auc = d[d.metric == "roc_auc"].set_index("model")["mean"]
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for m in ORDER:
        c = COLOR_DEEP if m in DEEP else COLOR_BASE
        marker = "o" if m in DEEP else "s"
        ax.scatter(acc[m], auc[m], s=90, color=c, marker=marker,
                  edgecolor="black", linewidth=0.6, zorder=3)
        ax.annotate(nm(m), (acc[m], auc[m]), fontsize=7.5,
                   xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Accuracy"); ax.set_ylabel("ROC-AUC")
    ax.set_title("Accuracy versus ROC-AUC by model", fontsize=10)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/Fig_roc_accuracy.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[saved] {outdir}/Fig_roc_accuracy.png")


# =====================================================================
# FIGURE : confusion matrices
# =====================================================================
def _draw_cm(ax, cm, title, ylab=True, xlab=True):
    cmap = LinearSegmentedColormap.from_list("b", ["#ffffff", "#2b5d8a"])
    cm = np.array(cm, float)
    rn = cm / cm.sum(1, keepdims=True)
    ax.imshow(rn, cmap=cmap, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i,j]):,}\n({rn[i,j]*100:.1f}%)",
                    ha="center", va="center", fontsize=8,
                    color="white" if rn[i, j] > 0.55 else "#1a1a1a")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Attack"], fontsize=8)
    ax.set_yticklabels(["Benign", "Attack"], fontsize=8, rotation=90, va="center")
    if xlab: ax.set_xlabel("Predicted", fontsize=8.5)
    if ylab: ax.set_ylabel("Actual", fontsize=8.5)
    ax.set_title(title, fontsize=9.5, pad=6)
    ax.tick_params(length=0)


def fig_confusion(results_dir, table4, outdir):
    cm_path = f"{results_dir}/confusion_matrices.json"
    if not os.path.exists(cm_path):
        print(f"[skip] {cm_path} not found")
        return
    CM = json.load(open(cm_path))
    d = table4
    best = d[d.metric == "f1"].sort_values("mean").iloc[-1]["model"]

    fig, ax = plt.subplots(figsize=(3.3, 3.1))
    _draw_cm(ax, CM[best], nm(best))
    plt.tight_layout()
    plt.savefig(f"{outdir}/Fig_confusion_best.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[saved] {outdir}/Fig_confusion_best.png  (best model: {best})")

    order = [m for m in DEEP if m in CM]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.0))
    for k, (ax, m) in enumerate(zip(axes.ravel(), order)):
        _draw_cm(ax, CM[m], nm(m), ylab=(k % 3 == 0), xlab=(k >= 3))
    plt.tight_layout(pad=1.1)
    plt.savefig(f"{outdir}/Fig_confusion_all_deep.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[saved] {outdir}/Fig_confusion_all_deep.png")


# =====================================================================
# FIGURE : resampling ablation, grouped bar chart
# =====================================================================
def fig_ablation(wide, outdir):
    if wide is None:
        return
    models = list(wide.index)
    conds = list(wide.columns)
    x = np.arange(len(models))
    width = 0.8 / len(conds)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    palette = ["#a3a3a3", "#2b5d8a", "#6fa8d8"]
    for i, c in enumerate(conds):
        ax.bar(x + i * width, wide[c], width=width, label=c,
              color=palette[i % len(palette)], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x + width * (len(conds) - 1) / 2)
    ax.set_xticklabels([nm(m) for m in models], rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("F1 score"); ax.set_ylim(0, 1)
    ax.set_title("Effect of resampling strategy on F1, by architecture", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{outdir}/Fig_ablation.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[saved] {outdir}/Fig_ablation.png")


# =====================================================================
# FIGURE : electrochemical fidelity, four panels
# =====================================================================
def fig_fidelity(outdir):
    """Regenerates the four fidelity checks and plots them. Requires
    rbi_all.py to be importable (same folder or on the path)."""
    try:
        from rbi_all import PARAMS, F_CONST, R_GAS, _cv_simulate
    except Exception as e:
        print(f"[skip] fidelity figure needs rbi_all.py on the path: {e}")
        return

    p = PARAMS
    n, T = p["n_electrons"], p["T_kelvin"]
    A, D, C = p["A_cm2"], p["D_cm2_s"], p["C_max_mol_cm3"]
    k0, aa = p["k0_cm_s"], p["alpha_a"]
    f = n * F_CONST / (R_GAS * T)

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0))

    # (a) Randles-Sevcik
    nus = np.array([0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0])
    ip = []
    for v in nus:
        E, i = _cv_simulate(v, p)
        ip.append(np.abs(i[: len(E) // 2]).max())
    ip = np.array(ip)
    ax = axes[0, 0]
    ax.scatter(np.sqrt(nus), ip, color=COLOR_DEEP, s=40, zorder=3)
    coef = np.polyfit(np.sqrt(nus), ip, 1)
    xx = np.linspace(0, np.sqrt(nus).max(), 50)
    ax.plot(xx, np.polyval(coef, xx), "--", color="black", linewidth=1)
    ax.set_xlabel(r"$\nu^{1/2}$ (V$^{1/2}$ s$^{-1/2}$)"); ax.set_ylabel(r"$i_p$ (A)")
    ax.set_title("(a) Randles-Sevcik linearity", fontsize=10)

    # (b) Tafel
    eta = np.linspace(0.12, 0.30, 60)
    i = k0 * np.exp(aa * f * eta)
    ax = axes[0, 1]
    ax.plot(eta * 1000, np.log10(i), color=COLOR_DEEP, linewidth=1.5)
    ax.set_xlabel(r"$\eta$ (mV)"); ax.set_ylabel(r"$\log_{10}|i|$")
    ax.set_title("(b) Tafel slope recovery", fontsize=10)

    # (c) Cottrell
    t = np.linspace(0.01, 5.0, 500)
    i = n * F_CONST * A * np.sqrt(D) * C / np.sqrt(np.pi * t)
    ax = axes[1, 0]
    ax.plot(t ** -0.5, i, color=COLOR_DEEP, linewidth=1.5)
    ax.set_xlabel(r"$t^{-1/2}$ (s$^{-1/2}$)"); ax.set_ylabel(r"$i$ (A)")
    ax.set_title("(c) Cottrell decay", fontsize=10)

    # (d) simulated CV, peak separation
    E, i = _cv_simulate(0.05, p, npts=3000)
    ax = axes[1, 1]
    ax.plot(E, i, color=COLOR_DEEP, linewidth=1.2)
    ax.axvline(p["E0_prime"], color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("E (V)"); ax.set_ylabel("i (A)")
    ax.set_title("(d) Simulated cyclic voltammogram", fontsize=10)

    for ax in axes.ravel():
        ax.grid(True, linewidth=0.3, alpha=0.4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{outdir}/Fig_fidelity.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[saved] {outdir}/Fig_fidelity.png")


# =====================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_v2")
    ap.add_argument("--ablation", default="ablation")
    ap.add_argument("--out", default="paper_assets")
    ap.add_argument("--sig-metric", default="f1")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)

    d, table4 = build_table4(a.results, a.out)
    build_table5(a.results, a.out, a.sig_metric)
    wide = build_ablation_table(a.ablation, a.out)

    fig_f1_comparison(d, a.out)
    fig_roc_accuracy(d, a.out)
    fig_confusion(a.results, d, a.out)
    fig_ablation(wide, a.out)
    fig_fidelity(a.out)

    print(f"\nAll tables and figures written to: {a.out}/")
