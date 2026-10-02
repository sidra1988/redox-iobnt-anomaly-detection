"""
rbi_all.py
Self-contained pipeline for the RBI anomaly-detection study.

Contains everything: dataset generation, electrochemical fidelity checks,
six deep architectures, six classical baselines, the cross-validation
protocol with calibrated thresholds, significance testing, the resampling
ablation, and figure generation.

Modes
-----
  --mode fidelity     electrochemical verification (Section 4.7)
  --mode experiment   main cross-validation run (Table 4, Table 5)
  --mode ablation     resampling comparison (Section 4.4.2)
  --mode figures      confusion matrix figures from a completed run

Examples
--------
  python rbi_all.py --mode fidelity
  python rbi_all.py --mode experiment --n 6000 --folds 10 --out results_v2
  python rbi_all.py --mode ablation --n 4000 --folds 5 --out ablation

Resampling conditions
---------------------
  none       no resampling for any model
  smote_raw  SMOTE interpolated between raw sequences (deep) and between
             feature vectors (classical)
  window     window-based oversampling with jitter and circular shift (deep),
             feature-space SMOTE (classical).  THIS IS THE REPORTED SETTING.
  python rbi_all.py --mode figures --out results_v2
"""

import argparse, json, os, warnings
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             mean_squared_error, mean_absolute_error)

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Dropout,
                                         Flatten, Dense, LSTM, GRU)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    HAS_TF = True
except Exception:
    HAS_TF = False

# =====================================================================
# 1. PHYSICAL CONSTANTS AND INTERFACE PARAMETERS  (manuscript Table 3)
# =====================================================================
F_CONST, R_GAS, Q_E, K_B = 96485.0, 8.314, 1.602e-19, 1.381e-23

PARAMS = dict(
    n_electrons=1, T_kelvin=310.15, A_cm2=1e-4, D_cm2_s=6.7e-6,
    k0_cm_s=1e-3, alpha_a=0.5, E0_prime=-0.20, eta_bias=0.15,
    d_cm=50e-4, fs_hz=100.0, T_window=256, Ts_s=0.32,
    C_max_mol_cm3=1e-7, Re_ohm=1e6, target_snr_db=12.0, rho_bio=0.05,
)

ATTACK_CLASSES = ["benign", "B1", "B2", "A1", "A2", "A3", "A4", "A5", "A6"]
POSITIVE_CLASSES = {"A1", "A2", "A3", "A4", "A5", "A6"}

# =====================================================================
# 2. HYPERPARAMETERS  (manuscript appendix)
# =====================================================================
HP = dict(
    epochs=150, batch_size=64, learning_rate=1e-3,
    early_stopping_patience=20, lr_reduce_patience=8, lr_reduce_factor=0.5,
    calibration_fraction=0.15,
    conv_filters=64, kernel_size=5, pool_size=2, conv_dropout=0.30,
    rnn_units=64, rnn_dropout=0.30, dense_units=32,
    optimizer="Adam", loss="binary_crossentropy",
    hybrid_topology="sequential (CNN feature extractor feeding recurrent layer)",
    threshold_selection="F1-optimal on inner calibration split, all models",
)

DL_MODELS = ["1D-CNN", "CNN", "GRU", "LSTM", "CNN+GRU", "CNN+LSTM"]

# =====================================================================
# 3. FORWARD MODEL  (Section 4.1 to 4.3)
# =====================================================================
def diffusion_impulse(t, D, d):
    t = np.maximum(t, 1e-9)
    return (1.0 / (4.0 * np.pi * D * t) ** 1.5) * np.exp(-(d ** 2) / (4.0 * D * t))


def channel_concentration(bits, p):
    T, fs = p["T_window"], p["fs_hz"]
    t = np.arange(T) / fs
    C = np.zeros(T)
    spb = max(int(round(p["Ts_s"] * fs)), 1)
    h = diffusion_impulse(t, p["D_cm2_s"], p["d_cm"])
    h = h / (h.max() + 1e-30)
    for k, b in enumerate(bits):
        if b == 0:
            continue
        s = k * spb
        if s >= T:
            break
        C[s:] += p["C_max_mol_cm3"] * h[: T - s]
    return C


def butler_volmer(C_ox, p, eta_offset=0.0, k0_scale=1.0, area_scale=1.0):
    n, T = p["n_electrons"], p["T_kelvin"]
    eta = p["eta_bias"] + eta_offset
    aa = p["alpha_a"]; ac = 1.0 - aa
    i0 = n * F_CONST * p["A_cm2"] * area_scale * p["k0_cm_s"] * k0_scale * np.maximum(C_ox, 0)
    return i0 * (np.exp(aa * n * F_CONST * eta / (R_GAS * T))
                 - np.exp(-ac * n * F_CONST * eta / (R_GAS * T)))


def noise_sigma(signal, p):
    return np.sqrt(np.mean(signal ** 2) / (10.0 ** (p["target_snr_db"] / 10.0)) + 1e-40)


# =====================================================================
# 4. ATTACK TRANSFORMATIONS  (Section 4.4)
# =====================================================================
def _mask(T, rng, min_frac=0.25, max_frac=0.75):
    dur = int(rng.uniform(min_frac, max_frac) * T)
    t0 = rng.integers(0, max(T - dur, 1))
    m = np.zeros(T, dtype=bool); m[t0:t0 + dur] = True
    return m, t0, dur


def apply_attack(x, clean, cls, p, rng):
    T, fs = p["T_window"], p["fs_hz"]
    m, t0, dur = _mask(T, rng)
    out = x.copy()

    if cls == "A1":                                    # signal injection
        beta = rng.uniform(0.30, 0.90)
        psi = int(0.37 * p["Ts_s"] * fs)
        out[m] += beta * np.roll(clean, psi)[m]

    elif cls == "A2":                                  # interfering species
        dE = rng.choice([-1, 1]) * rng.uniform(0.05, 0.15)
        n, Tk, aa = p["n_electrons"], p["T_kelvin"], p["alpha_a"]
        eta = p["eta_bias"] - dE
        i0i = rng.uniform(0.2, 0.6) * np.abs(clean).max()
        jam = i0i * (np.exp(aa * n * F_CONST * eta / (R_GAS * Tk))
                     - np.exp(-(1 - aa) * n * F_CONST * eta / (R_GAS * Tk)))
        out[m] += jam / (abs(jam) + 1e-30) * i0i

    elif cls == "A3":                                  # replay
        tau = int(rng.uniform(2.5, 6.0) * p["Ts_s"] * fs)
        out[m] = np.roll(x, tau)[m]

    elif cls == "A4":                                  # MITM scaling
        gamma = rng.choice([rng.uniform(0.45, 0.80), rng.uniform(1.25, 1.8)])
        delta = rng.uniform(-0.15, 0.15) * np.abs(clean).max()
        out[m] = gamma * out[m] + delta

    elif cls == "A5":                                  # potential perturbation
        dE = rng.choice([-1, 1]) * rng.uniform(0.03, 0.08)
        pert = butler_volmer(np.abs(clean) / (np.abs(clean).max() + 1e-30)
                             * p["C_max_mol_cm3"], p, eta_offset=dE)
        out[m] = pert[m] + (x - clean)[m]

    elif cls == "A6":                                  # depletion, fast
        lam = rng.uniform(3.0, 8.0) * p["rho_bio"]
        t = np.arange(T) / fs
        env = np.ones(T); env[t0:] = np.exp(-lam * (t[t0:] - t[t0]))
        out = out * env

    elif cls == "B1":                                  # fouling, slow (benign)
        lam = rng.uniform(0.2, 0.9) * p["rho_bio"]
        out = out * np.exp(-lam * np.arange(T) / fs)

    elif cls == "B2":                                  # pH / thermal (benign)
        dpH = rng.uniform(-0.4, 0.4)
        dE = -0.0592 * dpH / p["n_electrons"]
        drift = np.linspace(0, dE, T)
        n, Tk, aa = p["n_electrons"], p["T_kelvin"], p["alpha_a"]
        out = out * np.exp(aa * n * F_CONST * drift / (R_GAS * Tk))

    return out


# =====================================================================
# 5. CORPUS GENERATION  (Algorithm 1)
# =====================================================================
def generate_corpus(n_windows=6000, seed=42, params=None, class_weights=None):
    p = dict(PARAMS)
    if params:
        p.update(params)
    rng = np.random.default_rng(seed)

    if class_weights is None:
        class_weights = dict(benign=0.30, B1=0.18, B2=0.17, A1=0.07, A2=0.06,
                             A3=0.06, A4=0.06, A5=0.05, A6=0.05)
    names = list(class_weights)
    probs = np.array([class_weights[k] for k in names], float); probs /= probs.sum()

    T = p["T_window"]
    spb = max(int(round(p["Ts_s"] * p["fs_hz"])), 1)
    n_bits = int(np.ceil(T / spb)) + 1

    X = np.empty((n_windows, T)); y = np.empty(n_windows, dtype=np.int64)
    cls_arr = np.empty(n_windows, dtype=object)

    for j in range(n_windows):
        pj = dict(p)
        pj["A_cm2"] = p["A_cm2"] * rng.uniform(0.85, 1.15)
        pj["k0_cm_s"] = p["k0_cm_s"] * rng.uniform(0.7, 1.4)
        pj["T_kelvin"] = p["T_kelvin"] + rng.uniform(-1.0, 1.0)
        pj["d_cm"] = p["d_cm"] * rng.uniform(0.9, 1.1)

        bits = rng.integers(0, 2, n_bits)
        clean = butler_volmer(channel_concentration(bits, pj), pj)
        x = clean + rng.normal(0, noise_sigma(clean, pj), T)

        cls = names[rng.choice(len(names), p=probs)]
        if cls != "benign":
            x = apply_attack(x, clean, cls, pj, rng)

        X[j] = x; y[j] = 1 if cls in POSITIVE_CLASSES else 0; cls_arr[j] = cls

    X = X / (np.abs(X).max() + 1e-30)
    return X.astype(np.float32), y, cls_arr, p


def window_features(X):
    eps = 1e-12
    d1 = np.diff(X, axis=1)
    tc = np.arange(X.shape[1]) - (X.shape[1] - 1) / 2
    slope = (X * tc).sum(1) / (tc ** 2).sum()
    spec = np.abs(np.fft.rfft(X, axis=1)) ** 2
    sn = spec / (spec.sum(1, keepdims=True) + eps)
    ent = -(sn * np.log(sn + eps)).sum(1)
    b = spec.shape[1] // 4
    lo, hi = spec[:, :b].sum(1), spec[:, 3 * b:].sum(1)
    ac1 = np.array([np.corrcoef(r[:-1], r[1:])[0, 1] if r.std() > eps else 0.0
                    for r in X])
    F = np.column_stack([
        X.mean(1), X.std(1), X.min(1), X.max(1), np.ptp(X, axis=1),
        np.median(X, 1), sps.skew(X, axis=1), sps.kurtosis(X, axis=1),
        np.percentile(X, 25, axis=1), np.percentile(X, 75, axis=1),
        d1.mean(1), d1.std(1), np.abs(d1).max(1), slope, ent,
        np.log(lo + eps), np.log(hi + eps), np.log((hi + eps) / (lo + eps)),
        ac1, X.std(1) / (np.abs(X.mean(1)) + eps)])
    return np.nan_to_num(F).astype(np.float64)


# =====================================================================
# 6. FIDELITY CHECKS  (Section 4.7)
# =====================================================================
def _cv_simulate(nu, p, npts=1200, E_start=0.35, E_switch=-0.35):
    """Volterra integral-equation solver for a Nernstian CV."""
    n, T = p["n_electrons"], p["T_kelvin"]
    A, D, C = p["A_cm2"], p["D_cm2_s"], p["C_max_mol_cm3"]
    f = n * F_CONST / (R_GAS * T)
    half = npts // 2
    E = np.concatenate([np.linspace(E_start, E_switch, half),
                        np.linspace(E_switch, E_start, half)])
    dt = abs(E[1] - E[0]) / nu
    xi = f * (E - p["E0_prime"])
    C_surf = C * np.exp(xi) / (1.0 + np.exp(xi))
    m = n * F_CONST * A * np.sqrt(D) * (C - C_surf)
    N = len(E)
    w = np.sqrt(np.arange(1, N + 1)) - np.sqrt(np.arange(0, N))
    rhs = m * np.sqrt(np.pi) / (2.0 * np.sqrt(dt))
    i = np.zeros(N)
    for k in range(N):
        acc = np.dot(i[:k], w[k - np.arange(k)]) if k else 0.0
        i[k] = (rhs[k] - acc) / w[0]
    return E, i


def run_fidelity(p=None):
    p = p or PARAMS
    n, T = p["n_electrons"], p["T_kelvin"]
    A, D, C = p["A_cm2"], p["D_cm2_s"], p["C_max_mol_cm3"]
    k0, aa = p["k0_cm_s"], p["alpha_a"]
    f = n * F_CONST / (R_GAS * T)
    out = {}

    nus = np.array([0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0])
    ip = []
    for v in nus:
        E, i = _cv_simulate(v, p)
        ip.append(np.abs(i[: len(E) // 2]).max())
    r = sps.linregress(np.sqrt(nus), np.array(ip))
    theo = 2.69e5 * n ** 1.5 * A * np.sqrt(D) * C
    out["Randles-Sevcik"] = dict(r2=r.rvalue ** 2, slope=r.slope,
                                 theoretical_slope=theo,
                                 pct_dev=100 * abs(r.slope - theo) / theo)

    eta = np.linspace(0.12, 0.30, 60)
    r = sps.linregress(eta, np.log10(k0 * np.exp(aa * f * eta)))
    slope_mV = 1000.0 / r.slope
    a_rec = 2.303 * R_GAS * T / (F_CONST * slope_mV / 1000.0) / n
    out["Tafel"] = dict(r2=r.rvalue ** 2, tafel_slope_mV_per_decade=slope_mV,
                        alpha_recovered=a_rec, alpha_input=aa,
                        pct_dev=100 * abs(a_rec - aa) / aa)

    t = np.linspace(0.01, 5.0, 500)
    r = sps.linregress(t ** -0.5, n * F_CONST * A * np.sqrt(D) * C / np.sqrt(np.pi * t))
    theo = n * F_CONST * A * np.sqrt(D) * C / np.sqrt(np.pi)
    out["Cottrell"] = dict(r2=r.rvalue ** 2, slope=r.slope,
                           theoretical_slope=theo,
                           pct_dev=100 * abs(r.slope - theo) / theo)

    E, i = _cv_simulate(0.05, p, npts=3000)
    h = len(E) // 2
    Epc = E[:h][np.argmax(np.abs(i[:h]))]
    Epa = E[h:][np.argmax(np.abs(i[h:]))]
    dEp = abs(Epa - Epc) * 1000
    out["Peak separation"] = dict(dEp_mV=dEp, reversible_limit_mV=59.0 / n,
                                  pct_dev=100 * abs(dEp - 59.0 / n) / (59.0 / n),
                                  E_half_V=(Epa + Epc) / 2,
                                  E0_input_V=p["E0_prime"])
    for k, v in out.items():
        print(f"\n--- {k} ---")
        for kk, vv in v.items():
            print(f"  {kk:28s} {vv:.4e}" if abs(vv) < 1e-3 and vv != 0
                  else f"  {kk:28s} {vv:.4f}")
    return out


# =====================================================================
# 7. RESAMPLING  (Section 4.6)
# =====================================================================
def smote_feature_space(Xf, y, seed=0, k=5):
    rng = np.random.default_rng(seed)
    cnt = np.bincount(y); maj, mino = cnt.argmax(), cnt.argmin()
    need = cnt[maj] - cnt[mino]
    if need <= 0:
        return Xf, y
    Xm = Xf[y == mino]
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xm))).fit(Xm)
    _, idx = nn.kneighbors(Xm)
    i = rng.integers(0, len(Xm), need)
    j = idx[i, rng.integers(1, idx.shape[1], need)]
    Xn = Xm[i] + rng.random((need, 1)) * (Xm[j] - Xm[i])
    return np.vstack([Xf, Xn]), np.concatenate([y, np.full(need, mino)])


def window_oversample(X, y, seed=0, jitter=0.02):
    rng = np.random.default_rng(seed)
    cnt = np.bincount(y); maj, mino = cnt.argmax(), cnt.argmin()
    need = cnt[maj] - cnt[mino]
    if need <= 0:
        return X, y
    Xm = X[y == mino]
    i = rng.integers(0, len(Xm), need)
    Xn = Xm[i] * (1 + rng.normal(0, jitter, (need, 1)))
    sh = rng.integers(-3, 4, need)
    Xn = np.array([np.roll(r, s) for r, s in zip(Xn, sh)])
    return np.vstack([X, Xn]), np.concatenate([y, np.full(need, mino)])


# =====================================================================
# 8. STATISTICAL PROCESS CONTROL BASELINES
# =====================================================================
def cusum_detector(X, k=0.5):
    Z = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)
    ph = np.zeros(len(X)); pl = np.zeros(len(X)); mx = np.zeros(len(X))
    for t in range(X.shape[1]):
        ph = np.maximum(0, ph + Z[:, t] - k)
        pl = np.maximum(0, pl - Z[:, t] - k)
        mx = np.maximum(mx, np.maximum(ph, pl))
    return mx


def ewma_detector(X, lam=0.2):
    Z = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)
    e = np.zeros(len(X)); mx = np.zeros(len(X))
    for t in range(X.shape[1]):
        e = lam * Z[:, t] + (1 - lam) * e
        mx = np.maximum(mx, np.abs(e))
    return mx


# =====================================================================
# 9. ARCHITECTURES
# =====================================================================
def build_model(name, T):
    inp = Input(shape=(T, 1))
    k, f, p = HP["kernel_size"], HP["conv_filters"], HP["pool_size"]
    if name == "1D-CNN":
        x = Conv1D(f, k, activation="relu", padding="same")(inp)
        x = MaxPooling1D(p)(x)
        x = Conv1D(f, k, activation="relu", padding="same")(x)
        x = MaxPooling1D(p)(x)
        x = Dropout(HP["conv_dropout"])(x); x = Flatten()(x)
    elif name == "CNN":
        x = Conv1D(f, k, activation="relu", padding="same")(inp)
        x = MaxPooling1D(p)(x)
        x = Conv1D(f * 2, k, activation="relu", padding="same")(x)
        x = MaxPooling1D(p)(x)
        x = Conv1D(f * 2, k, activation="relu", padding="same")(x)
        x = MaxPooling1D(p)(x)
        x = Dropout(HP["conv_dropout"])(x); x = Flatten()(x)
    elif name == "GRU":
        x = GRU(HP["rnn_units"], dropout=HP["rnn_dropout"])(inp)
    elif name == "LSTM":
        x = LSTM(HP["rnn_units"], dropout=HP["rnn_dropout"])(inp)
    elif name == "CNN+GRU":
        x = Conv1D(f, k, activation="relu", padding="same")(inp)
        x = MaxPooling1D(p)(x); x = Dropout(HP["conv_dropout"])(x)
        x = GRU(HP["rnn_units"], dropout=HP["rnn_dropout"])(x)
    elif name == "CNN+LSTM":
        x = Conv1D(f, k, activation="relu", padding="same")(inp)
        x = MaxPooling1D(p)(x); x = Dropout(HP["conv_dropout"])(x)
        x = LSTM(HP["rnn_units"], dropout=HP["rnn_dropout"])(x)
    else:
        raise ValueError(name)
    x = Dense(HP["dense_units"], activation="relu")(x)
    m = Model(inp, Dense(1, activation="sigmoid")(x))
    m.compile(optimizer=Adam(HP["learning_rate"]), loss=HP["loss"],
              metrics=["accuracy"])
    return m


# =====================================================================
# 10. METRICS AND STATISTICS
# =====================================================================
def all_metrics(y_true, y_prob, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y_true, y_pred),
                precision=precision_score(y_true, y_pred, zero_division=0),
                recall=recall_score(y_true, y_pred, zero_division=0),
                f1=f1_score(y_true, y_pred, zero_division=0),
                fpr=fp / (fp + tn) if (fp + tn) else 0.0,
                roc_auc=roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan,
                mse=mean_squared_error(y_true, y_prob),
                mae=mean_absolute_error(y_true, y_prob),
                tn=tn, fp=fp, fn=fn, tp=tp)


def summarise(df, metric):
    rows = []
    for m, g in df.groupby("model", sort=False):
        v = g[metric].dropna().values; n = len(v)
        mu = v.mean(); sd = v.std(ddof=1) if n > 1 else 0.0
        half = sps.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else 0.0
        rows.append(dict(model=m, mean=mu, sd=sd,
                         ci_low=max(0.0, mu - half), ci_high=min(1.0, mu + half),
                         n_folds=n))
    return pd.DataFrame(rows)


def paired_tests(df, metric, reference):
    piv = df.pivot(index="fold", columns="model", values=metric)
    if reference not in piv.columns:
        return pd.DataFrame()
    out = []
    for m in piv.columns:
        if m == reference:
            continue
        a, b = piv[reference].values, piv[m].values
        ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
        d = a - b
        t_p = sps.ttest_rel(a, b).pvalue if len(a) > 1 and d.std() > 0 else 1.0
        try:
            w_p = sps.wilcoxon(a, b).pvalue
        except Exception:
            w_p = 1.0
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) or 1e-12
        out.append(dict(comparison=f"{reference} vs {m}", mean_diff=d.mean(),
                        cohens_d=d.mean() / pooled, ttest_p=t_p, wilcoxon_p=w_p))
    o = pd.DataFrame(out)
    for col in ["ttest_p", "wilcoxon_p"]:
        order = np.argsort(o[col].values); adj = np.empty(len(o)); run = 0.0
        for r, i in enumerate(order):
            run = max(run, (len(o) - r) * o[col].values[i]); adj[i] = min(run, 1.0)
        o[col + "_holm"] = adj
    o["significant_at_0.05"] = o["ttest_p_holm"] < 0.05
    return o


def best_threshold(y_cal, p_cal):
    grid = np.linspace(0.02, 0.98, 97)
    return float(grid[int(np.argmax(
        [f1_score(y_cal, (p_cal > t).astype(int), zero_division=0) for t in grid]))])


# =====================================================================
# 11. MAIN CROSS-VALIDATION PROTOCOL
# =====================================================================
def run_experiment(n_windows=6000, folds=10, seed=42,
                   resampling="window", outdir="results",
                   include_dl=True, resume=True, quiet=False):
    os.makedirs(outdir, exist_ok=True)
    X, y, cls, params = generate_corpus(n_windows=n_windows, seed=seed)
    Xf_all = window_features(X)
    T = X.shape[1]

    ckpt = f"{outdir}/_checkpoint_folds.csv"
    records, cms, done = [], {}, set()
    if resume and os.path.exists(ckpt):
        prev = pd.read_csv(ckpt)
        records = prev.to_dict("records"); done = set(prev["fold"].unique())
        for _, r in prev.iterrows():
            cms.setdefault(r["model"], np.zeros((2, 2), int))
            cms[r["model"]] += np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]], int)
        print(f"[resume] completed folds: {sorted(done)}")

    for fold, (tr, te) in enumerate(
            StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y), 1):
        if fold in done:
            continue
        print(f"--- fold {fold}/{folds} ---", flush=True)
        start = len(records)

        fit_i, cal_i = train_test_split(
            np.arange(len(tr)), test_size=HP["calibration_fraction"],
            stratify=y[tr], random_state=seed + fold)
        tr_fit, tr_cal = tr[fit_i], tr[cal_i]

        scf = StandardScaler().fit(Xf_all[tr_fit])
        Ffit, Fcal, Fte = (scf.transform(Xf_all[tr_fit]),
                           scf.transform(Xf_all[tr_cal]), scf.transform(Xf_all[te]))
        scr = StandardScaler().fit(X[tr_fit])
        Rfit, Rcal, Rte = (scr.transform(X[tr_fit]),
                           scr.transform(X[tr_cal]), scr.transform(X[te]))
        yfit, ycal, yte = y[tr_fit], y[tr_cal], y[te]

        # Three genuinely distinct conditions. The classical baselines are held
        # at feature-space SMOTE in conditions 2 and 3 so that only the
        # sequence-model treatment varies, which is what the ablation asks.
        if resampling == "none":
            Ffit_r, yf_r, Rfit_r, yr_r = Ffit, yfit, Rfit, yfit
        elif resampling == "smote_raw":
            # SMOTE interpolated directly between raw sequences. Included for
            # comparison only; this is the approach that disrupts temporal
            # dependency structure.
            Ffit_r, yf_r = smote_feature_space(Ffit, yfit, seed + fold)
            Rfit_r, yr_r = smote_feature_space(Rfit, yfit, seed + fold)
        elif resampling == "window":
            Ffit_r, yf_r = smote_feature_space(Ffit, yfit, seed + fold)
            Rfit_r, yr_r = window_oversample(Rfit, yfit, seed + fold)
        else:
            raise ValueError(f"unknown resampling condition: {resampling}")

        def log(name, p_cal, p_te):
            thr = best_threshold(ycal, p_cal)
            pred = (p_te > thr).astype(int)
            rec = all_metrics(yte, p_te, pred)
            rec.update(model=name, fold=fold, threshold=thr)
            records.append(rec)
            cms.setdefault(name, np.zeros((2, 2), int))
            cms[name] += confusion_matrix(yte, pred, labels=[0, 1])

        rf = RandomForestClassifier(300, random_state=seed, n_jobs=-1).fit(Ffit_r, yf_r)
        log("RandomForest", rf.predict_proba(Fcal)[:, 1], rf.predict_proba(Fte)[:, 1])

        iso = IsolationForest(random_state=seed, n_estimators=200).fit(Ffit[yfit == 0])
        s = -iso.score_samples(np.vstack([Fcal, Fte]))
        lo, hi = s.min(), np.ptp(s) + 1e-12
        log("IsolationForest", (-iso.score_samples(Fcal) - lo) / hi,
            (-iso.score_samples(Fte) - lo) / hi)

        oc = OneClassSVM(nu=0.1, gamma="scale").fit(Ffit[yfit == 0])
        s = -oc.decision_function(np.vstack([Fcal, Fte]))
        lo, hi = s.min(), np.ptp(s) + 1e-12
        log("OneClassSVM", (-oc.decision_function(Fcal) - lo) / hi,
            (-oc.decision_function(Fte) - lo) / hi)

        bf, bs, bsign = 0, -1.0, 1
        for c in range(Ffit.shape[1]):
            for sg in (1, -1):
                v = sg * Ffit[:, c]
                sc = f1_score(yfit, (v > np.median(v)).astype(int), zero_division=0)
                if sc > bs:
                    bs, bf, bsign = sc, c, sg
        ref = bsign * Ffit[:, bf]
        nv = lambda M: (bsign * M[:, bf] - ref.min()) / (np.ptp(ref) + 1e-12)
        log("NaiveThreshold", nv(Fcal), nv(Fte))

        for nm, fn_ in [("CUSUM", cusum_detector), ("EWMA", ewma_detector)]:
            sf = fn_(Rfit); lo, hi = sf.min(), np.ptp(sf) + 1e-12
            log(nm, (fn_(Rcal) - lo) / hi, (fn_(Rte) - lo) / hi)

        if include_dl and HAS_TF:
            for name in DL_MODELS:
                tf.keras.utils.set_random_seed(seed + fold)
                m = build_model(name, T)
                m.fit(Rfit_r[..., None], yr_r,
                      validation_data=(Rcal[..., None], ycal),
                      epochs=HP["epochs"], batch_size=HP["batch_size"],
                      callbacks=[
                          EarlyStopping(monitor="val_loss",
                                        patience=HP["early_stopping_patience"],
                                        restore_best_weights=True),
                          ReduceLROnPlateau(monitor="val_loss",
                                            factor=HP["lr_reduce_factor"],
                                            patience=HP["lr_reduce_patience"],
                                            min_lr=1e-6)],
                      verbose=0)
                log(name, m.predict(Rcal[..., None], verbose=0).ravel(),
                    m.predict(Rte[..., None], verbose=0).ravel())
                tf.keras.backend.clear_session()
                if not quiet:
                    print(f"    {name} done", flush=True)
        elif include_dl:
            print("  [tensorflow unavailable, deep models skipped]")

        pd.DataFrame(records).to_csv(ckpt, index=False)
        print(f"  [checkpoint saved, {len(records) - start} rows]", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(f"{outdir}/per_fold_metrics.csv", index=False)

    tab = []
    for met in ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc",
                "mse", "mae"]:
        s = summarise(df, met); s["metric"] = met; tab.append(s)
    tab = pd.concat(tab); tab.to_csv(f"{outdir}/table4_mean_ci.csv", index=False)

    print("\n=== Mean over folds ===")
    print(tab.pivot(index="model", columns="metric", values="mean").round(4).to_string())

    best = tab[tab.metric == "f1"].sort_values("mean").iloc[-1]["model"]
    print(f"\nBest model by mean F1: {best}")
    for met in ["accuracy", "f1", "roc_auc"]:
        s = paired_tests(df, met, best)
        if len(s):
            s.to_csv(f"{outdir}/significance_{met}.csv", index=False)
    s = paired_tests(df, "f1", best)
    if len(s):
        print(f"\n=== Paired tests vs {best} (F1) ===")
        print(s.round(4).to_string(index=False))

    json.dump({k: v.tolist() for k, v in cms.items()},
              open(f"{outdir}/confusion_matrices.json", "w"), indent=2)
    json.dump(dict(hyperparameters=HP, dataset_params=params,
                   n_windows=n_windows, folds=folds, seed=seed,
                   resampling=resampling),
              open(f"{outdir}/hyperparameters.json", "w"), indent=2, default=str)
    return df, tab


# =====================================================================
# 12. RESAMPLING ABLATION
# =====================================================================
def run_ablation(n_windows=4000, folds=5, seed=42, outdir="ablation"):
    os.makedirs(outdir, exist_ok=True)
    frames = []
    labels = {"none": "None",
              "smote_raw": "SMOTE on raw sequences",
              "window": "Window oversampling"}
    for mode in ["none", "smote_raw", "window"]:
        print(f"\n########## resampling = {mode} ##########", flush=True)
        df, _ = run_experiment(n_windows, folds, seed, mode,
                               f"{outdir}/{mode}", quiet=True)
        df["resampling"] = labels[mode]
        frames.append(df)
    allr = pd.concat(frames)
    allr.to_csv(f"{outdir}/smote_ablation.csv", index=False)
    piv = allr.groupby(["model", "resampling"])["f1"].agg(["mean", "std"]).round(4)
    piv.to_csv(f"{outdir}/ablation_summary.csv")
    print("\n=== Resampling ablation, F1 (mean, sd) ===")
    print(piv.unstack(level=1).to_string())
    print("\nExpected duplications, by construction:")
    print("  Isolation Forest, One-Class SVM, CUSUM, EWMA, naive threshold train")
    print("  on benign windows only, so minority resampling cannot affect them;")
    print("  their scores are identical across all three conditions.")
    print("  Random Forest receives feature-space SMOTE in both 'SMOTE on raw")
    print("  sequences' and 'Window oversampling', since only the sequence-model")
    print("  treatment is varied; its scores are identical in those two columns.")
    print("  The six deep architectures receive a distinct treatment in each of")
    print("  the three conditions.")
    return allr


# =====================================================================
# 13. FIGURES
# =====================================================================
def make_figures(outdir="results"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 9})
    CM = json.load(open(f"{outdir}/confusion_matrices.json"))
    cmap = LinearSegmentedColormap.from_list("b", ["#ffffff", "#2b5d8a"])
    lab = ["Benign", "Attack"]

    def draw(ax, cm, title, ylab=True, xlab=True):
        cm = np.array(cm, float); rn = cm / cm.sum(1, keepdims=True)
        ax.imshow(rn, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i,j]):,}\n({rn[i,j]*100:.1f}%)",
                        ha="center", va="center", fontsize=8,
                        color="white" if rn[i, j] > 0.55 else "#1a1a1a")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(lab, fontsize=8)
        ax.set_yticklabels(lab, fontsize=8, rotation=90, va="center")
        if xlab: ax.set_xlabel("Predicted", fontsize=8.5)
        if ylab: ax.set_ylabel("Actual", fontsize=8.5)
        ax.set_title(title, fontsize=9.5, pad=6); ax.tick_params(length=0)

    tab = pd.read_csv(f"{outdir}/table4_mean_ci.csv")
    best = tab[tab.metric == "f1"].sort_values("mean").iloc[-1]["model"]

    fig, ax = plt.subplots(figsize=(3.3, 3.1))
    draw(ax, CM[best], best)
    plt.tight_layout(); plt.savefig(f"{outdir}/Fig_confusion_best.png",
                                    dpi=600, bbox_inches="tight"); plt.close()

    order = [m for m in DL_MODELS if m in CM]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.0))
    for k, (ax, nm) in enumerate(zip(axes.ravel(), order)):
        draw(ax, CM[nm], nm, ylab=(k % 3 == 0), xlab=(k >= 3))
    plt.tight_layout(pad=1.1)
    plt.savefig(f"{outdir}/Fig_confusion_all.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"figures written to {outdir}/ (best model: {best})")


# =====================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="experiment",
                    choices=["fidelity", "experiment", "ablation", "figures"])
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    ap.add_argument("--resampling", default="window",
                    choices=["none", "smote_raw", "window"])
    ap.add_argument("--no-dl", action="store_true")
    a = ap.parse_args()

    if a.mode == "fidelity":
        run_fidelity()
    elif a.mode == "experiment":
        run_experiment(a.n, a.folds, a.seed, a.resampling, a.out,
                       include_dl=not a.no_dl)
        make_figures(a.out)
    elif a.mode == "ablation":
        run_ablation(a.n, a.folds, a.seed, a.out)
    else:
        make_figures(a.out)
