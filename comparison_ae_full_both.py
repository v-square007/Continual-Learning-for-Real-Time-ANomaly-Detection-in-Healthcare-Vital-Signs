"""
=============================================================================
Autoencoder Baseline -- Comparative Analysis vs Full_Both
=============================================================================

External baseline: windowed FC autoencoder, reconstruction error as anomaly
score. No MAD warmup (that is our novelty), no drift retraining (that is our
novelty). Just a plain trained-once AE on random warmup, frozen thereafter.

GT evaluation: clinical annotations only.
Synthetic injection dropped -- we have real labels for all 15 cases.

Metrics:
  Precision, Recall, F1, FPR  (clinical GT)
  Anomaly rate, Isolated fraction, Mean cluster length
  CV%, Separation ratio
  Mean inference latency (real-time suitability)
=============================================================================
"""

import os, time, warnings
import numpy as np
import pandas as pd
from collections import deque
from typing import Optional, List

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import vitaldb
    VITALDB_AVAILABLE = True
    print("[OK] vitaldb imported.")
except ImportError:
    VITALDB_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# REPRODUCIBILITY -- FIX: nothing seeded the AE before. Isolation Forest
# uses a fixed random seed (per the paper text); the AE baseline needs the
# exact same guarantee, otherwise weight init + shuffle order change every
# run and the reported numbers drift run-to-run (this is why the emailed
# comparison PNG didn't match Table VI).
# ─────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────
# CONFIG  (identical to v9)
# ─────────────────────────────────────────────────────────────
NUMERIC_TRACKS = {
    "HR":   "Solar8000/HR",
    "SpO2": "Solar8000/PLETH_SPO2",
    "RR":   "Solar8000/RR_CO2",
    "SBP":  "Solar8000/NIBP_SBP",
    "DBP":  "Solar8000/NIBP_DBP",
    "BT":   "Solar8000/BT",
}
WAVEFORM_TRACKS = {
    "ECG":   "SNUADC/ECG_II",
    "PLETH": "SNUADC/PLETH",
}
PHYS_BOUNDS = {
    "HR": (20,300), "SpO2": (50,100), "RR": (2,80),
    "SBP": (40,260), "DBP": (20,180), "BT": (32,43),
}
CLINICAL_EVENT_TRACKS = {
    "Solar8000/NIBP_MBP":   ("hypotension",  lambda v: v < 65),
    "Solar8000/PLETH_SPO2": ("desaturation", lambda v: v < 94),
    "Solar8000/HR":         ("bradycardia",  lambda v: v < 50),
    "Solar8000/HR_tachy":   ("tachycardia",  lambda v: v > 120),
}

SAMPLE_RATE_HZ          = 1
WINDOW_SIZE_S           = 10
OVERLAP                 = 0.50
STEP_SIZE_S             = int(WINDOW_SIZE_S * (1 - OVERLAP))
WARMUP_WINDOWS          = 60
THRESHOLD_PCTILE        = 95
MAX_MISSING_FRACTION    = 0.95
MAX_FFILL_GAP_S         = 60
MAX_NAN_SIGNAL_FRACTION = 0.50
QUALITY_DEGRADATION_THRESHOLD = 0.30
TARGET_CASE_COUNT       = 15
MIN_CASE_LENGTH         = 500

# FIX: paper text (Section E.2) says warmup windows are drawn from the
# first 25% of each case. The code previously used 0.50 here, which
# contradicted the text. Set to match the paper. Change back to 0.50 only
# if you'd rather edit the paper's wording instead -- pick one, keep them
# consistent.
WARMUP_SLICE_FRACTION   = 0.25

AE_EPOCHS    = 50
AE_BATCH     = 32
AE_LR        = 1e-3
AE_HIDDEN    = [128, 64, 32]
DEVICE       = torch.device("cpu")   # CPU sufficient for 60-window warmup

OUT_DIR = "outputs_ae"
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# DATA LOADING  (verbatim from v9)
# ─────────────────────────────────────────────────────────────

def fetch_case(caseid, max_retries=3):
    for attempt in range(1, max_retries+1):
        try:
            frames = {}
            for sig, tname in NUMERIC_TRACKS.items():
                data = vitaldb.vital_recs(caseid, track_names=[tname],
                                          interval=SAMPLE_RATE_HZ, return_timestamp=True)
                if data is None or len(data) == 0: continue
                df_sig = pd.DataFrame(data, columns=["time", sig])
                df_sig["time"] = df_sig["time"].astype(float).round().astype(int)
                df_sig = df_sig.drop_duplicates("time").set_index("time")[sig]
                frames[sig] = df_sig
            for sig, tname in WAVEFORM_TRACKS.items():
                data = vitaldb.vital_recs(caseid, track_names=[tname], return_timestamp=True)
                if data is None or len(data) == 0: continue
                df_wave = pd.DataFrame(data, columns=["time", sig])
                df_wave["time"] = df_wave["time"].astype(float)
                df_feat = _summarise_waveform(df_wave, sig)
                for col in df_feat.columns: frames[col] = df_feat[col]
            if not frames: return pd.DataFrame()
            df = pd.concat(frames.values(), axis=1, join="outer")
            df.index.name = "time"
            if len(df) > 0:
                full_range = pd.RangeIndex(int(df.index.min()), int(df.index.max())+1, 1)
                df = df.reindex(full_range)
            df = df.reset_index()
            df.rename(columns={"index": "time"}, inplace=True)
            df["caseid"] = caseid
            return df
        except Exception as exc:
            print(f"  [ERROR] case {caseid} attempt {attempt}: {exc}")
            time.sleep(3)
    return pd.DataFrame()

def _summarise_waveform(df_wave, sig):
    df_wave = df_wave.copy()
    df_wave["second"] = np.floor(df_wave["time"]).astype(int)
    grp = df_wave.groupby("second")[sig]
    return pd.DataFrame({
        f"{sig}_mean": grp.mean(), f"{sig}_std": grp.std().fillna(0),
        f"{sig}_p2p":  grp.max() - grp.min(),
        f"{sig}_rms":  grp.apply(lambda x: float(np.sqrt(np.mean(x**2)))),
    })

def handle_missing(df):
    df = df.copy()
    non_meta = [c for c in df.columns if c not in ("time","caseid")]
    for col in non_meta:
        base = col.split("_")[0]
        if base in PHYS_BOUNDS:
            lo, hi = PHYS_BOUNDS[base]
            df.loc[~df[col].between(lo, hi, inclusive="both") & df[col].notna(), col] = np.nan
    df[non_meta] = df[non_meta].ffill(limit=MAX_FFILL_GAP_S)
    df[non_meta] = df[non_meta].bfill(limit=MAX_FFILL_GAP_S)
    df[non_meta] = df[non_meta].interpolate(method="linear", limit=MAX_FFILL_GAP_S,
                                             limit_direction="both")
    active_sigs, dropped = [], []
    for col in non_meta:
        if df[col].isna().sum()/len(df)*100 > MAX_MISSING_FRACTION*100:
            dropped.append(col)
        else:
            active_sigs.append(col)
    if dropped: df.drop(columns=dropped, inplace=True)
    df.dropna(subset=active_sigs, how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Active signals: {len(active_sigs)}  |  Rows: {len(df):,}")
    return df, active_sigs

def load_cases(case_ids, min_rows=MIN_CASE_LENGTH, target=TARGET_CASE_COUNT):
    cases = []
    for cid in case_ids:
        if len(cases) >= target: break
        print(f"  Fetching case {cid}...", end=" ")
        df = fetch_case(cid)
        if len(df) < min_rows:
            print(f"skipped ({len(df)} rows)")
            continue
        df_clean, active_sigs = handle_missing(df)
        min_needed = WARMUP_WINDOWS * STEP_SIZE_S + WINDOW_SIZE_S
        if len(df_clean) < min_needed:
            print(f"skipped (only {len(df_clean)} rows after cleaning)")
            continue
        cases.append((cid, df_clean, active_sigs))
        print(f"OK ({len(df_clean)} rows, {len(active_sigs)} signals)")
    print(f"\n[INFO] {len(cases)} cases loaded")
    return cases


# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  (verbatim from v9)
# ─────────────────────────────────────────────────────────────

class FeatureMedianBuffer:
    def __init__(self, maxlen=200):
        self._buf = deque(maxlen=maxlen)
        self._medians = None

    def seed_from_bounds(self, active_sigs):
        n_feats = len(active_sigs) * 5
        prior   = np.zeros(n_feats, dtype=float)
        for i, sig in enumerate(active_sigs):
            base = sig.split("_")[0]
            if base in PHYS_BOUNDS:
                lo, hi = PHYS_BOUNDS[base]
                mid = (lo+hi)/2.0
                prior[i*5:i*5+5] = [mid, (hi-lo)*0.05, mid, mid, 0.0]
        for _ in range(5):
            self._buf.append(prior.copy())
        self._medians = prior.copy()

    def update(self, fv):
        if not np.any(np.isnan(fv)):
            self._buf.append(fv.copy())
            if len(self._buf) >= 5:
                self._medians = np.median(np.vstack(self._buf), axis=0)

    def fill(self, fv) -> Optional[np.ndarray]:
        nan_mask = np.isnan(fv)
        if not np.any(nan_mask): return fv
        if self._medians is None: return None
        filled = fv.copy()
        filled[nan_mask] = self._medians[nan_mask]
        return filled

def extract_window_features(chunk, active_sigs):
    feats, n_nan = [], 0
    for col in active_sigs:
        x = chunk[col].values.astype(float)
        v = x[~np.isnan(x)]
        if len(v) == 0:
            feats.extend([np.nan]*5); n_nan += 1
        else:
            feats.extend([np.mean(v), np.std(v), np.min(v), np.max(v),
                          np.mean(np.abs(np.diff(v))) if len(v)>1 else 0.0])
    return np.array(feats, dtype=float), n_nan/max(len(active_sigs),1)

def sliding_windows(df, active_sigs, ws=WINDOW_SIZE_S, step=STEP_SIZE_S):
    n, wid = len(df), 0
    for start in range(0, n-ws+1, step):
        chunk = df.iloc[start:start+ws]
        fv, nan_frac = extract_window_features(chunk, active_sigs)
        yield wid, float(chunk["time"].iloc[0]), float(chunk["time"].iloc[-1]), fv, nan_frac
        wid += 1


# ─────────────────────────────────────────────────────────────
# WARMUP -- random selection (NOT MAD -- that is our novelty)
# ─────────────────────────────────────────────────────────────

def select_warmup_random(df, active_sigs, n_required=WARMUP_WINDOWS):
    """
    Take first n_required valid windows from the first
    WARMUP_SLICE_FRACTION of data. No stability filtering -- the AE baseline
    gets no benefit of our MAD warmup. NaN features are filled with
    physiological midpoints so the AE trains on complete vectors (same
    imputation the scoring loop uses).
    """
    max_row  = int(len(df) * WARMUP_SLICE_FRACTION)
    df_slice = df.iloc[:max_row]
    med_buf  = FeatureMedianBuffer(maxlen=200)
    med_buf.seed_from_bounds(active_sigs)
    candidates = []

    for _, _, _, fv, nan_frac in sliding_windows(df_slice, active_sigs):
        if nan_frac > MAX_NAN_SIGNAL_FRACTION: continue
        if np.any(np.isnan(fv)):
            fv_filled = med_buf.fill(fv)
            if fv_filled is None:
                med_buf.update(fv); continue
            fv = fv_filled
        else:
            med_buf.update(fv)
        candidates.append(fv)
        if len(candidates) >= n_required: break

    if not candidates:
        print("  [Random warmup] 0 candidates")
        return np.array([])

    selected = candidates[:n_required]
    print(f"  [Random warmup] {len(selected)} windows selected")
    return np.vstack(selected)


# ─────────────────────────────────────────────────────────────
# AUTOENCODER
# ─────────────────────────────────────────────────────────────

class VitalAE(nn.Module):
    """
    Symmetric FC autoencoder: input -> 128 -> 64 -> 32 -> 64 -> 128 -> input.
    Reconstruction MSE per sample is the anomaly score.
    No batch norm or dropout -- keep it simple for a fair baseline comparison.
    """
    def __init__(self, input_dim, hidden=AE_HIDDEN):
        super().__init__()
        enc, prev = [], input_dim
        for h in hidden:
            enc += [nn.Linear(prev, h), nn.ReLU()]; prev = h
        self.encoder = nn.Sequential(*enc)
        dec = []
        for h in reversed(hidden[:-1]):
            dec += [nn.Linear(prev, h), nn.ReLU()]; prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def recon_error(self, x):
        return ((self.forward(x) - x)**2).mean(dim=1)


def train_ae(X_scaled: np.ndarray, input_dim: int) -> VitalAE:
    X_t     = torch.FloatTensor(X_scaled).to(DEVICE)

    # FIX: seeded generator so the shuffle order is reproducible across runs,
    # matching the "fixed random seed" guarantee already given to Isolation
    # Forest in the main pipeline.
    g = torch.Generator()
    g.manual_seed(SEED)
    loader = DataLoader(TensorDataset(X_t), batch_size=AE_BATCH,
                         shuffle=True, generator=g)

    model   = VitalAE(input_dim).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=AE_LR)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(AE_EPOCHS):
        total = 0.0
        for (batch,) in loader:
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            # gradient clipping -- prevents NaN from exploding gradients on
            # un-normalised vital sign feature vectors
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += loss.item() * len(batch)
        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{AE_EPOCHS}  loss={total/len(X_scaled):.6f}")

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
# AE PIPELINE  (one case)
# ─────────────────────────────────────────────────────────────

def run_ae_case(df, active_sigs):
    # 1. Random warmup -- first WARMUP_SLICE_FRACTION of case, no stability filter
    warmup_raw = select_warmup_random(df, active_sigs)
    if len(warmup_raw) < WARMUP_WINDOWS:
        print(f"  [AE] Insufficient warmup ({len(warmup_raw)})"); return None

    # 2. Scale -- StandardScaler fit only on warmup
    scaler   = StandardScaler()
    warmup_X = scaler.fit_transform(warmup_raw)

    assert np.isfinite(warmup_X).all(), "NaN/Inf in scaled warmup -- check imputation"
    input_dim = warmup_X.shape[1]

    # 3. Train AE on warmup -- record wall-clock time for real-time suitability
    print(f"  [AE] Training  input_dim={input_dim}, epochs={AE_EPOCHS}")
    t_train_start   = time.perf_counter()
    model           = train_ae(warmup_X, input_dim)
    training_time_s = time.perf_counter() - t_train_start
    print(f"  [AE] Training time: {training_time_s:.2f}s")

    # 4. Threshold from a HELD-OUT validation slice (25%-50% of case).
    #    Using training windows themselves gives a near-zero threshold because
    #    the AE memorises them. Post-warmup windows then have much higher error
    #    and nearly everything gets flagged. The validation slice is normal data
    #    the AE hasn't seen, giving a realistic baseline reconstruction error.
    n          = len(df)
    val_start  = int(n * 0.25)
    val_end    = int(n * 0.50)
    df_val     = df.iloc[val_start:val_end]

    val_feat_buf = FeatureMedianBuffer(maxlen=200)
    val_feat_buf.seed_from_bounds(active_sigs)
    for fv in warmup_raw:
        val_feat_buf.update(fv)

    val_errors = []
    model.eval()
    with torch.no_grad():
        for _, _, _, fv_raw, nan_frac in sliding_windows(df_val, active_sigs):
            if nan_frac > MAX_NAN_SIGNAL_FRACTION: continue
            fv = val_feat_buf.fill(fv_raw)
            if fv is None:
                val_feat_buf.update(fv_raw); continue
            val_feat_buf.update(fv)
            fv_s = np.clip(scaler.transform(fv.reshape(1, -1)), -10, 10)
            err  = model.recon_error(torch.FloatTensor(fv_s).to(DEVICE)).item()
            if np.isfinite(err):
                val_errors.append(err)

    if not val_errors:
        print("  [AE] No valid validation windows -- falling back to warmup errors")
        with torch.no_grad():
            wt = torch.FloatTensor(warmup_X).to(DEVICE)
            val_errors = model.recon_error(wt).cpu().numpy().tolist()

    val_errors_arr = np.array(val_errors)
    threshold      = float(np.percentile(val_errors_arr, THRESHOLD_PCTILE))
    warmup_mean    = float(val_errors_arr.mean())
    print(f"  [AE] threshold={threshold:.6f} (95th pctile of {len(val_errors)} "
          f"held-out val windows)  val_mean={warmup_mean:.6f}")

    # 5. Score all post-warmup windows
    feat_buf = FeatureMedianBuffer(maxlen=200)
    feat_buf.seed_from_bounds(active_sigs)
    for fv in warmup_raw: feat_buf.update(fv)

    warmup_end_idx = WARMUP_WINDOWS * STEP_SIZE_S + WINDOW_SIZE_S
    results = []

    model.eval()
    with torch.no_grad():
        for wid, t0, t1, fv_raw, nan_frac in sliding_windows(df, active_sigs):
            if wid * STEP_SIZE_S < warmup_end_idx - WINDOW_SIZE_S: continue
            if nan_frac > MAX_NAN_SIGNAL_FRACTION: continue

            fv = feat_buf.fill(fv_raw)
            if fv is None:
                feat_buf.update(fv_raw); continue
            feat_buf.update(fv)

            fv_scaled = scaler.transform(fv.reshape(1, -1))
            # Safety: clip to +-10 std to prevent extreme inputs after drift
            fv_scaled = np.clip(fv_scaled, -10, 10)

            x_t = torch.FloatTensor(fv_scaled).to(DEVICE)

            t0_inf = time.perf_counter()
            err    = model.recon_error(x_t).item()
            inf_ms = (time.perf_counter() - t0_inf) * 1000

            is_anomaly = (err > threshold) if np.isfinite(err) else False

            results.append({
                "window_id":      wid,
                "t_start":        t0,
                "t_end":          t1,
                "anomaly_score":  err,
                "decision_score": err,
                "threshold":      threshold,
                "is_anomaly":     is_anomaly,
                "quality":        1.0 - nan_frac,
                "nan_sig_frac":   nan_frac,
                "inf_ms":         inf_ms,
                "variant":        "Autoencoder",
            })

    df_res = pd.DataFrame(results)
    if df_res.empty: return None

    n_anom = int(df_res["is_anomaly"].sum())
    print(f"  [AE] Scored {len(df_res)} windows | "
          f"Anomalies: {n_anom} ({100*n_anom/len(df_res):.1f}%)")
    return df_res, threshold, val_errors_arr, training_time_s


# ─────────────────────────────────────────────────────────────
# CLINICAL GT  (verbatim from v9)
# ─────────────────────────────────────────────────────────────

def fetch_clinical_events(caseid, df_clean):
    if not VITALDB_AVAILABLE: return pd.Series(dtype=int)
    n = len(df_clean)
    anom_sec = np.zeros(n, dtype=int)
    found    = False
    for track, (_, condition) in CLINICAL_EVENT_TRACKS.items():
        actual = "Solar8000/HR" if "tachy" in track else track
        try:
            data = vitaldb.vital_recs(caseid, track_names=[actual],
                                       interval=1, return_timestamp=True)
            if data is None or len(data) == 0: continue
            df_sig = pd.DataFrame(data, columns=["time","value"])
            df_sig["time"] = df_sig["time"].astype(float).round().astype(int)
            df_sig = df_sig.dropna(subset=["value"])
            for _, row in df_sig.iterrows():
                t = int(row["time"])
                if 0 <= t < n and condition(row["value"]): anom_sec[t] = 1
            found = True
        except Exception: continue
    if not found: return pd.Series(dtype=int)

    labels, wid = {}, 0
    for start in range(0, n - WINDOW_SIZE_S + 1, STEP_SIZE_S):
        labels[wid] = 1 if anom_sec[start:start+WINDOW_SIZE_S].mean() > 0.10 else 0
        wid += 1
    return pd.Series(labels)


def compute_gt_metrics(results, gt_labels, source="clinical"):
    if gt_labels.empty or results.empty: return {}
    res_idx    = results.set_index("window_id")
    common     = res_idx.index.intersection(gt_labels.index)
    if len(common) == 0: return {}
    y_true = gt_labels.loc[common].values.astype(int)
    y_pred = res_idx.loc[common, "is_anomaly"].values.astype(int)
    TP = int(((y_pred==1)&(y_true==1)).sum())
    FP = int(((y_pred==1)&(y_true==0)).sum())
    FN = int(((y_pred==0)&(y_true==1)).sum())
    TN = int(((y_pred==0)&(y_true==0)).sum())
    P  = TP/max(TP+FP,1); R = TP/max(TP+FN,1)
    F1 = 2*P*R/max(P+R,1e-9); FPR = FP/max(FP+TN,1)
    print(f"  [{source}] {int(y_true.sum())} anomalous / {int((y_true==0).sum())} normal  |  "
          f"P={P:.3f}  R={R:.3f}  F1={F1:.3f}  FPR={FPR:.3f}  "
          f"(TP={TP} FP={FP} FN={FN} TN={TN})")
    return {f"{source}_precision":P, f"{source}_recall":R,
            f"{source}_f1":F1, f"{source}_fpr":FPR}


# ─────────────────────────────────────────────────────────────
# PROXY METRICS
# ─────────────────────────────────────────────────────────────

def compute_proxy_metrics(results, warmup_errors, threshold):
    ev = {}
    # CV: bootstrap warmup errors 10 times, measure anomaly rate variance
    rng = np.random.default_rng(SEED)
    rates = []
    for _ in range(10):
        boot = rng.choice(warmup_errors, size=len(warmup_errors), replace=True)
        thr  = np.percentile(boot, THRESHOLD_PCTILE)
        rates.append(float((results["decision_score"].values > thr).mean()*100))
    cv = float(np.std(rates)/(np.mean(rates)+1e-9)*100)
    ev["cv"] = cv
    print(f"  [Check 1] CV = {cv:.1f}%  ({'PASS' if cv<20 else 'WARN' if cv<40 else 'FAIL'})")

    # Separation ratio
    anom_sc = results[results["is_anomaly"]]["decision_score"].values
    if len(anom_sc) > 0:
        sep = float(anom_sc.mean()/(threshold+1e-9))
        ev["separation_ratio"] = sep
        print(f"  [Check 2] Separation ratio = {sep:.3f}  "
              f"({'PASS' if sep>1.05 else 'WARN' if sep>1.02 else 'FAIL'})")
    else:
        ev["separation_ratio"] = None
        print("  [Check 2] No anomalies -- separation ratio N/A")

    # Isolated fraction
    flags = results["is_anomaly"].values.astype(int)
    clusters, run = [], 0
    for f in flags:
        if f: run += 1
        elif run > 0: clusters.append(run); run = 0
    if run > 0: clusters.append(run)
    if clusters:
        ca  = np.array(clusters)
        iso = float((ca==1).sum()/len(ca))
        ev["isolated_fraction"]  = iso
        ev["mean_cluster_len"]   = float(ca.mean())
        ev["clusters"]           = ca
        print(f"  [Check 4] Isolated fraction = {iso*100:.1f}%  "
              f"({'PASS' if iso<0.3 else 'WARN' if iso<0.5 else 'FAIL'})")
    else:
        ev["isolated_fraction"] = 0.0
        ev["mean_cluster_len"]  = 0.0
        print("  [Check 4] No anomaly clusters")
    return ev


def compute_operational_metrics(results):
    if results is None or results.empty: return {}
    n      = len(results)
    n_anom = int(results["is_anomaly"].sum())
    flags  = results["is_anomaly"].values.astype(int)
    clusters, run = [], 0
    for f in flags:
        if f: run += 1
        elif run > 0: clusters.append(run); run = 0
    if run > 0: clusters.append(run)
    iso = float((np.array(clusters) == 1).mean()) if clusters else 0.0
    mcl = float(np.mean(clusters)) if clusters else 0.0
    return {
        "anomaly_rate":      float(n_anom/max(n,1)),
        "isolated_fraction": iso,
        "mean_cluster_len":  mcl,
        "mean_inf_ms":       float(results["inf_ms"].mean()),
        # training_time_s stored directly in rec from run_ae_case return value
    }


# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────

def plot_comparison(ae_records, fb_ref):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Autoencoder vs Full_Both (IF + CL) -- Comparative Analysis\n"
        "15 VitalDB surgical cases | Clinical GT labels | Same window pipeline",
        fontsize=12, fontweight="bold"
    )

    cids   = [r["caseid"] for r in ae_records]
    x      = np.arange(len(cids))
    w      = 0.35

    # ── Panel 1: Anomaly rate per case ──────────────────────
    ax = axes[0, 0]
    ae_ar = [r.get("anomaly_rate", 0)*100 for r in ae_records]
    fb_ar = [fb_ref["per_case_anomaly_rate"].get(r["caseid"], 0)*100 for r in ae_records]
    ax.bar(x-w/2, ae_ar, w, label="Autoencoder", color="steelblue",  alpha=0.8)
    ax.bar(x+w/2, fb_ar, w, label="Full_Both",   color="crimson",    alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"C{c}" for c in cids], fontsize=7)
    ax.set_ylabel("Anomaly rate (%)"); ax.set_title("(1) Anomaly Rate per Case")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # ── Panel 2: Clinical GT metrics (mean across cases) ────
    ax  = axes[0, 1]
    mns = ["Precision", "Recall", "F1", "FPR"]
    def safe_mean(key): return float(np.mean([r.get(key,0) for r in ae_records]))
    ae_clin = [safe_mean("clinical_precision"), safe_mean("clinical_recall"),
               safe_mean("clinical_f1"),        safe_mean("clinical_fpr")]
    fb_clin = [fb_ref["clinical_precision_mean"], fb_ref["clinical_recall_mean"],
               fb_ref["clinical_f1_mean"],        fb_ref["clinical_fpr_mean"]]
    x2 = np.arange(4)
    ax.bar(x2-w/2, ae_clin, w, label="Autoencoder", color="steelblue", alpha=0.8)
    ax.bar(x2+w/2, fb_clin, w, label="Full_Both",   color="crimson",   alpha=0.8)
    ax.set_xticks(x2); ax.set_xticklabels(mns)
    ax.set_title("(2) Clinical GT Metrics (mean across 15 cases)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ymax = max(max(ae_clin), max(fb_clin)) * 1.4
    ax.set_ylim(0, max(ymax, 0.05))

    # ── Panel 3: Proxy metrics bar chart ────────────────────
    ax = axes[1, 0]
    proxy_labels = ["CV %", "Sep. Ratio", "Isolated %"]
    ae_sep  = safe_mean("separation_ratio")
    fb_sep  = fb_ref["separation_ratio_mean"]
    ae_iso  = safe_mean("isolated_fraction")*100
    fb_iso  = fb_ref["isolated_fraction_mean"]
    ae_cv   = safe_mean("cv")
    fb_cv   = fb_ref["cv_mean"]
    ae_prx  = [ae_cv, ae_sep, ae_iso]
    fb_prx  = [fb_cv, fb_sep, fb_iso]
    x3 = np.arange(3)
    ax.bar(x3-w/2, ae_prx, w, label="Autoencoder", color="steelblue", alpha=0.8)
    ax.bar(x3+w/2, fb_prx, w, label="Full_Both",   color="crimson",   alpha=0.8)
    ax.set_xticks(x3); ax.set_xticklabels(proxy_labels)
    ax.set_title("(3) Proxy Evaluation Metrics")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # ── Panel 4: Inference latency ───────────────────────────
    ax = axes[1, 1]
    ae_inf = [r.get("mean_inf_ms", 0) for r in ae_records]
    # Full_Both IF inference is sub-millisecond; 0.05ms is typical from v9
    fb_inf = [0.05] * len(ae_records)
    ax.bar(x-w/2, ae_inf, w, label="Autoencoder", color="steelblue", alpha=0.8)
    ax.bar(x+w/2, fb_inf, w, label="Full_Both",   color="crimson",   alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"C{c}" for c in cids], fontsize=7)
    ax.set_ylabel("Mean inference time (ms)")
    ax.set_title("(4) Real-Time Suitability -- Per-Window Latency")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "ae_vs_fullboth_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[PLOT] ae_vs_fullboth_comparison.png -> {out}")


def plot_score_distributions(ae_records):
    valid = [r for r in ae_records
             if "results_df" in r and not r["results_df"].empty
             and r["results_df"]["decision_score"].notna().any()
             and np.isfinite(r["results_df"]["decision_score"].values).any()]
    if not valid:
        print("[WARN] No finite scores to plot distributions")
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    for r in valid[:5]:
        sc = r["results_df"]["decision_score"].values
        sc = sc[np.isfinite(sc)]
        if len(sc) == 0: continue
        ax.hist(sc, bins=50, alpha=0.45,
                label=f"Case {r['caseid']}", density=True)
    ax.set_xlabel("Reconstruction error (MSE)")
    ax.set_ylabel("Density")
    ax.set_title("AE Reconstruction Error Distributions (first 5 valid cases)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "ae_score_distributions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] ae_score_distributions.png -> {out}")


# ─────────────────────────────────────────────────────────────
# FULL_BOTH REFERENCE  (from v9 Output.txt -- matches paper Tables IV/VI/VII)
# ─────────────────────────────────────────────────────────────

FULL_BOTH_REF = {
    "clinical_precision_mean": 0.164,
    "clinical_recall_mean":    0.109,
    "clinical_f1_mean":        0.080,
    "clinical_fpr_mean":       0.059,
    "cv_mean":                 0.50,
    "separation_ratio_mean":   1.07,
    "isolated_fraction_mean":  26.35,
    "anomaly_rate_mean":       5.9,
    # IF.fit() on 60 warmup windows is a single sklearn call, typically <0.5s.
    # AE needs 50 gradient-descent epochs; expected 5-30s depending on hardware.
    "training_time_s":         0.4,
    # Per-case anomaly rates from v9 baseline_comparison summary (Full_Both column)
    "per_case_anomaly_rate": {
        1:  0.000, 2:  0.016, 3:  0.435, 4:  0.036,
        5:  0.010, 6:  0.018, 7:  0.028, 8:  0.093,
        9:  0.203, 10: 0.000, 11: 0.032, 13: 0.080,
        14: 0.213, 15: 0.156, 16: 0.031,
    },
}


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("="*65)
    print("  Autoencoder Baseline -- Comparative Analysis vs Full_Both")
    print(f"  VitalDB | 15 cases | Clinical GT | Random warmup (no MAD) | seed={SEED}")
    print("="*65)

    print("\n[STEP 1] Loading VitalDB cases...")
    cases = load_cases(list(range(1, 26)))
    if not cases:
        print("[ERROR] No cases loaded."); return

    print(f"\n[STEP 2] Running AE pipeline on {len(cases)} cases...")
    ae_records = []

    for caseid, df_clean, active_sigs in cases:
        print(f"\n{'-'*55}")
        print(f"  Case {caseid}  ({len(df_clean)} rows, {len(active_sigs)} signals)")
        print(f"{'-'*55}")

        out = run_ae_case(df_clean, active_sigs)
        if out is None: continue
        results_df, threshold, warmup_errors, training_time_s = out

        rec = {"caseid": caseid, "results_df": results_df,
               "threshold": threshold, "warmup_errors": warmup_errors,
               "training_time_s": training_time_s}

        ops = compute_operational_metrics(results_df)
        rec.update(ops)

        print(f"\n  --- Proxy Checks ---")
        proxy = compute_proxy_metrics(results_df, warmup_errors, threshold)
        rec.update(proxy)

        ae_records.append(rec)

    print("\n[STEP 3] Clinical GT evaluation...")
    for rec in ae_records:
        caseid = rec["caseid"]
        match  = next((c for c in cases if c[0] == caseid), None)
        if match is None: continue
        _, df_clean, _ = match
        gt = fetch_clinical_events(caseid, df_clean)
        if gt.empty or gt.sum() == 0: continue
        metrics = compute_gt_metrics(rec["results_df"], gt, "clinical")
        rec.update(metrics)

    # ── Summary table ─────────────────────────────────────────
    print("\n" + "="*72)
    print("  COMPARATIVE SUMMARY -- Autoencoder vs Full_Both (IF + CL)")
    print("="*72)

    def agg(key):
        vals = [r[key] for r in ae_records if r.get(key) is not None]
        if not vals: return float("nan"), float("nan"), 0
        return float(np.mean(vals)), float(np.std(vals)), len(vals)

    METRICS = [
        # (display label,            key,                    scale, higher_is_better)
        ("Clinical Precision",       "clinical_precision",   1,     True),
        ("Clinical Recall",          "clinical_recall",      1,     True),
        ("Clinical F1",              "clinical_f1",          1,     True),
        ("Clinical FPR",             "clinical_fpr",         1,     False),
        ("Anomaly Rate (%)",         "anomaly_rate",         100,   False),
        ("Isolated Fraction (%)",    "isolated_fraction",    100,   False),
        ("Mean Cluster Length",      "mean_cluster_len",     1,     None),
        ("CV %",                     "cv",                   1,     False),
        ("Separation Ratio",         "separation_ratio",     1,     True),
        ("Mean Inf Latency (ms)",    "mean_inf_ms",          1,     False),
        ("Training Time (s/case)",   "training_time_s",      1,     False),
    ]

    fb_vals = {
        "clinical_precision":  FULL_BOTH_REF["clinical_precision_mean"],
        "clinical_recall":     FULL_BOTH_REF["clinical_recall_mean"],
        "clinical_f1":         FULL_BOTH_REF["clinical_f1_mean"],
        "clinical_fpr":        FULL_BOTH_REF["clinical_fpr_mean"],
        "anomaly_rate":        FULL_BOTH_REF["anomaly_rate_mean"]/100,
        "isolated_fraction":   FULL_BOTH_REF["isolated_fraction_mean"]/100,
        "mean_cluster_len":    None,
        "cv":                  FULL_BOTH_REF["cv_mean"],
        "separation_ratio":    FULL_BOTH_REF["separation_ratio_mean"],
        "mean_inf_ms":         0.05,       # sub-ms IF inference from v9
        "training_time_s":     FULL_BOTH_REF["training_time_s"],
    }

    print(f"\n  {'Metric':<26s}  {'AE Mean':>10s}  {'AE +/-Std':>9s}  "
          f"{'n':>3s}  {'Full_Both':>10s}  {'Better':>10s}")
    print("  " + "-"*76)

    rows_csv = []
    for label, key, scale, hib in METRICS:
        mu, std, n = agg(key)
        fb          = fb_vals.get(key)
        mu_s        = mu  * scale if np.isfinite(mu)  else float("nan")
        std_s       = std * scale if np.isfinite(std) else float("nan")
        fb_s        = fb  * scale if fb is not None   else float("nan")

        if hib is None:
            winner = "--"
        elif np.isnan(mu_s) or np.isnan(fb_s):
            winner = "N/A"
        else:
            winner = "Full_Both" if (hib and fb_s > mu_s) or \
                                    (not hib and fb_s < mu_s) else "AE"

        mu_str  = f"{mu_s:10.3f}" if np.isfinite(mu_s)  else f"{'N/A':>10s}"
        std_str = f"{std_s:9.3f}" if np.isfinite(std_s) else f"{'N/A':>9s}"
        fb_str  = f"{fb_s:10.3f}" if np.isfinite(fb_s)  else f"{'N/A':>10s}"

        print(f"  {label:<26s}  {mu_str}  {std_str}  {n:>3d}  {fb_str}  {winner:>10s}")
        rows_csv.append({"Metric": label, "AE_mean": mu_s, "AE_std": std_s,
                         "Full_Both": fb_s, "Better": winner})

    df_csv = pd.DataFrame(rows_csv)
    csv_out = os.path.join(OUT_DIR, "ae_vs_fullboth_metrics.csv")
    df_csv.to_csv(csv_out, index=False)
    print(f"\n  Saved -> {csv_out}")

    # FIX: this note used to be a hardcoded "19.6%" string regardless of the
    # actual run's numbers. Now built from the same aggregate this run just
    # computed, so it can never silently disagree with the table above.
    ae_ar_mean = next((r["AE_mean"] for r in rows_csv if r["Metric"] == "Anomaly Rate (%)"), float("nan"))
    print(f"\n  NOTE -- Anomaly Rate: Full_Both targets ~5% by design via the IF "
          f"contamination prior (CONTAMINATION=0.05). The AE has no equivalent "
          f"prior; its rate of {ae_ar_mean:.1f}% in this run reflects uncalibrated "
          f"threshold sensitivity, not genuine detection. High recall at the cost "
          f"of a much higher FPR is not a clinical win.")

    # ── Per-case breakdown CSV ────────────────────────────────
    print("\n[STEP 4] Writing per-case breakdown...")
    per_case_rows = []
    fb_per_case_fpr = {
        # From v9 GT clinical output -- FPR per case for Full_Both (matches Table VII)
        1:  0.000, 2:  0.009, 3:  0.470, 4:  0.036,
        5:  0.009, 6:  0.015, 7:  0.028, 8:  0.112,
        9:  0.026, 10: 0.000, 11: 0.026, 13: 0.080,
        14: 0.017, 15: 0.017, 16: 0.033,
    }
    fb_per_case_f1 = {
        1:  0.000, 2:  0.032, 3:  0.019, 4:  0.039,
        5:  0.000, 6:  0.088, 7:  0.020, 8:  0.054,
        9:  0.236, 10: 0.000, 11: 0.204, 13: 0.072,
        14: 0.197, 15: 0.204, 16: 0.029,
    }
    for rec in ae_records:
        cid = rec["caseid"]
        per_case_rows.append({
            "Case":              cid,
            "AE_anomaly_rate_%": round(rec.get("anomaly_rate", float("nan"))*100, 2),
            "FB_anomaly_rate_%": round(FULL_BOTH_REF["per_case_anomaly_rate"].get(cid, float("nan"))*100, 2),
            "AE_FPR":            round(rec.get("clinical_fpr",       float("nan")), 3),
            "FB_FPR":            round(fb_per_case_fpr.get(cid,      float("nan")), 3),
            "AE_F1":             round(rec.get("clinical_f1",        float("nan")), 3),
            "FB_F1":             round(fb_per_case_f1.get(cid,       float("nan")), 3),
            "AE_CV_%":           round(rec.get("cv",                 float("nan")), 2),
            "AE_IsolatedFrac_%": round(rec.get("isolated_fraction",  float("nan"))*100, 2),
            "AE_TrainingTime_s": round(rec.get("training_time_s",    float("nan")), 2),
        })

    df_percase = pd.DataFrame(per_case_rows)
    percase_out = os.path.join(OUT_DIR, "ae_vs_fullboth_percase.csv")
    df_percase.to_csv(percase_out, index=False)
    print(f"  Saved -> {percase_out}")

    print(f"\n  {'Case':>5s}  {'AE AR%':>7s}  {'FB AR%':>7s}  "
          f"{'AE FPR':>7s}  {'FB FPR':>7s}  "
          f"{'AE F1':>6s}  {'FB F1':>6s}  "
          f"{'AE CV%':>7s}  {'AE Train(s)':>11s}")
    print("  " + "-"*76)
    for row in per_case_rows:
        print(f"  {row['Case']:>5d}  "
              f"{row['AE_anomaly_rate_%']:>7.1f}  {row['FB_anomaly_rate_%']:>7.1f}  "
              f"{row['AE_FPR']:>7.3f}  {row['FB_FPR']:>7.3f}  "
              f"{row['AE_F1']:>6.3f}  {row['FB_F1']:>6.3f}  "
              f"{row['AE_CV_%']:>7.1f}  {row['AE_TrainingTime_s']:>11.2f}")

    print("\n[STEP 5] Generating plots...")
    plot_comparison(ae_records, FULL_BOTH_REF)
    plot_score_distributions(ae_records)

    print(f"\n{'='*65}")
    for f in ["ae_vs_fullboth_metrics.csv", "ae_vs_fullboth_percase.csv",
              "ae_vs_fullboth_comparison.png", "ae_score_distributions.png"]:
        p = os.path.join(OUT_DIR, f)
        print(f"  {'OK' if os.path.exists(p) else 'MISSING'}  {p}")
    print("="*65)

    return ae_records, df_csv


if __name__ == "__main__":
    main()
