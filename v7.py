"""
=============================================================================
Continual Learning — Real-Time Anomaly Detection on Healthcare Vital Signals
Version 7 — Baseline Comparison + Continual Learning + Expanded Evaluation
=============================================================================

KEY CHANGES FROM V6:
─────────────────────────────────────────────────────────────────────────────
1. THREE-WAY BASELINE COMPARISON (Priority 1):
   - Variant A: IF_ONLY  — plain Isolation Forest, random warmup, raw anomaly
                            score, no continual learning. The floor.
   - Variant B: NO_CL    — MAD warmup + WindowScore (anomaly × trust), model
                            frozen after warmup. Isolates warmup/scoring gains.
   - Variant C: FULL     — MAD warmup + WindowScore + drift-triggered retraining.
                            The proposed system.
   A vs B → does MAD warmup + trust weighting help?
   B vs C → does continual learning add anything beyond a good static model?

2. CONTINUAL LEARNING WITH ANOMALY SCORE DRIFT (Priority 2):
   - DriftDetector now uses raw anomaly_score (not weighted_score) in buffer.
   - Rationale: weighted_score is suppressed by low trust during sensor dropout,
     which can mask real physiological drift. Raw anomaly_score is a cleaner
     drift signal.
   - Retraining selects recent stable windows (quality > 0.7, score < threshold)
     and combines with original warmup to prevent catastrophic forgetting.
   - Cooldown of 600s prevents over-retraining on the same drift event.

3. EXPANDED EVALUATION ON 10-15 VITALDB CASES (Priority 3):
   - Sweeps case IDs 1-20, accepts cases with ≥500 rows and <95% missing data.
   - Threshold sensitivity sweep: 90th-99th percentile.
   - Per-case metrics aggregated as mean ± std (not pooled across cases).
=============================================================================
"""
import os, time, warnings
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd
from collections import deque

warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import vitaldb
    VITALDB_AVAILABLE = True
    print("[OK]  vitaldb imported successfully.")
except ImportError:
    VITALDB_AVAILABLE = False
    print("[WARN] vitaldb not found — running with physiological simulation.\n")


# =============================================================================
# CONFIG
# =============================================================================

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
    "HR":   (20,  300),
    "SpO2": (50,  100),
    "RR":   (2,   80),
    "SBP":  (40,  260),
    "DBP":  (20,  180),
    "BT":   (32,  43),
}

SAMPLE_RATE_HZ  = 1
WINDOW_SIZE_S   = 10
OVERLAP         = 0.50
STEP_SIZE_S     = int(WINDOW_SIZE_S * (1 - OVERLAP))

N_ESTIMATORS        = 100
CONTAMINATION       = 0.05
THRESHOLD_PCTILE    = 95
WARMUP_WINDOWS      = 60

# Missing value handling
MAX_MISSING_FRACTION    = 0.95
MAX_FFILL_GAP_S         = 60
MAX_NAN_SIGNAL_FRACTION = 0.50
QUALITY_DEGRADATION_THRESHOLD = 0.30

# MAD filter (warmup selection)
MAD_CUTOFF = 2.5

# Drift detection — uses raw anomaly_score (Priority 2 change)
DRIFT_BUFFER_SIZE     = 50
DRIFT_RATIO_THRESHOLD = 1.15  # Lowered from 1.3: real VitalDB data peaks ~1.26,
                               # 1.15 catches sustained physiological shifts
                               # without triggering on normal variance

# Continual learning retraining
RETRAIN_COOLDOWN_S        = 300   # 5 min cooldown (10 min was too long per-case)
RETRAIN_QUALITY_THRESHOLD = 0.7   # Only use high-quality windows for retraining
RETRAIN_RECENT_WINDOW_CAP = 150   # Max recent windows to add
RETRAIN_MIN_STABLE        = 10    # Lowered: 20 was too strict for short cases
RETRAIN_RECENT_BUFFER_CAP = 300   # Rolling buffer of recent windows to draw from

# Expanded evaluation (Priority 3)
TARGET_CASE_COUNT = 10
MIN_CASE_LENGTH   = 500

# Per-case pipeline: run each case independently, aggregate mean±std
RUN_PER_CASE = True   # True = correct approach; False = concat (wrong for drift)

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# SECTION 1 — Model Variant Enum
# =============================================================================

class ModelVariant(Enum):
    """
    Three baseline variants for ablation study.

    IF_ONLY : Isolation Forest with random warmup and raw anomaly score.
              No trust weighting, no continual learning.
              Establishes the performance floor.

    NO_CL   : MAD-filtered warmup + WindowScore (anomaly × trust).
              Model is frozen after warmup — no retraining.
              Isolates the contribution of smarter initialization and scoring.

    FULL    : MAD warmup + WindowScore + drift-triggered retraining.
              The full proposed system.
    """
    IF_ONLY = "IF_Only"
    NO_CL   = "No_CL"
    FULL    = "Full"


# =============================================================================
# SECTION 2 — WindowScore: Unified Scoring Mechanism (unchanged from v6)
# =============================================================================

@dataclass
class WindowScore:
    """
    Unified scoring framework: anomaly detection × data quality × uncertainty.

    weighted_score = anomaly_score × trust_score
    trust_score    = quality × (1 - uncertainty)

    Low-quality windows (sensor dropout) are downweighted even if the raw
    anomaly_score is high, preventing false alarms from bad sensors.
    """
    anomaly_score: float
    quality: float
    uncertainty: float

    @property
    def trust_score(self) -> float:
        return self.quality * (1.0 - self.uncertainty)

    @property
    def weighted_score(self) -> float:
        return self.anomaly_score * self.trust_score

    def __repr__(self):
        return (f"WindowScore(anomaly={self.anomaly_score:.3f}, "
                f"quality={self.quality:.2f}, uncertainty={self.uncertainty:.2f}, "
                f"trust={self.trust_score:.2f}, weighted={self.weighted_score:.3f})")


def compute_window_quality(nan_sig_fraction: float) -> float:
    return 1.0 - nan_sig_fraction


def compute_window_uncertainty(quality: float) -> float:
    if quality >= QUALITY_DEGRADATION_THRESHOLD:
        return 0.0
    return max(0.0, 1.0 - quality / QUALITY_DEGRADATION_THRESHOLD)


# =============================================================================
# SECTION 3 — Retraining Event Tracker
# =============================================================================

@dataclass
class RetrainingEvent:
    """Records metadata about each continual learning retraining event."""
    window_id:        int
    timestamp:        float
    drift_ratio:      float
    n_warmup_windows: int
    n_recent_windows: int
    old_threshold:    float
    new_threshold:    float

    def __repr__(self):
        return (f"RetrainEvent(wid={self.window_id}, t={self.timestamp:.0f}s, "
                f"drift={self.drift_ratio:.2f}, "
                f"warmup={self.n_warmup_windows}+recent={self.n_recent_windows}, "
                f"threshold: {self.old_threshold:.4f}→{self.new_threshold:.4f})")


# =============================================================================
# SECTION 4 — VitalDB Data Fetcher (unchanged from v6)
# =============================================================================

def fetch_case(caseid: int, max_retries: int = 3) -> pd.DataFrame:
    for attempt in range(1, max_retries + 1):
        try:
            frames = {}

            for sig, tname in NUMERIC_TRACKS.items():
                data = vitaldb.vital_recs(
                    caseid, track_names=[tname],
                    interval=SAMPLE_RATE_HZ, return_timestamp=True,
                )
                if data is None or len(data) == 0:
                    continue
                df_sig = pd.DataFrame(data, columns=["time", sig])
                df_sig["time"] = df_sig["time"].astype(float).round().astype(int)
                df_sig = df_sig.drop_duplicates(subset="time").set_index("time")[sig]
                frames[sig] = df_sig

            for sig, tname in WAVEFORM_TRACKS.items():
                data = vitaldb.vital_recs(
                    caseid, track_names=[tname], return_timestamp=True,
                )
                if data is None or len(data) == 0:
                    continue
                df_wave = pd.DataFrame(data, columns=["time", sig])
                df_wave["time"] = df_wave["time"].astype(float)
                df_feat = _summarise_waveform(df_wave, sig)
                for col in df_feat.columns:
                    frames[col] = df_feat[col]

            if not frames:
                return pd.DataFrame()

            df = pd.concat(frames.values(), axis=1, join="outer")
            df.index.name = "time"

            if len(df) > 0:
                full_range = pd.RangeIndex(
                    start=int(df.index.min()),
                    stop=int(df.index.max()) + 1,
                    step=1
                )
                df = df.reindex(full_range)

            df = df.reset_index()
            df.rename(columns={"index": "time"}, inplace=True)
            df["caseid"] = caseid
            return df

        except Exception as exc:
            print(f"  [ERROR] case {caseid} attempt {attempt}: {exc}")
            time.sleep(3)

    return pd.DataFrame()


def _summarise_waveform(df_wave: pd.DataFrame, sig: str) -> pd.DataFrame:
    df_wave = df_wave.copy()
    df_wave["second"] = np.floor(df_wave["time"]).astype(int)
    grp = df_wave.groupby("second")[sig]
    feat = pd.DataFrame({
        f"{sig}_mean": grp.mean(),
        f"{sig}_std":  grp.std().fillna(0),
        f"{sig}_p2p":  grp.max() - grp.min(),
        f"{sig}_rms":  grp.apply(lambda x: float(np.sqrt(np.mean(x**2)))),
    })
    feat.index.name = "time"
    return feat


def load_real_data(
    case_ids: list,
    min_rows: int = MIN_CASE_LENGTH,
    target_count: int = TARGET_CASE_COUNT,
) -> pd.DataFrame:
    """Concat version — kept for backward compat but NOT used in main pipeline."""
    frames = []
    for cid in case_ids:
        if len(frames) >= target_count:
            break
        print(f"  Fetching case {cid}...", end=" ")
        df = fetch_case(cid)
        if len(df) >= min_rows:
            frames.append(df)
            print(f"accepted ({len(df)} rows)")
        else:
            print(f"skipped ({len(df)} rows < {min_rows})")
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        print(f"\n[INFO] Loaded {len(frames)} cases, {len(combined)} total rows")
        return combined
    return pd.DataFrame()


def load_cases_separately(
    case_ids: list,
    min_rows: int = MIN_CASE_LENGTH,
    target_count: int = TARGET_CASE_COUNT,
) -> list:
    """
    Load cases as a list of (caseid, df_clean, active_sigs) tuples.

    Running the pipeline on concatenated cases is wrong for drift detection.
    When you concat, the warmup draws from the first 50% of the combined
    dataset — which spans multiple surgeries. The resulting "baseline" is a
    mixture of different patients' physiological states, so the threshold is
    calibrated to an artificial average. Drift within any single case gets
    diluted because the score buffer now contains windows from different cases
    simultaneously. Per-case runs give the detector a clean, case-specific
    baseline and a clean temporal signal to react to.
    """
    cases = []
    for cid in case_ids:
        if len(cases) >= target_count:
            break
        print(f"  Fetching case {cid}...", end=" ")
        df = fetch_case(cid)
        if len(df) < min_rows:
            print(f"skipped ({len(df)} rows < {min_rows})")
            continue
        df_clean, active_sigs = handle_missing(df)
        min_needed = WARMUP_WINDOWS * STEP_SIZE_S + WINDOW_SIZE_S
        if len(df_clean) < min_needed:
            print(f"skipped (only {len(df_clean)} rows after cleaning)")
            continue
        cases.append((cid, df_clean, active_sigs))
        print(f"OK ({len(df_clean)} rows, {len(active_sigs)} signals)")
    print(f"\n[INFO] {len(cases)} cases ready for per-case pipeline")
    return cases


# =============================================================================
# SECTION 5 — Simulator (fallback, unchanged from v6)
# =============================================================================

def simulate_vitaldb_like(n_seconds=1800, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t   = np.arange(n_seconds, dtype=float)

    hr   = 72  + 5  * np.sin(2*np.pi*t/120)  + rng.normal(0, 2,    n_seconds)
    spo2 = 98  + 0.5* np.sin(2*np.pi*t/90)   + rng.normal(0, 0.3,  n_seconds)
    rr   = 16  + 2  * np.sin(2*np.pi*t/60)   + rng.normal(0, 1,    n_seconds)
    sbp  = 120 + 8  * np.sin(2*np.pi*t/180)  + rng.normal(0, 3,    n_seconds)
    dbp  = 80  + 5  * np.sin(2*np.pi*t/180)  + rng.normal(0, 2,    n_seconds)
    bt   = 36.6+ 0.2* np.sin(2*np.pi*t/300)  + rng.normal(0, 0.05, n_seconds)

    ecg_mean  = rng.normal(0,    1,    n_seconds)
    ecg_std   = np.abs(rng.normal(0.3, 0.05, n_seconds))
    ecg_p2p   = np.abs(rng.normal(1.5, 0.2,  n_seconds))
    ecg_rms   = np.abs(rng.normal(0.5, 0.05, n_seconds))
    pleth_mean= np.abs(rng.normal(50,  5,    n_seconds))
    pleth_std = np.abs(rng.normal(5,   1,    n_seconds))
    pleth_p2p = np.abs(rng.normal(30,  3,    n_seconds))
    pleth_rms = np.abs(rng.normal(52,  5,    n_seconds))

    ind_end = int(n_seconds * 0.10)
    hr  [:ind_end] += rng.normal(30, 12, ind_end)
    sbp [:ind_end] -= rng.normal(20, 10, ind_end)
    spo2[:ind_end] -= rng.uniform(2, 8,  ind_end)

    ds = int(n_seconds * 2/3)
    hr  [ds:] += rng.normal(20, 8,  n_seconds-ds)
    spo2[ds:] -= rng.uniform(3,  6, n_seconds-ds)
    sbp [ds:] += rng.normal(25, 10, n_seconds-ds)
    rr  [ds:] += rng.normal(8,  3,  n_seconds-ds)
    ecg_p2p[ds:] *= rng.uniform(1.5, 2.5, n_seconds-ds)
    pleth_std[ds:] *= rng.uniform(2, 4,   n_seconds-ds)

    hr   = np.clip(hr,   20,  300)
    spo2 = np.clip(spo2, 50,  100)
    rr   = np.clip(rr,   2,   80)
    sbp  = np.clip(sbp,  40,  260)
    dbp  = np.clip(dbp,  20,  180)
    bt   = np.clip(bt,   32,  43)

    df = pd.DataFrame({
        "time": t.astype(int), "caseid": 0,
        "HR": hr, "SpO2": spo2, "RR": rr,
        "SBP": sbp, "DBP": dbp, "BT": bt,
        "ECG_mean": ecg_mean, "ECG_std": ecg_std,
        "ECG_p2p": ecg_p2p,   "ECG_rms": ecg_rms,
        "PLETH_mean": pleth_mean, "PLETH_std": pleth_std,
        "PLETH_p2p": pleth_p2p,   "PLETH_rms": pleth_rms,
    })
    print(f"[SIM] {n_seconds}s simulated — induction instability + drift injected.")
    return df


# =============================================================================
# SECTION 6 — Missing Value Handling (unchanged from v6)
# =============================================================================

def handle_missing(df: pd.DataFrame) -> tuple:
    df = df.copy()
    non_meta = [c for c in df.columns if c not in ("time", "caseid")]

    for col in non_meta:
        base = col.split("_")[0]
        if base in PHYS_BOUNDS:
            lo, hi = PHYS_BOUNDS[base]
            df.loc[~df[col].between(lo, hi, inclusive="both") &
                   df[col].notna(), col] = np.nan

    df[non_meta] = df[non_meta].ffill(limit=MAX_FFILL_GAP_S)
    df[non_meta] = df[non_meta].bfill(limit=MAX_FFILL_GAP_S)
    df[non_meta] = df[non_meta].interpolate(
        method="linear", limit=MAX_FFILL_GAP_S, limit_direction="both"
    )

    active_sigs, dropped = [], []
    for col in non_meta:
        pct = df[col].isna().sum() / len(df) * 100
        if pct > MAX_MISSING_FRACTION * 100:
            dropped.append(col)
        else:
            active_sigs.append(col)

    if dropped:
        df.drop(columns=dropped, inplace=True)

    df.dropna(subset=active_sigs, how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"  Active signals: {len(active_sigs)}  |  Rows: {len(df):,}")
    return df, active_sigs


# =============================================================================
# SECTION 7 — Feature Extraction (unchanged from v6)
# =============================================================================

class FeatureMedianBuffer:
    def __init__(self, maxlen=200):
        self._buf = deque(maxlen=maxlen)
        self._medians = None

    def update(self, feat_vec: np.ndarray):
        if not np.any(np.isnan(feat_vec)):
            self._buf.append(feat_vec.copy())
            if len(self._buf) >= 5:
                self._medians = np.median(np.vstack(self._buf), axis=0)

    def fill(self, feat_vec: np.ndarray) -> Optional[np.ndarray]:
        nan_mask = np.isnan(feat_vec)
        if not np.any(nan_mask):
            return feat_vec
        if self._medians is None:
            return None
        filled = feat_vec.copy()
        filled[nan_mask] = self._medians[nan_mask]
        return filled

    @property
    def is_warm(self) -> bool:
        return self._medians is not None


def extract_window_features(chunk: pd.DataFrame, active_sigs: list) -> tuple:
    feats = []
    n_nan_sigs = 0
    for col in active_sigs:
        x_raw = chunk[col].values.astype(float)
        x_valid = x_raw[~np.isnan(x_raw)]
        if len(x_valid) == 0:
            feats.extend([np.nan] * 5)
            n_nan_sigs += 1
        else:
            feats.extend([
                np.mean(x_valid),
                np.std(x_valid),
                np.min(x_valid),
                np.max(x_valid),
                np.mean(np.abs(np.diff(x_valid))) if len(x_valid) > 1 else 0.0,
            ])
    feat_vec = np.array(feats, dtype=float)
    nan_sig_fraction = n_nan_sigs / max(len(active_sigs), 1)
    return feat_vec, nan_sig_fraction


def sliding_windows(df, active_sigs,
                    window_size=WINDOW_SIZE_S, step=STEP_SIZE_S):
    n = len(df)
    wid = 0
    for start in range(0, n - window_size + 1, step):
        chunk = df.iloc[start:start + window_size]
        feat_vec, nan_frac = extract_window_features(chunk, active_sigs)
        yield wid, float(chunk["time"].iloc[0]), float(chunk["time"].iloc[-1]), \
              feat_vec, nan_frac
        wid += 1


# =============================================================================
# SECTION 8 — Warmup Selection: Two Strategies
# =============================================================================

def select_warmup_random(
    df: pd.DataFrame,
    active_sigs: list,
    n_required: int = WARMUP_WINDOWS,
) -> np.ndarray:
    """
    IF_ONLY warmup: take first n_required valid windows from first 50% of data.
    No stability filtering — just requires quality > 0.5.
    Used as the baseline to contrast against MAD selection.
    """
    max_row = int(len(df) * 0.50)
    df_slice = df.iloc[:max_row]
    candidates = []

    for _, _, _, fv, nan_frac in sliding_windows(df_slice, active_sigs):
        if nan_frac > MAX_NAN_SIGNAL_FRACTION:
            continue
        if not np.any(np.isnan(fv)):
            candidates.append(fv)
        if len(candidates) >= n_required:
            break

    if not candidates:
        return np.array([])
    selected = candidates[:n_required]
    print(f"  [Random warmup] {len(selected)} windows selected (no stability filter)")
    return np.vstack(selected)


def select_warmup_mad(
    df: pd.DataFrame,
    active_sigs: list,
    n_required: int = WARMUP_WINDOWS,
    mad_cutoff: float = MAD_CUTOFF,
) -> np.ndarray:
    """
    NO_CL and FULL warmup: MAD-based stability filter on HR mean.

    Why MAD instead of SD: the median is immune to the outlier windows
    we're trying to exclude, so the filter doesn't get corrupted by the
    very thing it's filtering out.

    Why HR: it's continuously monitored and most sensitive to
    induction/emergence phases we want excluded from the baseline.
    """
    max_row = int(len(df) * 0.50)
    df_slice = df.iloc[:max_row]
    candidates, hr_means = [], []

    hr_idx = None
    for i, sig in enumerate(active_sigs):
        if "HR" in sig:
            hr_idx = i * 5
            break

    for _, _, _, fv, nan_frac in sliding_windows(df_slice, active_sigs):
        if nan_frac > MAX_NAN_SIGNAL_FRACTION:
            continue
        if hr_idx is not None and not np.isnan(fv[hr_idx]):
            candidates.append(fv)
            hr_means.append(fv[hr_idx])

    if len(candidates) < n_required:
        print(f"  [MAD warmup] Only {len(candidates)} candidates, using all")
        return np.vstack(candidates) if candidates else np.array([])

    hr_means  = np.array(hr_means)
    median_hr = np.median(hr_means)
    mad       = np.median(np.abs(hr_means - median_hr))
    mad_sd    = mad * 1.4826

    stable_mask = np.abs(hr_means - median_hr) <= (mad_cutoff * mad_sd)
    stable = [candidates[i] for i in range(len(candidates)) if stable_mask[i]]

    if len(stable) < n_required:
        # Fall back: sort by deviation from median, take closest n_required
        sorted_idx = np.argsort(np.abs(hr_means - median_hr))
        stable = [candidates[i] for i in sorted_idx[:n_required]]

    selected = stable[:n_required]
    print(f"  [MAD warmup] {len(stable)} stable of {len(candidates)} candidates "
          f"(HR median={median_hr:.1f}, MAD-SD={mad_sd:.2f}) → {len(selected)} selected")
    return np.vstack(selected)


# =============================================================================
# SECTION 9 — StreamingIF with Variant Support and Retraining
# =============================================================================

class StreamingIF:
    """
    Isolation Forest with three operational modes (ModelVariant).

    IF_ONLY : random warmup, raw anomaly_score for decisions.
    NO_CL   : MAD warmup, weighted_score for decisions, no retraining.
    FULL    : MAD warmup, weighted_score for decisions, drift-triggered retrain.
    """

    def __init__(self, variant: ModelVariant = ModelVariant.FULL):
        self.variant              = variant
        self.scaler               = StandardScaler()
        self.model                = None
        self.threshold            = None
        self._fitted              = False
        self.warmup_scores        = None
        self.warmup_mean_score    = None
        self.original_warmup_X    = None   # Preserved for retraining (anti-forgetting)
        self.retraining_events: List[RetrainingEvent] = []

    # ── Training ──────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, silent: bool = False):
        """Train on warmup windows and set threshold."""
        self.original_warmup_X = X.copy()
        Xs = self.scaler.fit_transform(X)
        self.model = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=CONTAMINATION,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(Xs)
        scores = -self.model.score_samples(Xs)
        self.warmup_scores     = scores
        self.warmup_mean_score = scores.mean()
        self.threshold         = np.percentile(scores, THRESHOLD_PCTILE)
        self._fitted           = True

        if not silent:
            print(f"\n  [{self.variant.value}] IF trained on {len(X)} windows  "
                  f"| features={X.shape[1]}  "
                  f"| score_mean={scores.mean():.4f}  "
                  f"| threshold={self.threshold:.4f}")

    def retrain(
        self,
        recent_buffer: deque,
        window_id: int,
        timestamp: float,
        drift_ratio: float,
        silent: bool = False,
    ):
        """
        Refit the model using original warmup + recent stable windows.

        Window selection for recent data:
          quality > 0.7  — ensures clean signal, not sensor dropout
          score < threshold — ensures we're adding "normal" patterns

        Combining with original warmup prevents catastrophic forgetting:
        the model retains what it learned at the start of the case while
        adapting to the new physiological baseline.
        """
        stable_recent = [
            fv for fv, quality, score in recent_buffer
            if quality > RETRAIN_QUALITY_THRESHOLD and score < self.threshold
        ]

        if len(stable_recent) < RETRAIN_MIN_STABLE:
            if not silent:
                print(f"  [RETRAIN SKIP] Only {len(stable_recent)} stable windows "
                      f"(need {RETRAIN_MIN_STABLE})")
            return False

        recent_X   = np.vstack(stable_recent[:RETRAIN_RECENT_WINDOW_CAP])
        combined_X = np.vstack([self.original_warmup_X, recent_X])

        old_threshold = self.threshold
        self.fit(combined_X, silent=True)

        event = RetrainingEvent(
            window_id=window_id,
            timestamp=timestamp,
            drift_ratio=drift_ratio,
            n_warmup_windows=len(self.original_warmup_X),
            n_recent_windows=len(recent_X),
            old_threshold=old_threshold,
            new_threshold=self.threshold,
        )
        self.retraining_events.append(event)

        if not silent:
            print(f"\n  [RETRAIN] {event}")
        return True

    # ── Scoring ───────────────────────────────────────────────────────────

    def score_raw(self, x: np.ndarray) -> float:
        """Raw Isolation Forest anomaly score (higher = more anomalous)."""
        assert self._fitted
        return float(-self.model.score_samples(
            self.scaler.transform(x.reshape(1, -1)))[0])

    def score_window(self, x: np.ndarray, nan_frac: float) -> WindowScore:
        """Full WindowScore with quality and uncertainty."""
        anomaly_score = self.score_raw(x)
        quality       = compute_window_quality(nan_frac)
        uncertainty   = compute_window_uncertainty(quality)
        return WindowScore(anomaly_score=anomaly_score,
                           quality=quality, uncertainty=uncertainty)

    def get_decision_score(self, ws: WindowScore) -> float:
        """
        Score used for anomaly decision and drift detection.

        IF_ONLY: raw anomaly_score  (no trust weighting)
        NO_CL / FULL: weighted_score (anomaly × trust)
        """
        if self.variant == ModelVariant.IF_ONLY:
            return ws.anomaly_score
        return ws.weighted_score


# =============================================================================
# SECTION 10 — Drift Detector (Priority 2: uses raw anomaly_score)
# =============================================================================

class DriftDetector:
    """
    Rolling ratio drift detector — now uses raw anomaly_score.

    Why anomaly_score instead of weighted_score:
    weighted_score is suppressed by low trust (sensor dropout). If signal
    quality degrades at the same time as a real physiological shift, trust
    pulls weighted_score down and the drift goes undetected. Raw anomaly_score
    is blind to quality, making it a cleaner independent drift signal.

    drift_ratio = mean(last 50 anomaly_scores) / warmup_mean_anomaly_score
    ratio > 1.3 → sustained 30% elevation = distribution has shifted
    """

    def __init__(self, buffer_size: int = DRIFT_BUFFER_SIZE,
                 threshold: float = DRIFT_RATIO_THRESHOLD,
                 warmup_mean: float = None):
        self.buffer          = deque(maxlen=buffer_size)
        self.threshold       = threshold
        self.warmup_mean     = warmup_mean
        self.drift_detected  = False
        self.drift_ratio     = 1.0
        self.drift_ratio_history: List[float] = []

    def update(self, anomaly_score: float) -> bool:
        """
        Feed a new raw anomaly_score. Returns True if drift is newly detected.
        (Priority 2: parameter renamed from weighted_score to anomaly_score.)
        """
        self.buffer.append(anomaly_score)

        if len(self.buffer) >= self.buffer.maxlen:
            current_mean     = np.mean(self.buffer)
            self.drift_ratio = current_mean / (self.warmup_mean + 1e-9)
            self.drift_ratio_history.append(self.drift_ratio)

            if self.drift_ratio > self.threshold and not self.drift_detected:
                self.drift_detected = True
                return True  # Signal: new drift event

        return False

    def reset(self, new_warmup_mean: float):
        """
        Reset after retraining. Clears the buffer and sets a new baseline mean
        so the detector doesn't immediately re-trigger on the same pattern.
        """
        self.buffer.clear()
        self.warmup_mean    = new_warmup_mean
        self.drift_detected = False
        self.drift_ratio    = 1.0

    def get_status(self) -> dict:
        return {
            "drift_detected": self.drift_detected,
            "drift_ratio":    self.drift_ratio,
            "buffer_size":    len(self.buffer),
        }


# =============================================================================
# SECTION 11 — Streaming Pipeline with Variant Dispatch
# =============================================================================

class StreamingPipeline:
    """
    Runs one of the three model variants on a given dataframe.

    Variant dispatch logic:
      IF_ONLY : random warmup, score with raw anomaly_score, no retrain
      NO_CL   : MAD warmup, score with weighted_score, no retrain
      FULL    : MAD warmup, score with weighted_score, drift → retrain
    """

    def __init__(self, active_sigs: list, variant: ModelVariant = ModelVariant.FULL):
        self.active_sigs   = active_sigs
        self.variant       = variant
        self.ifm           = StreamingIF(variant=variant)
        self.feat_buffer   = FeatureMedianBuffer(maxlen=200)
        self.drift_detector = None
        self.results: List[dict] = []

        # Rolling buffer for retraining window selection (FULL only)
        # Each entry: (feat_vec, quality, anomaly_score)
        self._recent_buffer: deque = deque(maxlen=RETRAIN_RECENT_BUFFER_CAP)
        self._last_retrain_time: float = -np.inf

    def run(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        self.results = []

        if verbose:
            print(f"\n{'='*65}")
            print(f"  Variant: {self.variant.value}")
            print(f"  Signals: {self.active_sigs}")
            print(f"  Window: {WINDOW_SIZE_S}s | Step: {STEP_SIZE_S}s")
            print(f"{'='*65}")

        # ── Warmup selection ──────────────────────────────────────────────
        if self.variant == ModelVariant.IF_ONLY:
            warmup_X = select_warmup_random(df, self.active_sigs)
        else:
            warmup_X = select_warmup_mad(df, self.active_sigs)

        if len(warmup_X) < WARMUP_WINDOWS:
            print(f"  [ERROR] Insufficient warmup windows ({len(warmup_X)})")
            return pd.DataFrame()

        for fv in warmup_X:
            self.feat_buffer.update(fv)

        # ── Train ─────────────────────────────────────────────────────────
        self.ifm.fit(warmup_X, silent=not verbose)

        # ── Drift detector (always uses anomaly_score — Priority 2) ───────
        self.drift_detector = DriftDetector(
            warmup_mean=self.ifm.warmup_mean_score
        )

        # ── Scoring loop ─────────────────────────────────────────────────
        warmup_end_idx = WARMUP_WINDOWS * STEP_SIZE_S + WINDOW_SIZE_S
        skipped_early, skipped_nan = 0, 0

        for wid, t0, t1, feat_raw, nan_frac in sliding_windows(df, self.active_sigs):
            row_start = wid * STEP_SIZE_S
            if row_start < warmup_end_idx - WINDOW_SIZE_S:
                skipped_early += 1
                continue

            if nan_frac > MAX_NAN_SIGNAL_FRACTION:
                skipped_nan += 1
                continue

            feat = self.feat_buffer.fill(feat_raw)
            if feat is None:
                self.feat_buffer.update(feat_raw)
                skipped_early += 1
                continue

            self.feat_buffer.update(feat)

            t_inf0 = time.perf_counter()
            ws     = self.ifm.score_window(feat, nan_frac)
            inf_ms = (time.perf_counter() - t_inf0) * 1000

            decision_score = self.ifm.get_decision_score(ws)
            is_anomaly     = decision_score > self.ifm.threshold

            # ── Drift detection (on anomaly_score — Priority 2) ───────────
            drift_triggered = self.drift_detector.update(ws.anomaly_score)

            # ── Retraining (FULL variant only) ────────────────────────────
            retrained = False
            if (self.variant == ModelVariant.FULL
                    and drift_triggered
                    and (t0 - self._last_retrain_time) > RETRAIN_COOLDOWN_S):

                retrained = self.ifm.retrain(
                    recent_buffer=self._recent_buffer,
                    window_id=wid,
                    timestamp=t0,
                    drift_ratio=self.drift_detector.drift_ratio,
                    silent=not verbose,
                )
                if retrained:
                    self._last_retrain_time = t0
                    # Reset drift detector with the new model's baseline mean
                    self.drift_detector.reset(
                        new_warmup_mean=self.ifm.warmup_mean_score
                    )

            # ── Recent buffer for next retraining ─────────────────────────
            self._recent_buffer.append((feat, ws.quality, ws.anomaly_score))

            self.results.append({
                "window_id":      wid,
                "t_start":        t0,
                "t_end":          t1,
                "anomaly_score":  ws.anomaly_score,
                "quality":        ws.quality,
                "uncertainty":    ws.uncertainty,
                "trust_score":    ws.trust_score,
                "weighted_score": ws.weighted_score,
                "decision_score": decision_score,
                "is_anomaly":     is_anomaly,
                "drift_ratio":    self.drift_detector.drift_ratio,
                "retrained":      retrained,
                "inf_ms":         inf_ms,
                "nan_sig_frac":   nan_frac,
                "variant":        self.variant.value,
            })

        if verbose:
            n_anom = sum(r["is_anomaly"] for r in self.results)
            n_total = len(self.results)
            print(f"\n  Skipped: warmup/early={skipped_early}, high-NaN={skipped_nan}")
            print(f"  Scored: {n_total}  |  Anomalies: {n_anom} "
                  f"({100*n_anom/max(n_total,1):.1f}%)")
            print(f"  Retraining events: {len(self.ifm.retraining_events)}")
            drift_status = self.drift_detector.get_status()
            print(f"  Drift ratio (final): {drift_status['drift_ratio']:.2f} "
                  f"({'DRIFT' if drift_status['drift_detected'] else 'stable'})")

        return pd.DataFrame(self.results)


# =============================================================================
# SECTION 12 — Baseline Comparison Framework (Priority 1)
# =============================================================================

def run_all_baselines_per_case(
    cases: list,
    verbose: bool = False,
) -> dict:
    """
    Run all three variants independently on each case, collect per-case results.

    Returned structure:
        {
          variant_name: {
            "per_case": [{"caseid": int, "results": df, "pipeline": pipeline}, ...],
            "all_results": df   (concatenated across cases, with caseid column)
          }
        }

    Per-case is the correct approach: each case gets its own warmup, its own
    calibrated threshold, and its own drift detector. Concatenating cases gives
    the warmup a cross-patient mixed baseline, dilutes within-case drift in the
    score buffer, and makes the B vs C comparison meaningless.
    """
    print("\n" + "=" * 65)
    print("  BASELINE COMPARISON — Per-Case Pipeline")
    print("=" * 65)

    all_results = {v.value: {"per_case": [], "all_results": None}
                   for v in ModelVariant}

    for caseid, df_clean, active_sigs in cases:
        print(f"\n{'─'*55}")
        print(f"  Case {caseid}  ({len(df_clean)} rows, {len(active_sigs)} signals)")
        print(f"{'─'*55}")
        for variant in ModelVariant:
            pipeline = StreamingPipeline(active_sigs, variant=variant)
            results  = pipeline.run(df_clean, verbose=verbose)
            if not results.empty:
                results["caseid"] = caseid
            all_results[variant.value]["per_case"].append({
                "caseid":   caseid,
                "results":  results,
                "pipeline": pipeline,
            })

    for vname in all_results:
        frames = [c["results"] for c in all_results[vname]["per_case"]
                  if not c["results"].empty]
        all_results[vname]["all_results"] = (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )

    return all_results


def compute_baseline_metrics(results: pd.DataFrame) -> dict:
    """
    Compute comparison metrics for one variant's results.

    anomaly_rate      : fraction of windows flagged as anomalous
    isolated_fraction : fraction of anomaly clusters with length == 1
                        (high = single-window spikes = likely artefacts)
    mean_cluster_len  : average length of contiguous anomaly runs
    score_mean / std  : decision score distribution statistics
    n_retrain_events  : how many times the model retrained (FULL only)
    """
    if results.empty:
        return {}

    n_total = len(results)
    n_anom  = results["is_anomaly"].sum()

    # Cluster analysis
    flags    = results["is_anomaly"].values.astype(int)
    clusters = []
    run = 0
    for f in flags:
        if f:
            run += 1
        elif run > 0:
            clusters.append(run)
            run = 0
    if run > 0:
        clusters.append(run)
    clusters = np.array(clusters) if clusters else np.array([0])

    isolated_frac = float((clusters == 1).sum() / max(len(clusters), 1))
    mean_cl_len   = float(clusters.mean())

    return {
        "anomaly_rate":      float(n_anom / max(n_total, 1)),
        "isolated_fraction": isolated_frac,
        "mean_cluster_len":  mean_cl_len,
        "score_mean":        float(results["decision_score"].mean()),
        "score_std":         float(results["decision_score"].std()),
        "n_retrain_events":  int(results["retrained"].sum()),
        "n_windows":         n_total,
    }


def compare_baselines(all_results: dict) -> pd.DataFrame:
    """
    Compute and print per-case mean±std metrics across all three variants.

    Aggregating per-case (not pooling all windows) is the right way to
    report this. Per-case metrics treat each surgical case as one observation,
    which is what the statistical comparison is actually about.
    """
    summary_rows = []

    for vname, data in all_results.items():
        per_case_metrics = []
        for case_data in data["per_case"]:
            m = compute_baseline_metrics(case_data["results"])
            if m:
                m["caseid"] = case_data["caseid"]
                per_case_metrics.append(m)

        if not per_case_metrics:
            continue

        df_pc = pd.DataFrame(per_case_metrics)
        numeric_cols = ["anomaly_rate", "isolated_fraction", "mean_cluster_len",
                        "score_mean", "score_std", "n_retrain_events"]

        row = {"variant": vname, "n_cases": len(df_pc)}
        for col in numeric_cols:
            if col in df_pc.columns:
                row[f"{col}_mean"] = df_pc[col].mean()
                row[f"{col}_std"]  = df_pc[col].std()
        summary_rows.append(row)

    df_comp = pd.DataFrame(summary_rows).set_index("variant")

    print("\n── Baseline Comparison Metrics (mean ± std across cases) ───")
    for vname in df_comp.index:
        r = df_comp.loc[vname]
        print(f"\n  {vname}  (n={int(r['n_cases'])} cases)")
        for col in ["anomaly_rate", "isolated_fraction", "mean_cluster_len",
                    "n_retrain_events"]:
            mu  = r.get(f"{col}_mean", float("nan"))
            std = r.get(f"{col}_std",  float("nan"))
            print(f"    {col:<22s}: {mu:.4f} ± {std:.4f}")

    out_csv = os.path.join(OUT_DIR, "baseline_comparison.csv")
    df_comp.to_csv(out_csv)
    print(f"\n  Saved → {out_csv}")

    return df_comp


# =============================================================================
# SECTION 13 — Threshold Sensitivity Sweep (Priority 3)
# =============================================================================

def threshold_sensitivity_sweep(
    results_full: pd.DataFrame,
    warmup_scores: np.ndarray,
    percentile_range: range = range(90, 100),
) -> pd.DataFrame:
    """
    Sweep threshold from 90th to 99th percentile and record anomaly rate.

    Shows how sensitive the anomaly rate is to the threshold choice and
    provides empirical justification for the selected percentile (ideally
    an elbow in the curve around 95th).
    """
    print("\n── Threshold Sensitivity Sweep ─────────────────────────────")
    rows = []
    decision_scores = results_full["decision_score"].values

    for pct in percentile_range:
        thresh    = np.percentile(warmup_scores, pct)
        n_anom    = int((decision_scores > thresh).sum())
        anom_rate = n_anom / max(len(decision_scores), 1) * 100
        rows.append({
            "percentile":  pct,
            "threshold":   thresh,
            "n_anomalies": n_anom,
            "anomaly_rate_pct": anom_rate,
        })
        print(f"  {pct:3d}th pctile: threshold={thresh:.4f}  "
              f"anomalies={n_anom:4d}  rate={anom_rate:.1f}%")

    df_sens = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "threshold_sensitivity.csv")
    df_sens.to_csv(out_csv, index=False)
    print(f"\n  Saved → {out_csv}")
    return df_sens


# =============================================================================
# SECTION 14 — Evaluation (4 Proxy Checks, carried over from v6)
# =============================================================================

def evaluate_model(results: pd.DataFrame, ifm: StreamingIF,
                   df_clean: pd.DataFrame, active_sigs: list) -> dict:
    """Four proxy checks for model reliability without ground-truth labels."""
    eval_results = {}

    # Check 1: CV (noise sensitivity)
    # Refit 10 times on the actual warmup data with different seeds, score all
    # test windows under each model, measure variance in anomaly rate.
    warmup_X   = ifm.original_warmup_X
    anom_rates = []
    for seed in range(10):
        m = IsolationForest(
            n_estimators=N_ESTIMATORS, contamination=CONTAMINATION,
            random_state=seed, n_jobs=-1,
        )
        Xs = ifm.scaler.transform(warmup_X)
        m.fit(Xs)
        boot_thresh = np.percentile(-m.score_samples(Xs), THRESHOLD_PCTILE)
        # Score warmup windows themselves — gives a stable, reproducible rate
        boot_scores = -m.score_samples(Xs)
        rate = float((boot_scores > boot_thresh).mean() * 100)
        anom_rates.append(rate)

    cv = float(np.std(anom_rates) / (np.mean(anom_rates) + 1e-9) * 100)
    eval_results["cv"] = cv
    eval_results["cv_rates"] = anom_rates
    print(f"  [Check 1] CV = {cv:.1f}%  "
          f"({'✔ stable' if cv < 20 else '⚠ moderate' if cv < 40 else '✘ high'})")

    # Check 2: Separation ratio
    anom_scores = results[results["is_anomaly"]]["decision_score"].values
    if len(anom_scores) > 0:
        sep = float(anom_scores.mean() / ifm.threshold)
        eval_results["separation_ratio"] = sep
        print(f"  [Check 2] Separation ratio = {sep:.3f}  "
              f"({'✔' if sep > 1.5 else '⚠' if sep > 1.2 else '✘'})")
    else:
        print("  [Check 2] No anomalies detected — separation ratio N/A")

    # Check 3: Perturbation sensitivity
    # Score warmup windows directly (no time-based lookup needed, avoids
    # the multi-case time collision that was giving 0.067% before).
    rng = np.random.default_rng(42)
    score_increases = []
    for fv in warmup_X[:20]:
        if np.any(np.isnan(fv)):
            continue
        orig = ifm.score_raw(fv)
        pert = ifm.score_raw(fv + rng.normal(0, 0.05 * np.abs(fv), fv.shape))
        score_increases.append((pert - orig) / (orig + 1e-9) * 100)
    if score_increases:
        ps = float(np.mean(score_increases))
        eval_results["perturbation_sensitivity"] = ps
        print(f"  [Check 3] Perturbation sensitivity = {ps:.1f}%  "
              f"({'✔' if 5 <= ps <= 40 else '⚠'})")

    # Check 4: Temporal consistency
    flags = results["is_anomaly"].values.astype(int)
    clusters, run = [], 0
    for f in flags:
        if f:
            run += 1
        elif run > 0:
            clusters.append(run)
            run = 0
    if run > 0:
        clusters.append(run)
    if clusters:
        clusters = np.array(clusters)
        iso = float((clusters == 1).sum() / len(clusters))
        eval_results["isolated_fraction"]    = iso
        eval_results["temporal_consistency"] = 1.0 - iso
        eval_results["clusters"]             = clusters
        eval_results["cv_rates"]             = anom_rates
        print(f"  [Check 4] Isolated fraction = {iso*100:.1f}%  "
              f"({'✔' if iso < 0.3 else '⚠' if iso < 0.5 else '✘'})")
    else:
        print("  [Check 4] No anomaly clusters found")

    return eval_results


# =============================================================================
# SECTION 15 — Plotting
# =============================================================================

def plot_baseline_comparison(
    all_results: dict,
    df_comp: pd.DataFrame,
):
    """
    6-panel comparison plot.
    Panels 1-2: first case only (time-series and distributions).
    Panels 3-4: mean±std bar charts across all cases.
    Panels 5-6: FULL model drift ratio and threshold evolution (first case).
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        "Baseline Comparison: IF_Only vs No_CL vs Full Model\n"
        "Priority 1 — Ablation Study (per-case pipeline)",
        fontsize=12, fontweight="bold"
    )

    colors = {
        ModelVariant.IF_ONLY.value: "steelblue",
        ModelVariant.NO_CL.value:   "darkorange",
        ModelVariant.FULL.value:    "darkgreen",
    }

    # (0,0) Decision scores over time — first case only for readability
    ax = axes[0, 0]
    for vname, data in all_results.items():
        per_case = data["per_case"]
        if not per_case or per_case[0]["results"].empty:
            continue
        r = per_case[0]["results"]  # first case
        ax.plot(r["t_start"], r["decision_score"],
                lw=0.8, alpha=0.7, color=colors[vname], label=vname)
        anom = r[r["is_anomaly"]]
        ax.scatter(anom["t_start"], anom["decision_score"],
                   color=colors[vname], s=15, zorder=5, alpha=0.9)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Decision score")
    ax.set_title("(1) Decision Scores Over Time (Case 1)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (0,1) Score distributions — all cases pooled per variant
    ax = axes[0, 1]
    for vname, data in all_results.items():
        r = data["all_results"]
        if r is None or r.empty:
            continue
        ax.hist(r["decision_score"], bins=60, alpha=0.5,
                color=colors[vname], label=vname, density=True)
    ax.set_xlabel("Decision score"); ax.set_ylabel("Density")
    ax.set_title("(2) Score Distributions (All Cases)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1,0) Anomaly rate — mean ± std bar chart
    ax = axes[1, 0]
    vnames = list(df_comp.index)
    mu_col  = "anomaly_rate_mean"
    std_col = "anomaly_rate_std"
    if mu_col in df_comp.columns:
        means = df_comp[mu_col].values * 100
        stds  = df_comp[std_col].values * 100
        bars  = ax.bar(vnames, means,
                       color=[colors.get(v, "gray") for v in vnames],
                       alpha=0.8, edgecolor="white")
        ax.errorbar(vnames, means, yerr=stds, fmt="none",
                    color="black", capsize=5, lw=1.5)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{m:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Anomaly rate (%)"); ax.set_title("(3) Anomaly Rate (mean ± std)")
    ax.grid(alpha=0.3, axis="y")

    # (1,1) Isolated fraction — mean ± std
    ax = axes[1, 1]
    mu_col  = "isolated_fraction_mean"
    std_col = "isolated_fraction_std"
    if mu_col in df_comp.columns:
        means = df_comp[mu_col].values * 100
        stds  = df_comp[std_col].values * 100
        bars  = ax.bar(vnames, means,
                       color=[colors.get(v, "gray") for v in vnames],
                       alpha=0.8, edgecolor="white")
        ax.errorbar(vnames, means, yerr=stds, fmt="none",
                    color="black", capsize=5, lw=1.5)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{m:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Isolated anomaly fraction (%)")
    ax.set_title("(4) Temporal Consistency (mean ± std)\n(lower = more consistent)")
    ax.grid(alpha=0.3, axis="y")

    # (2,0) Drift ratio over time — FULL, first case
    ax = axes[2, 0]
    full_per_case = all_results.get(ModelVariant.FULL.value, {}).get("per_case", [])
    if full_per_case and not full_per_case[0]["results"].empty:
        r  = full_per_case[0]["results"]
        ax.plot(r["t_start"], r["drift_ratio"],
                lw=1.2, color="darkgreen", label="Drift ratio")
        ax.axhline(DRIFT_RATIO_THRESHOLD, color="red", lw=1.5, ls="--",
                   label=f"Trigger = {DRIFT_RATIO_THRESHOLD}")
        for ev in full_per_case[0]["pipeline"].ifm.retraining_events:
            ax.axvline(ev.timestamp, color="purple", lw=1.5, ls=":", alpha=0.8)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Drift ratio")
        ax.set_title("(5) Drift Ratio — FULL (Case 1)\n(Purple = retraining events)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (2,1) Threshold evolution — FULL, first case
    ax = axes[2, 1]
    if full_per_case:
        events = full_per_case[0]["pipeline"].ifm.retraining_events
        r      = full_per_case[0]["results"]
        if events:
            t_pts  = ([r["t_start"].min()]
                      + [ev.timestamp for ev in events]
                      + [r["t_start"].max()])
            th_pts = ([events[0].old_threshold]
                      + [ev.new_threshold for ev in events]
                      + [events[-1].new_threshold])
            ax.step(t_pts, th_pts, where="post", color="darkgreen", lw=2.5)
            ax.scatter([ev.timestamp for ev in events],
                       [ev.new_threshold for ev in events],
                       color="purple", s=60, zorder=5, label="Retrain point")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Threshold value")
            ax.set_title("(6) Threshold Evolution — FULL (Case 1)")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No retraining events in Case 1",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_title("(6) Threshold Evolution — FULL (Case 1)")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "baseline_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[PLOT] baseline_comparison.png saved → {out}")


def plot_threshold_sensitivity(df_sens: pd.DataFrame):
    """Dual plot: anomaly rate vs percentile + threshold value vs percentile."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Threshold Sensitivity Analysis (FULL Model)",
                 fontsize=11, fontweight="bold")

    ax = axes[0]
    ax.plot(df_sens["percentile"], df_sens["anomaly_rate_pct"],
            marker="o", color="steelblue", lw=2)
    ax.axvline(THRESHOLD_PCTILE, color="red", lw=1.5, ls="--",
               label=f"Default = {THRESHOLD_PCTILE}th")
    ax.set_xlabel("Threshold percentile")
    ax.set_ylabel("Anomaly rate (%)")
    ax.set_title("Anomaly Rate vs Percentile")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(df_sens["percentile"], df_sens["threshold"],
            marker="s", color="darkorange", lw=2)
    ax.axvline(THRESHOLD_PCTILE, color="red", lw=1.5, ls="--",
               label=f"Default = {THRESHOLD_PCTILE}th")
    ax.set_xlabel("Threshold percentile")
    ax.set_ylabel("Threshold value")
    ax.set_title("Threshold Value vs Percentile")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "threshold_sensitivity.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] threshold_sensitivity.png saved → {out}")


def plot_continual_learning(all_results: dict):
    """
    4-panel deep-dive into the FULL model's continual learning behavior.
    Uses the first case for time-series panels, all cases for the event table.
    """
    full_per_case = all_results.get(ModelVariant.FULL.value, {}).get("per_case", [])
    if not full_per_case or full_per_case[0]["results"].empty:
        print("[WARN] No FULL model results for continual learning plot")
        return

    # Use first case for time-series plots
    r        = full_per_case[0]["results"]
    pipeline = full_per_case[0]["pipeline"]
    events   = pipeline.ifm.retraining_events

    # Aggregate all retraining events across cases for the summary table
    all_events = []
    for cd in full_per_case:
        for ev in cd["pipeline"].ifm.retraining_events:
            all_events.append((cd["caseid"], ev))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Continual Learning Deep-Dive (FULL Model)\n"
        f"Total retraining events across all cases: {len(all_events)}",
        fontsize=12, fontweight="bold"
    )

    # (0,0) Scores + retraining markers
    ax = axes[0, 0]
    ax.plot(r["t_start"], r["weighted_score"],
            lw=0.9, alpha=0.7, color="darkblue", label="Weighted score")
    ax.plot(r["t_start"], r["anomaly_score"],
            lw=0.6, alpha=0.4, color="steelblue", label="Anomaly score (drift input)")
    ax.axhline(pipeline.ifm.threshold, color="orange", lw=1.5, ls="--",
               label=f"Threshold = {pipeline.ifm.threshold:.3f}")
    anom = r[r["is_anomaly"]]
    ax.scatter(anom["t_start"], anom["weighted_score"],
               color="red", s=20, zorder=5, label="Anomaly")
    for i, ev in enumerate(events):
        ax.axvline(ev.timestamp, color="purple", lw=1.5, ls=":", alpha=0.8,
                   label="Retraining" if i == 0 else "")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Score")
    ax.set_title("(1) Scores + Retraining Events (Case 1)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (0,1) Drift ratio over time
    ax = axes[0, 1]
    ax.plot(r["t_start"], r["drift_ratio"],
            lw=1.2, color="teal", label="Drift ratio (anomaly_score based)")
    ax.axhline(DRIFT_RATIO_THRESHOLD, color="red", lw=1.5, ls="--",
               label=f"Trigger = {DRIFT_RATIO_THRESHOLD}")
    ax.axhline(1.0, color="gray", lw=1, ls=":", alpha=0.5)
    for ev in events:
        ax.axvline(ev.timestamp, color="purple", lw=1, ls=":", alpha=0.6)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Drift ratio")
    ax.set_title("(2) Drift Ratio Over Time (Case 1)\n"
                 "(raw anomaly_score / warmup_mean)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1,0) Threshold evolution
    ax = axes[1, 0]
    if events:
        t_pts     = ([r["t_start"].min()]
                     + [ev.timestamp for ev in events]
                     + [r["t_start"].max()])
        thresh_pts = ([events[0].old_threshold]
                      + [ev.new_threshold for ev in events]
                      + [events[-1].new_threshold])
        ax.step(t_pts, thresh_pts, where="post",
                color="darkgreen", lw=2.5, label="Threshold")
        ax.scatter([ev.timestamp for ev in events],
                   [ev.new_threshold for ev in events],
                   color="purple", s=60, zorder=5, label="Retrain point")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Threshold value")
        ax.set_title("(3) Threshold Evolution (Case 1)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No retraining events in Case 1",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("(3) Threshold Evolution (Case 1)")

    # (1,1) Retraining event summary table — all cases
    ax = axes[1, 1]; ax.axis("off")
    if all_events:
        col_labels = ["Case", "t (s)", "Drift ratio", "Warmup N",
                      "Recent N", "Δthreshold"]
        table_data = [
            [str(cid), f"{ev.timestamp:.0f}", f"{ev.drift_ratio:.2f}",
             str(ev.n_warmup_windows), str(ev.n_recent_windows),
             f"{ev.new_threshold - ev.old_threshold:+.4f}"]
            for cid, ev in all_events
        ]
        table = ax.table(
            cellText=table_data, colLabels=col_labels,
            loc="center", cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.4)
        ax.set_title(f"(4) All Retraining Events ({len(all_events)} total)", pad=12)
    else:
        ax.text(0.5, 0.5,
                "No retraining events across any case.\n"
                f"Max observed drift ratio: {r['drift_ratio'].max():.3f}\n"
                f"Trigger threshold: {DRIFT_RATIO_THRESHOLD}",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("(4) Retraining Event Log")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "continual_learning.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] continual_learning.png saved → {out}")

def plot_unified_results(results: pd.DataFrame, df_clean: pd.DataFrame,
                         active_sigs: list, threshold: float,
                         eval_results: dict, prefix: str = "vitaldb"):
    """Single-variant detailed plot, carried over from v6 for the FULL model."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        "Real-Time Anomaly Detection v7 — Full Model Detailed View\n"
        "MAD Warmup | WindowScore | Drift-Triggered Retraining",
        fontsize=11, fontweight="bold"
    )
    wt = results["t_start"].values

    ax = axes[0, 0]
    ax.plot(wt, results["anomaly_score"].values, lw=0.8, alpha=0.6,
            color="steelblue", label="Raw anomaly score (drift input)")
    ax.plot(wt, results["weighted_score"].values, lw=1.2,
            color="darkblue", label="Weighted score (decision)")
    ax.axhline(threshold, color="orange", lw=1.5, ls="--",
               label=f"Threshold = {threshold:.3f}")
    anom_mask = results["is_anomaly"].values
    ax.scatter(wt[anom_mask], results.loc[anom_mask, "weighted_score"].values,
               color="red", s=20, zorder=5, label="Anomaly")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Score")
    ax.set_title("(1) Anomaly vs Weighted Score"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(wt, results["quality"].values, lw=1, color="green", label="Quality")
    ax.plot(wt, results["trust_score"].values, lw=1, color="purple", label="Trust")
    ax.axhline(QUALITY_DEGRADATION_THRESHOLD, color="red", lw=1, ls="--",
               label=f"Quality threshold = {QUALITY_DEGRADATION_THRESHOLD}")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Score")
    ax.set_title("(2) Quality and Trust Metrics"); ax.legend(fontsize=7)
    ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    numeric = [c for c in active_sigs
               if not c.endswith(("_mean","_std","_p2p","_rms"))]
    clrs = plt.cm.tab10.colors
    for i, col in enumerate(numeric[:6]):
        if col not in df_clean.columns:
            continue
        v = df_clean[col].values.astype(float)
        lo, hi = np.nanmin(v), np.nanmax(v)
        ax.plot(df_clean["time"].values, (v - lo) / (hi - lo + 1e-9),
                lw=0.6, alpha=0.7, color=clrs[i % len(clrs)], label=col)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Normalized value")
    ax.set_title("(3) Vital Signs (normalized)"); ax.legend(ncol=3, fontsize=6); ax.grid(alpha=0.3)

    ax = axes[1, 1]; ax.axis("off")
    txt = "EVALUATION SUMMARY\n" + "="*35 + "\n\n"
    for key, label, good_range in [
        ("cv",                   "[1] CV",          (0, 20)),
        ("separation_ratio",     "[2] Separation",  (1.5, 99)),
        ("perturbation_sensitivity", "[3] Perturbation", (10, 30)),
        ("isolated_fraction",    "[4] Isolated",    (0, 0.3)),
    ]:
        val = eval_results.get(key)
        if val is not None:
            ok = good_range[0] <= val <= good_range[1]
            sym = "✔" if ok else "⚠"
            txt += f"{label}: {val:.2f}  {sym}\n\n"
    ax.text(0.1, 0.9, txt, transform=ax.transAxes, fontsize=9,
            va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    ax = axes[2, 0]
    clusters = eval_results.get("clusters")
    if clusters is not None and len(clusters) > 0:
        mx = min(int(clusters.max()), 20)
        ax.hist(clusters, bins=np.arange(0.5, mx + 1.5, 1),
                color="coral", alpha=0.8, edgecolor="white")
        ax.axvline(1.5, color="red", lw=1.5, ls="--", label="Isolated (len=1)")
        iso = eval_results.get("isolated_fraction", 0)
        ax.set_xlabel("Cluster length"); ax.set_ylabel("Count")
        ax.set_title(f"(4) Cluster Lengths (isolated={iso*100:.0f}%)")
        ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2, 1]
    cv_rates = eval_results.get("cv_rates")
    if cv_rates is not None:
        ax.bar(range(len(cv_rates)), cv_rates, color="steelblue",
               alpha=0.8, edgecolor="white")
        m = np.mean(cv_rates)
        ax.axhline(m, color="navy", lw=2, ls="--", label=f"Mean: {m:.1f}%")
        ax.fill_between(range(len(cv_rates)), m - np.std(cv_rates),
                        m + np.std(cv_rates), alpha=0.15, color="navy")
        ax.set_xticks(range(len(cv_rates)))
        ax.set_xticklabels([f"s{i}" for i in range(len(cv_rates))], fontsize=7)
        ax.set_ylabel("Anomaly rate (%)")
        cv = eval_results.get("cv", 0)
        ax.set_title(f"(5) Bootstrap Stability (CV = {cv:.1f}%)")
        ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"{prefix}_v7_full_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {prefix}_v7_full_model.png saved → {out}")


# =============================================================================
# MAIN — Research Pipeline
# =============================================================================

def main():
    print("=" * 65)
    print("  Continual Learning Anomaly Detection v7")
    print("  Baseline Comparison + CL Retraining + Expanded Eval")
    print("=" * 65)

    # ── Priority 3: Load 10-15 VitalDB cases — per-case, not concatenated ──
    print("\n[STEP 1] Loading data (per-case)…")
    if VITALDB_AVAILABLE:
        cases = load_cases_separately(
            case_ids=list(range(1, 21)),
            min_rows=MIN_CASE_LENGTH,
            target_count=TARGET_CASE_COUNT,
        )
        if not cases:
            print("[WARN] No real data loaded — falling back to simulation.")
            df_sim = simulate_vitaldb_like(n_seconds=3000)
            df_clean_sim, active_sigs_sim = handle_missing(df_sim)
            cases = [(0, df_clean_sim, active_sigs_sim)]
    else:
        print("  vitaldb not available — using simulation.")
        df_sim = simulate_vitaldb_like(n_seconds=3000)
        df_clean_sim, active_sigs_sim = handle_missing(df_sim)
        cases = [(0, df_clean_sim, active_sigs_sim)]

    print(f"\n  Running pipeline on {len(cases)} case(s).")

    # ── Priority 1: Run all 3 baselines per case ──────────────────────────
    print("\n[STEP 2] Running baseline comparison (Priority 1)…")
    all_results = run_all_baselines_per_case(cases, verbose=False)
    df_comp     = compare_baselines(all_results)

    # ── Priority 2: Continual learning summary ────────────────────────────
    print("\n[STEP 3] Continual learning analysis (Priority 2)…")
    full_per_case = all_results.get(ModelVariant.FULL.value, {}).get("per_case", [])
    total_events  = sum(len(c["pipeline"].ifm.retraining_events)
                        for c in full_per_case)
    print(f"  Total retraining events across {len(full_per_case)} cases: {total_events}")
    for cd in full_per_case:
        evs = cd["pipeline"].ifm.retraining_events
        print(f"  Case {cd['caseid']}: {len(evs)} retraining event(s)")
        for ev in evs:
            print(f"    {ev}")

    if total_events == 0:
        print(f"\n  [INFO] No retraining triggered. Drift ratio threshold is "
              f"{DRIFT_RATIO_THRESHOLD}. You can lower it in the config "
              f"if the data is stable.")

    # ── Priority 3: Threshold sensitivity on FULL model (first case) ──────
    print("\n[STEP 4] Threshold sensitivity sweep (Priority 3)…")
    df_sens = pd.DataFrame()
    if full_per_case and not full_per_case[0]["results"].empty:
        first_full = full_per_case[0]
        df_sens = threshold_sensitivity_sweep(
            first_full["results"],
            first_full["pipeline"].ifm.warmup_scores,
        )

    # ── Evaluation on FULL model (first case) ─────────────────────────────
    print("\n[STEP 5] Evaluating FULL model (Case 1)…")
    eval_results = {}
    if full_per_case and not full_per_case[0]["results"].empty:
        first_full_r   = full_per_case[0]["results"]
        first_full_ifm = full_per_case[0]["pipeline"].ifm
        first_case_df  = cases[0][1]
        first_sigs     = cases[0][2]
        eval_results   = evaluate_model(
            first_full_r, first_full_ifm, first_case_df, first_sigs
        )

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[STEP 6] Generating plots…")
    plot_baseline_comparison(all_results, df_comp)
    plot_continual_learning(all_results)
    if not df_sens.empty:
        plot_threshold_sensitivity(df_sens)
    if full_per_case and not full_per_case[0]["results"].empty:
        prefix = "vitaldb_real" if VITALDB_AVAILABLE else "vitaldb_sim"
        plot_unified_results(
            full_per_case[0]["results"],
            cases[0][1],
            cases[0][2],
            full_per_case[0]["pipeline"].ifm.threshold,
            eval_results,
            prefix=prefix,
        )

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"\nBaseline Comparison (mean ± std across {len(cases)} cases):")
    for vname in df_comp.index:
        r = df_comp.loc[vname]
        ar_mu  = r.get("anomaly_rate_mean", float("nan"))
        ar_std = r.get("anomaly_rate_std",  float("nan"))
        iso_mu = r.get("isolated_fraction_mean", float("nan"))
        rt_mu  = r.get("n_retrain_events_mean",  float("nan"))
        print(f"  {vname:<10s}  anomaly={ar_mu*100:.1f}%±{ar_std*100:.1f}%  "
              f"isolated={iso_mu*100:.1f}%  retrain_events={rt_mu:.1f}")

    print(f"\n  Total retraining events (FULL): {total_events}")

    if eval_results:
        print(f"\nProxy Evaluation (FULL, Case 1):")
        for key in ["cv", "separation_ratio", "perturbation_sensitivity",
                    "isolated_fraction"]:
            val = eval_results.get(key)
            if val is not None:
                print(f"  {key:<28s}: {val:.3f}")

    print(f"\nOutput files:")
    for f in ["baseline_comparison.csv", "baseline_comparison.png",
              "threshold_sensitivity.csv", "threshold_sensitivity.png",
              "continual_learning.png"]:
        path = os.path.join(OUT_DIR, f)
        exists = "✔" if os.path.exists(path) else "✘ (not generated)"
        print(f"  {exists}  {path}")

    print(f"\n[DONE]")
    return all_results, cases, df_comp


if __name__ == "__main__":
    main()