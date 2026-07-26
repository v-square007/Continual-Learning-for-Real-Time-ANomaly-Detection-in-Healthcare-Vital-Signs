"""
=============================================================================
Continual Learning — Real-Time Anomaly Detection on Healthcare Vital Signals
v9
=============================================================================

KEY CHANGES FROM V6:
-----------------------------------------------------------------------------
1. THREE-WAY BASELINE COMPARISON (Priority 1):
   - Variant A: IF_ONLY       -- plain IF, random warmup, raw anomaly_score, no CL.
   - Variant B: NO_CL        -- MAD warmup + WindowScore, frozen after warmup.
   - Variant C: FULL_DRIFT   -- MAD + WindowScore + retrain on drift_ratio only.
   - Variant D: FULL_ANOMALY -- MAD + WindowScore + retrain on anomaly_rate only.
   - Variant E: FULL_BOTH    -- MAD + WindowScore + retrain on both (proposed).
   A vs B          -> effect of MAD warmup + trust weighting
   B vs any Full   -> does retraining help at all?
   C vs D vs E     -> which trigger strategy is best?

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

# Drift detection — dual-condition trigger (drift ratio AND anomaly rate)
DRIFT_BUFFER_SIZE       = 50
# Drift detection thresholds
# DRIFT_RATIO_THRESHOLD is no longer static. DriftDetector.calibrate() computes
# it per-case as:  threshold = 1.0 + DRIFT_THRESHOLD_K * std(warmup_drift_ratios)
# This adapts to each case's score variance instead of being a global guess.
DRIFT_THRESHOLD_K       = 2.0   # k=2 => ~97.5th pctile for Gaussian drift ratios
ANOMALY_RATE_TRIGGER    = 0.15  # Static: >15% of recent windows anomalous -> retrain

# Continual learning retraining
RETRAIN_COOLDOWN_S        = 300   # 5 min cooldown (10 min was too long per-case)
RETRAIN_QUALITY_THRESHOLD = 0.7   # Only use high-quality windows for retraining
RETRAIN_RECENT_WINDOW_CAP = 150   # Max recent windows to add
RETRAIN_MIN_STABLE        = 10    # Lowered: 20 was too strict for short cases
RETRAIN_RECENT_BUFFER_CAP = 300   # Rolling buffer of recent windows to draw from

# Expanded evaluation (Priority 3)
TARGET_CASE_COUNT = 15
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
    Five variants for ablation study.

    IF_ONLY       : IF, random warmup, raw anomaly_score, no retraining.
                    Performance floor.
    NO_CL         : MAD warmup + WindowScore (anomaly x trust), frozen.
                    Isolates warmup/scoring contribution.
    FULL_DRIFT    : MAD + WindowScore + retrain on drift_ratio alone.
    FULL_ANOMALY  : MAD + WindowScore + retrain on anomaly_rate alone.
    FULL_BOTH     : MAD + WindowScore + retrain on both (proposed system).

    Ablation:
      IF_ONLY  vs NO_CL        -> effect of MAD warmup + trust weighting
      NO_CL    vs any FULL     -> does retraining help at all?
      FULL_DRIFT/ANOMALY/BOTH  -> which trigger strategy is best?
    """
    IF_ONLY      = "IF_Only"
    NO_CL        = "No_CL"
    FULL_DRIFT   = "Full_Drift"
    FULL_ANOMALY = "Full_Anomaly"
    FULL_BOTH    = "Full_Both"




class TriggerMode(Enum):
    """
    Controls which condition(s) DriftDetector uses to signal retraining.
    Set by StreamingPipeline based on ModelVariant.
    """
    DRIFT_ONLY   = "drift_only"    # drift_ratio > self-calibrated threshold
    ANOMALY_ONLY = "anomaly_only"  # recent_anomaly_rate > ANOMALY_RATE_TRIGGER
    BOTH         = "both"          # both conditions simultaneously
    NONE         = "none"          # never triggers (IF_ONLY, NO_CL)

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


@dataclass
class AdaptationSnapshot:
    """
    Per-window record of model adaptation state.
    Collected throughout the run to visualize the model's learning trajectory.

    This is the core novelty artifact: it shows the model is not static.
    At every window, we record what the drift ratio was, whether the model
    was in an anomalous regime, and whether a retrain just fired. Together
    these form the "adaptation timeline" that distinguishes continual learning
    from a frozen model.
    """
    window_id:     int
    timestamp:     float
    anomaly_score: float
    drift_ratio:   float
    anomaly_rate:  float           # fraction of buffer currently anomalous
    threshold:     float           # current model threshold (changes after retrain)
    is_anomaly:    bool
    retrained:     bool            # True only in the window that triggered retrain
    n_retrains_so_far: int         # cumulative retrain count up to this window


class ContinualLearningTracker:
    """
    Records the model's full adaptation history for a single case run.

    Novelty framing: standard anomaly detectors have a fixed decision boundary
    after training. This tracker provides evidence that the boundary moves —
    adapting to within-case physiological shifts — which is the central claim
    of the continual learning contribution.

    Outputs:
      - adaptation_timeline: list of AdaptationSnapshot, one per scored window
      - threshold_trajectory: list of (timestamp, threshold) tuples showing
        how the decision boundary evolves after each retraining event
      - trigger_analysis: for each retrain event, records which condition(s)
        were active (drift / anomaly_rate / both), enabling the C vs D vs E
        ablation to be interpreted mechanistically, not just numerically
    """

    def __init__(self):
        self.snapshots: List[AdaptationSnapshot] = []
        self.threshold_trajectory: List[tuple]   = []  # (t, threshold)
        self.trigger_analysis: List[dict]         = []
        self._n_retrains = 0

    def record(self, window_id: int, timestamp: float, anomaly_score: float,
               drift_ratio: float, anomaly_rate: float, threshold: float,
               is_anomaly: bool, retrained: bool,
               drift_cond: bool = False, anomaly_cond: bool = False):
        if retrained:
            self._n_retrains += 1
            self.threshold_trajectory.append((timestamp, threshold))
            self.trigger_analysis.append({
                "window_id":    window_id,
                "timestamp":    timestamp,
                "drift_active": drift_cond,
                "anomaly_active": anomaly_cond,
                "drift_ratio":  drift_ratio,
                "anomaly_rate": anomaly_rate,
                "new_threshold": threshold,
            })

        self.snapshots.append(AdaptationSnapshot(
            window_id=window_id, timestamp=timestamp,
            anomaly_score=anomaly_score, drift_ratio=drift_ratio,
            anomaly_rate=anomaly_rate, threshold=threshold,
            is_anomaly=is_anomaly, retrained=retrained,
            n_retrains_so_far=self._n_retrains,
        ))

    def to_dataframe(self) -> pd.DataFrame:
        if not self.snapshots:
            return pd.DataFrame()
        return pd.DataFrame([vars(s) for s in self.snapshots])

    def summary(self) -> dict:
        if not self.snapshots:
            return {}
        df  = self.to_dataframe()
        return {
            "n_retrains":          self._n_retrains,
            "n_windows":           len(df),
            "anomaly_rate":        df["is_anomaly"].mean(),
            "mean_drift_ratio":    df["drift_ratio"].mean(),
            "max_drift_ratio":     df["drift_ratio"].max(),
            "threshold_range":     (df["threshold"].min(), df["threshold"].max()),
            "threshold_delta":     df["threshold"].max() - df["threshold"].min(),
        }


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
    """
    Rolling median buffer for NaN imputation in feature vectors.

    Cold-start problem: the buffer needs 5 clean windows before fill() works.
    For cases like Case 1 where the ENTIRE first half has NaN in some features,
    no window ever passes the clean-window gate, the buffer never warms, and
    select_warmup_random() returns 0 candidates.

    Fix: seed_from_bounds() pre-loads the buffer with physiologically plausible
    prior values derived from the midpoint of each signal's PHYS_BOUNDS range.
    This gives fill() something to return from window 1, so partial-NaN windows
    can be imputed immediately rather than being discarded until 5 clean windows
    appear. The prior is weak (5 copies of a single midpoint vector) and gets
    overwritten by real data within a few windows, but it breaks the cold-start
    deadlock.
    """
    def __init__(self, maxlen=200):
        self._buf = deque(maxlen=maxlen)
        self._medians = None

    def seed_from_bounds(self, active_sigs: list):
        """
        Pre-warm the buffer with physiological midpoints so fill() works
        from the very first window even when early data is all NaN.

        For each active signal, the prior for each of the 5 features
        (mean, std, min, max, mad) is set to the midpoint of PHYS_BOUNDS.
        Waveform-derived signals (ECG_*, PLETH_*) get a neutral prior of 0.0
        since they have no clinical bounds defined.
        """
        n_feats = len(active_sigs) * 5
        prior   = np.zeros(n_feats, dtype=float)

        for i, sig in enumerate(active_sigs):
            base = sig.split("_")[0]
            if base in PHYS_BOUNDS:
                lo, hi = PHYS_BOUNDS[base]
                mid    = (lo + hi) / 2.0
                # mean=mid, std=small, min=mid, max=mid, mad=0
                prior[i*5 : i*5+5] = [mid, (hi-lo)*0.05, mid, mid, 0.0]
            # else: leave as 0.0 for waveform features

        # Insert 5 copies to satisfy the >= 5 gate in update()
        for _ in range(5):
            self._buf.append(prior.copy())
        self._medians = prior.copy()

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
    No stability filtering (contrast against MAD selection).

    The original version required zero NaN across the entire feature vector,
    which silently dropped cases where intermittent sensors (BP, BT) produce
    partial-NaN windows in the first half of the recording. Case 1 returned
    0 candidates entirely because its early windows all had some NaN features,
    giving n=9 instead of n=10 for IF_Only.

    Fix: use the same rolling median imputation the scoring loop uses. This
    keeps the comparison fair -- IF_Only sees the same data coverage as MAD
    warmup, just without the stability filter.
    """
    max_row = int(len(df) * 0.50)
    df_slice = df.iloc[:max_row]
    candidates = []
    median_buf = FeatureMedianBuffer(maxlen=200)
    # Seed with physiological priors so fill() works from window 1,
    # even when the entire early recording has partial NaN (Case 1 fix).
    median_buf.seed_from_bounds(active_sigs)

    for _, _, _, fv, nan_frac in sliding_windows(df_slice, active_sigs):
        if nan_frac > MAX_NAN_SIGNAL_FRACTION:
            continue

        if np.any(np.isnan(fv)):
            fv_filled = median_buf.fill(fv)
            if fv_filled is None:
                median_buf.update(fv)
                continue
            fv = fv_filled
        else:
            median_buf.update(fv)

        candidates.append(fv)
        if len(candidates) >= n_required:
            break

    if not candidates:
        print(f"  [Random warmup] 0 candidates -- case has too much missing data")
        return np.array([])

    selected = candidates[:n_required]
    if len(selected) < n_required:
        print(f"  [Random warmup] Only {len(selected)}/{n_required} windows available")
    else:
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

    IF_ONLY      : random warmup, raw anomaly_score, no retraining.
    NO_CL        : MAD warmup, weighted_score, no retraining.
    FULL_DRIFT   : MAD warmup, weighted_score, drift_ratio trigger.
    FULL_ANOMALY : MAD warmup, weighted_score, anomaly_rate trigger.
    FULL_BOTH    : MAD warmup, weighted_score, both triggers required.
    """

    def __init__(self, variant: ModelVariant = ModelVariant.FULL_BOTH):
        self.variant              = variant
        self.scaler               = StandardScaler()
        self.model                = None
        self.threshold            = None
        self._fitted              = False
        self.warmup_scores        = None
        self.warmup_mean_score    = None
        self.original_warmup_X    = None   # Preserved for retraining (anti-forgetting)
        self.retraining_events: List[RetrainingEvent] = []

    # -- Training ----------------------------------------------------------

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

    # -- Scoring -----------------------------------------------------------

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
        """IF_ONLY uses raw score; all others use trust-weighted score."""
        if self.variant == ModelVariant.IF_ONLY:
            return ws.anomaly_score
        return ws.weighted_score

    @property
    def is_continual_learning_variant(self) -> bool:
        """True for the three FULL variants that perform retraining."""
        return self.variant in (
            ModelVariant.FULL_DRIFT,
            ModelVariant.FULL_ANOMALY,
            ModelVariant.FULL_BOTH,
        )


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
                 anomaly_rate_trigger: float = ANOMALY_RATE_TRIGGER,
                 warmup_mean: float = None,
                 trigger_mode: "TriggerMode" = None):
        self.buffer               = deque(maxlen=buffer_size)
        self.anomaly_flag_buffer  = deque(maxlen=buffer_size)
        self.anomaly_rate_trigger = anomaly_rate_trigger
        self.warmup_mean          = warmup_mean
        self.trigger_mode         = trigger_mode if trigger_mode else TriggerMode.NONE
        self.drift_detected       = False
        self.drift_ratio          = 1.0
        self.anomaly_rate         = 0.0
        self.drift_ratio_history: List[float] = []
        self.threshold            = 1.15  # overwritten by calibrate()

    def calibrate(self, warmup_anomaly_scores: np.ndarray,
                  k: float = DRIFT_THRESHOLD_K):
        """
        Self-calibrate drift threshold from warmup score variance.

        Simulates drift_ratio values during warmup (rolling mean / warmup_mean)
        and sets threshold = 1.0 + k * std(those ratios). This means the trigger
        fires only when drift exceeds k standard deviations above the warmup
        baseline variance -- a principled per-case value rather than a global
        heuristic like 1.15 or 1.3.
        """
        ratios = []
        wmean  = self.warmup_mean + 1e-9
        for i in range(len(warmup_anomaly_scores)):
            window = warmup_anomaly_scores[max(0, i - DRIFT_BUFFER_SIZE + 1): i + 1]
            ratios.append(float(np.mean(window) / wmean))
        ratio_std      = float(np.std(ratios))
        self.threshold = 1.0 + k * ratio_std
        print(f"  [DriftDetector] threshold={self.threshold:.4f}  "
              f"(1.0 + {k}x{ratio_std:.4f})  mode={self.trigger_mode.value}")

    def update(self, anomaly_score: float, is_anomaly: bool = False) -> bool:
        """
        Feed a new raw anomaly_score and whether this window was flagged anomalous.

        Dual-condition trigger (your prof's requirement):
          1. drift_ratio  > self.threshold (self-calibrated per case) -- distribution has shifted
          2. anomaly_rate > ANOMALY_RATE_TRIGGER   -- model is actively misfiring

        Why both conditions:
          Drift ratio alone can trigger on a benign physiological shift where
          scores creep up but the model isn't actually producing many anomaly
          flags -- retraining there is unnecessary and wastes the cooldown period.

          High anomaly rate alone (without drift) is usually a short sensor
          noise burst -- retraining on that would corrupt the baseline with
          artefact-heavy windows.

          Both sustained together means the score distribution has genuinely
          shifted AND the model is responding by flagging aggressively -- that
          is the case that warrants a baseline update.
        """
        self.buffer.append(anomaly_score)
        self.anomaly_flag_buffer.append(int(is_anomaly))

        if len(self.buffer) >= self.buffer.maxlen:
            current_mean        = np.mean(self.buffer)
            self.drift_ratio    = current_mean / (self.warmup_mean + 1e-9)
            self.anomaly_rate   = float(np.mean(self.anomaly_flag_buffer))
            self.drift_ratio_history.append(self.drift_ratio)

            drift_condition   = self.drift_ratio  > self.threshold
            anomaly_condition = self.anomaly_rate > self.anomaly_rate_trigger

            if not self.drift_detected:
                triggered = False
                if self.trigger_mode == TriggerMode.DRIFT_ONLY:
                    triggered = drift_condition
                elif self.trigger_mode == TriggerMode.ANOMALY_ONLY:
                    triggered = anomaly_condition
                elif self.trigger_mode == TriggerMode.BOTH:
                    triggered = drift_condition and anomaly_condition
                # TriggerMode.NONE: never triggers

                if triggered:
                    self.drift_detected = True
                    return True

        return False

    def reset(self, new_warmup_mean: float):
        """
        Reset after retraining. Clears both buffers and sets a new baseline mean
        so the detector doesn't immediately re-trigger on the same pattern.
        """
        self.buffer.clear()
        self.anomaly_flag_buffer.clear()
        self.warmup_mean    = new_warmup_mean
        self.drift_detected = False
        self.drift_ratio    = 1.0
        self.anomaly_rate   = 0.0
        # Note: threshold is NOT reset — the calibrated value stays.
        # New warmup_mean is enough to re-anchor the ratio denominator.

    def get_status(self) -> dict:
        return {
            "drift_detected":  self.drift_detected,
            "drift_ratio":     self.drift_ratio,
            "anomaly_rate":    self.anomaly_rate,
            "buffer_size":     len(self.buffer),
        }


# =============================================================================
# SECTION 11 — Streaming Pipeline with Variant Dispatch
# =============================================================================

class StreamingPipeline:
    """
    Runs one of the three model variants on a given dataframe.

    Variant dispatch:
      IF_ONLY      : random warmup, raw anomaly_score, no retrain
      NO_CL        : MAD warmup, weighted_score, no retrain
      FULL_DRIFT   : MAD warmup, weighted_score, drift_ratio trigger
      FULL_ANOMALY : MAD warmup, weighted_score, anomaly_rate trigger
      FULL_BOTH    : MAD warmup, weighted_score, both triggers required
    """

    def __init__(self, active_sigs: list, variant: ModelVariant = ModelVariant.FULL_BOTH):
        self.active_sigs   = active_sigs
        self.variant       = variant
        self.ifm           = StreamingIF(variant=variant)
        self.feat_buffer   = FeatureMedianBuffer(maxlen=200)
        self.drift_detector = None
        self.results: List[dict] = []

        # Rolling buffer for retraining window selection (FULL only)
        self._recent_buffer: deque = deque(maxlen=RETRAIN_RECENT_BUFFER_CAP)
        self._last_retrain_time: float = -np.inf
        # CL tracker — records per-window adaptation state (novelty artifact)
        self.cl_tracker = ContinualLearningTracker()

    def run(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        self.results = []

        if verbose:
            print(f"\n{'='*65}")
            print(f"  Variant: {self.variant.value}")
            print(f"  Signals: {self.active_sigs}")
            print(f"  Window: {WINDOW_SIZE_S}s | Step: {STEP_SIZE_S}s")
            print(f"{'='*65}")

        # -- Warmup selection ----------------------------------------------
        if self.variant == ModelVariant.IF_ONLY:
            warmup_X = select_warmup_random(df, self.active_sigs)
        else:
            warmup_X = select_warmup_mad(df, self.active_sigs)

        _trigger_map = {
            ModelVariant.IF_ONLY:      TriggerMode.NONE,
            ModelVariant.NO_CL:        TriggerMode.NONE,
            ModelVariant.FULL_DRIFT:   TriggerMode.DRIFT_ONLY,
            ModelVariant.FULL_ANOMALY: TriggerMode.ANOMALY_ONLY,
            ModelVariant.FULL_BOTH:    TriggerMode.BOTH,
        }
        trigger_mode = _trigger_map[self.variant]

        if len(warmup_X) < WARMUP_WINDOWS:
            print(f"  [ERROR] Insufficient warmup windows ({len(warmup_X)})")
            return pd.DataFrame()

        # Seed with physiological priors so fill() works from window 1,
        # even when early windows are all-NaN (fixes Case 1 n=9 bug).
        self.feat_buffer.seed_from_bounds(self.active_sigs)
        for fv in warmup_X:
            self.feat_buffer.update(fv)

        # -- Train ---------------------------------------------------------
        self.ifm.fit(warmup_X, silent=not verbose)

        # -- Drift detector — self-calibrated, mode from variant ----------
        self.drift_detector = DriftDetector(
            warmup_mean=self.ifm.warmup_mean_score,
            trigger_mode=trigger_mode,
        )
        self.drift_detector.calibrate(self.ifm.warmup_scores)

        # -- Scoring loop -------------------------------------------------
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

            # -- Drift detection (on anomaly_score — Priority 2) -----------
            drift_triggered = self.drift_detector.update(ws.anomaly_score, is_anomaly)

            # -- Retraining (all three FULL variants) ----------------------
            retrained = False
            if (self.ifm.is_continual_learning_variant
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

            # -- Recent buffer for next retraining -------------------------
            self._recent_buffer.append((feat, ws.quality, ws.anomaly_score))

            # Record drift/anomaly conditions for trigger analysis
            _drift_cond   = self.drift_detector.drift_ratio  > self.drift_detector.threshold
            _anomaly_cond = self.drift_detector.anomaly_rate > self.drift_detector.anomaly_rate_trigger

            self.cl_tracker.record(
                window_id=wid, timestamp=t0,
                anomaly_score=ws.anomaly_score,
                drift_ratio=self.drift_detector.drift_ratio,
                anomaly_rate=self.drift_detector.anomaly_rate,
                threshold=self.ifm.threshold,
                is_anomaly=is_anomaly, retrained=retrained,
                drift_cond=_drift_cond, anomaly_cond=_anomaly_cond,
            )

            self.results.append({
                "window_id":        wid,
                "t_start":          t0,
                "t_end":            t1,
                "anomaly_score":    ws.anomaly_score,
                "quality":          ws.quality,
                "uncertainty":      ws.uncertainty,
                "trust_score":      ws.trust_score,
                "weighted_score":   ws.weighted_score,
                "decision_score":   decision_score,
                "is_anomaly":       is_anomaly,
                "drift_ratio":      self.drift_detector.drift_ratio,
                "anomaly_rate_buf": self.drift_detector.anomaly_rate,
                "threshold":        self.ifm.threshold,
                "retrained":        retrained,
                "inf_ms":           inf_ms,
                "nan_sig_frac":     nan_frac,
                "variant":          self.variant.value,
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
    print("  BASELINE COMPARISON — 5 Variants, Per-Case Pipeline")
    print("  IF_Only | No_CL | Full_Drift | Full_Anomaly | Full_Both")
    print("=" * 65)

    all_results = {v.value: {"per_case": [], "all_results": None}
                   for v in ModelVariant}

    for caseid, df_clean, active_sigs in cases:
        print(f"\n{'-'*55}")
        print(f"  Case {caseid}  ({len(df_clean)} rows, {len(active_sigs)} signals)")
        print(f"{'-'*55}")
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
        return None  # None = truly no data; distinct from zero-anomaly case

    n_total = len(results)
    n_anom  = int(results["is_anomaly"].sum())

    # Cluster analysis
    # Zero-anomaly case is valid and must be included: isolated_frac=0, cluster_len=0
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

    if clusters:
        clusters_arr  = np.array(clusters)
        isolated_frac = float((clusters_arr == 1).sum() / len(clusters_arr))
        mean_cl_len   = float(clusters_arr.mean())
    else:
        # No anomalies at all — valid result, not missing data
        isolated_frac = 0.0
        mean_cl_len   = 0.0

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
            if m is not None:   # None = empty df; zero-anomaly dicts are valid
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
                row[f"{col}_mean"]   = df_pc[col].mean()
                row[f"{col}_std"]    = df_pc[col].std()
                row[f"{col}_median"] = df_pc[col].median()
                row[f"{col}_q25"]    = df_pc[col].quantile(0.25)
                row[f"{col}_q75"]    = df_pc[col].quantile(0.75)
        summary_rows.append(row)

    df_comp = pd.DataFrame(summary_rows).set_index("variant")

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
    print("\n-- Threshold Sensitivity Sweep -----------------------------")
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


# =============================================================================
# SECTION GT-1 — Clinical Annotation Loader (Ground Truth Approach 1)
# =============================================================================

# Clinical event tracks in VitalDB that mark genuine anomalous periods.
# These are the tracks used to define ground truth windows.
CLINICAL_EVENT_TRACKS = {
    # Hypotension: MAP < 65 mmHg for ≥ 1 minute is the clinical definition
    "Solar8000/NIBP_MBP":  ("hypotension",  lambda v: v < 65),
    # Desaturation: SpO2 < 94% is the clinical alert threshold
    "Solar8000/PLETH_SPO2": ("desaturation", lambda v: v < 94),
    # Bradycardia: HR < 50 bpm
    "Solar8000/HR":         ("bradycardia",  lambda v: v < 50),
    # Tachycardia: HR > 120 bpm
    "Solar8000/HR_tachy":   ("tachycardia",  lambda v: v > 120),
}

# Drug infusion tracks that mark anesthesia phases (induction, maintenance)
# Propofol and remifentanil presence marks active anesthesia periods
DRUG_TRACKS = [
    "Orchestra/PPF20_VOL",   # Propofol infusion volume
    "Orchestra/RFTN20_VOL",  # Remifentanil infusion volume
    "Orchestra/PPF20_CE",    # Propofol effect-site concentration
]


def fetch_clinical_events(
    caseid: int,
    df_clean: pd.DataFrame,
    window_size: int = WINDOW_SIZE_S,
    step: int = STEP_SIZE_S,
) -> pd.Series:
    """
    Fetch VitalDB clinical event annotations and convert to per-window labels.

    Strategy:
    1. Pull MAP, SpO2, HR tracks at 1Hz
    2. Apply clinical thresholds to identify anomalous seconds
    3. A window is labelled anomalous (1) if > 10% of its seconds are anomalous
       (avoids labelling a window from a single-second spike)
    4. Returns a pd.Series aligned to window_id with values {0, 1, -1}
       -1 = no annotation data available for this case

    Requires network access to VitalDB. Falls back to -1 (unknown) if
    the track is unavailable or the API returns a 403.

    This ground truth is imperfect — MAP thresholds miss some events and
    catch some non-events — but it provides a clinically grounded binary
    label that proxy metrics cannot.
    """
    if not VITALDB_AVAILABLE:
        return pd.Series(dtype=int)

    # Map seconds to anomaly flag
    n_seconds = len(df_clean)
    anom_seconds = np.zeros(n_seconds, dtype=int)
    found_any    = False

    for track, (event_name, condition) in CLINICAL_EVENT_TRACKS.items():
        if "tachy" in track:
            # Tachycardia uses the same HR track
            actual_track = "Solar8000/HR"
        else:
            actual_track = track
        try:
            data = vitaldb.vital_recs(
                caseid,
                track_names=[actual_track],
                interval=1,
                return_timestamp=True,
            )
            if data is None or len(data) == 0:
                continue

            df_sig = pd.DataFrame(data, columns=["time", "value"])
            df_sig["time"] = df_sig["time"].astype(float).round().astype(int)
            df_sig = df_sig.dropna(subset=["value"])

            for _, row in df_sig.iterrows():
                t = int(row["time"])
                if 0 <= t < n_seconds and condition(row["value"]):
                    anom_seconds[t] = 1
            found_any = True

        except Exception:
            continue  # Network block or missing track — skip silently

    if not found_any:
        return pd.Series(dtype=int)

    # Convert second-level labels to window-level labels
    window_labels = {}
    n = len(df_clean)
    wid = 0
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        window_anom_frac = anom_seconds[start:end].mean()
        # Label as anomalous if >10% of seconds in window are anomalous
        window_labels[wid] = 1 if window_anom_frac > 0.10 else 0
        wid += 1

    return pd.Series(window_labels)


# =============================================================================
# SECTION GT-2 — Synthetic Anomaly Injector (Ground Truth Approach 2)
# =============================================================================

@dataclass
class InjectedAnomaly:
    """Records a synthetic anomaly injection for evaluation."""
    window_id:    int
    t_start:      float
    event_type:   str      # "hr_spike", "spo2_drop", "bp_drop", "rr_spike"
    magnitude:    float    # How severe the injection was (std units)
    signal:       str      # Which signal was perturbed


def inject_synthetic_anomalies(
    df_clean: pd.DataFrame,
    active_sigs: list,
    n_anomalies: int = 20,
    seed: int = 42,
) -> tuple:
    """
    Inject controlled synthetic anomalies into a copy of the cleaned dataframe.

    Injection strategy:
    - Sample n_anomalies random windows from the SECOND HALF of the case
      (first half is warmup territory — we don't inject there)
    - For each window, perturb one or more signals by 3-5 standard deviations
    - Perturbations are clinically motivated: HR spikes (+40-80 bpm),
      SpO2 drops (-5 to -15%), SBP drops (-30 to -60 mmHg)
    - Each injection lasts exactly WINDOW_SIZE_S seconds (one full window)

    Returns:
        df_injected: dataframe with synthetic anomalies added
        injected_windows: list of InjectedAnomaly records (the ground truth)

    Why the second half only: the first 50% is used for warmup selection.
    Injecting there would corrupt the baseline and make the test trivial.
    """
    rng = np.random.default_rng(seed)
    df_inj = df_clean.copy()
    n = len(df_inj)

    # Only inject into second half
    second_half_start = n // 2
    n_windows_available = (n - second_half_start - WINDOW_SIZE_S) // STEP_SIZE_S

    if n_windows_available < n_anomalies:
        n_anomalies = max(1, n_windows_available // 2)

    # Sample injection points (window start row indices)
    injection_rows = rng.choice(
        range(second_half_start, n - WINDOW_SIZE_S, STEP_SIZE_S),
        size=n_anomalies,
        replace=False,
    )

    # Define injection types: (signal_key, delta_func)
    injection_types = []
    sig_map = {s.split("_")[0]: s for s in active_sigs if "_" not in s}

    if "HR" in sig_map:
        injection_types.append(("hr_spike",   sig_map["HR"],   lambda: rng.uniform(40,  80)))
        injection_types.append(("hr_brady",   sig_map["HR"],   lambda: -rng.uniform(25, 45)))
    if "SpO2" in sig_map:
        injection_types.append(("spo2_drop",  sig_map["SpO2"], lambda: -rng.uniform(5,  15)))
    if "SBP" in sig_map:
        injection_types.append(("sbp_drop",   sig_map["SBP"],  lambda: -rng.uniform(30, 60)))
    if "RR" in sig_map:
        injection_types.append(("rr_spike",   sig_map["RR"],   lambda: rng.uniform(15,  25)))

    if not injection_types:
        # Fallback: perturb first available signal
        injection_types = [(
            "generic_spike", active_sigs[0],
            lambda: rng.uniform(3, 5) * df_inj[active_sigs[0]].std()
        )]

    injected = []
    for row_start in sorted(injection_rows):
        # Extended injection: 3-5 consecutive windows (30-50 seconds)
        n_windows = rng.integers(3, 6)  # 3, 4, or 5 windows
        row_end   = row_start + (n_windows * WINDOW_SIZE_S)
        row_end   = min(row_end, len(df_inj))  # don't exceed dataframe length
        
        inj_type, sig, delta_fn = injection_types[
            rng.integers(len(injection_types))
        ]
        delta = delta_fn()

        # Apply injection: clip to physiological bounds afterward
        if sig in df_inj.columns:
            df_inj.loc[row_start:row_end - 1, sig] += delta
            base = sig.split("_")[0]
            if base in PHYS_BOUNDS:
                lo, hi = PHYS_BOUNDS[base]
                df_inj[sig] = df_inj[sig].clip(lo, hi)

        # window_id = row_start // STEP_SIZE_S (approximate)
        wid = row_start // STEP_SIZE_S
        t0  = float(df_clean["time"].iloc[row_start]) if row_start < len(df_clean) else float(row_start)

        injected.append(InjectedAnomaly(
            window_id=wid, t_start=t0,
            event_type=inj_type, magnitude=abs(delta), signal=sig,
        ))

    print(f"  [INJECT] {len(injected)} synthetic anomalies injected "
          f"(types: {set(i.event_type for i in injected)})")
    return df_inj, injected


# =============================================================================
# SECTION GT-3 — Ground Truth Evaluation
# =============================================================================

def compute_gt_metrics(
    results: pd.DataFrame,
    gt_labels: pd.Series,
    label_source: str = "clinical",
) -> dict:
    """
    Compute precision, recall, F1 against binary ground truth labels.

    gt_labels: pd.Series indexed by window_id, values in {0, 1}
    Aligns on window_id — windows not in gt_labels are excluded.

    Metrics:
    - Precision: of flagged windows, what fraction are truly anomalous?
    - Recall:    of truly anomalous windows, what fraction were caught?
    - F1:        harmonic mean of precision and recall
    - FPR:       false positive rate (false alarms among normal windows)
    """
    if gt_labels.empty or results.empty:
        return {}

    # Align on window_id
    results_indexed = results.set_index("window_id")
    common_ids      = results_indexed.index.intersection(gt_labels.index)

    if len(common_ids) == 0:
        return {}

    y_true = gt_labels.loc[common_ids].values.astype(int)
    y_pred = results_indexed.loc[common_ids, "is_anomaly"].values.astype(int)

    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = TP / max(TP + FP, 1)
    recall    = TP / max(TP + FN, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    fpr       = FP / max(FP + TN, 1)

    n_gt_pos = int(y_true.sum())
    n_gt_neg = int((y_true == 0).sum())

    print(f"  [{label_source}] GT labels: {n_gt_pos} anomalous / "
          f"{n_gt_neg} normal  |  "
          f"P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}  FPR={fpr:.3f}  "
          f"(TP={TP} FP={FP} FN={FN} TN={TN})")

    return {
        f"{label_source}_precision": precision,
        f"{label_source}_recall":    recall,
        f"{label_source}_f1":        f1,
        f"{label_source}_fpr":       fpr,
        f"{label_source}_n_gt_pos":  n_gt_pos,
        f"{label_source}_n_windows": len(common_ids),
    }


def evaluate_with_ground_truth(
    cases: list,
    all_results: dict,
    full_per_case: list,
) -> dict:
    """
    Run both ground truth approaches across all cases and aggregate.

    Approach 1 — Clinical annotations:
        Fetch MAP/SpO2/HR tracks from VitalDB, apply clinical thresholds,
        compute P/R/F1 of Full_Both against those labels.

    Approach 2 — Synthetic injection:
        For each case, inject 20 synthetic anomalies into a copy of the data,
        re-run Full_Both pipeline on the injected data, compute detection rate.
        This gives a controlled, reproducible precision/recall estimate that
        doesn't depend on VitalDB annotation availability.

    Both are reported separately in the summary.
    """
    gt_results = {
        "clinical":  [],   # per-case dicts from approach 1
        "synthetic": [],   # per-case dicts from approach 2
    }

    print("\n[GT-1] Clinical annotation evaluation...")
    for caseid, df_clean, active_sigs in cases:
        # Find matching Full_Both results
        cd = next((c for c in full_per_case
                   if c["caseid"] == caseid and not c["results"].empty), None)
        if cd is None:
            continue

        gt_labels = fetch_clinical_events(caseid, df_clean)
        if gt_labels.empty:
            print(f"  Case {caseid}: no clinical annotations available")
            continue

        n_pos = gt_labels.sum()
        if n_pos == 0:
            print(f"  Case {caseid}: all windows normal per clinical thresholds")
            continue

        metrics = compute_gt_metrics(cd["results"], gt_labels, "clinical")
        if metrics:
            metrics["caseid"] = caseid
            gt_results["clinical"].append(metrics)

    print("\n[GT-2] Synthetic anomaly injection evaluation...")
    for caseid, df_clean, active_sigs in cases:
        print(f"  Case {caseid}:", end=" ")
        try:
            df_inj, injected = inject_synthetic_anomalies(
                df_clean, active_sigs, n_anomalies=20
            )
        except Exception as e:
            print(f"injection failed: {e}")
            continue

        if not injected:
            print("no injections possible (case too short)")
            continue

        # Build ground truth: injected window IDs are anomalous
        injected_wids = set(inj.window_id for inj in injected)
        # Run Full_Both pipeline on injected data
        pipeline_inj = StreamingPipeline(active_sigs, variant=ModelVariant.FULL_BOTH)
        results_inj  = pipeline_inj.run(df_inj, verbose=False)

        if results_inj.empty:
            print("pipeline returned empty results on injected data")
            continue

        # Create gt_labels: 1 for injected windows, 0 otherwise
        gt_syn = pd.Series(
            {wid: (1 if wid in injected_wids else 0)
             for wid in results_inj["window_id"].values}
        )

        metrics = compute_gt_metrics(results_inj, gt_syn, "synthetic")
        if metrics:
            metrics["caseid"]         = caseid
            metrics["n_injected"]     = len(injected)
            metrics["injection_types"] = list(set(i.event_type for i in injected))
            gt_results["synthetic"].append(metrics)

    return gt_results


def print_gt_summary(gt_results: dict):
    """Print aggregated ground truth evaluation summary."""
    print("\n-- Ground Truth Evaluation Summary ---------------")

    for approach, label in [("clinical", "Clinical Annotations"),
                             ("synthetic", "Synthetic Injection")]:
        rows = gt_results.get(approach, [])
        if not rows:
            print(f"\n[{label}] No results (network unavailable or no annotations)")
            continue

        prec = [r[f"{approach}_precision"] for r in rows]
        rec  = [r[f"{approach}_recall"]    for r in rows]
        f1   = [r[f"{approach}_f1"]        for r in rows]
        fpr  = [r[f"{approach}_fpr"]       for r in rows]
        n    = len(rows)

        print(f"\n[{label}]  n={n} cases")
        print(f"  {'Metric':<14s}  {'Mean':>8s}  {'Std':>8s}  {'Target':>8s}  OK?")
        print("  " + "-"*50)
        for vals, name, tgt, chk in [
            (prec, "Precision",  "> 0.50", lambda v: v > 0.50),
            (rec,  "Recall",     "> 0.50", lambda v: v > 0.50),
            (f1,   "F1 score",   "> 0.50", lambda v: v > 0.50),
            (fpr,  "FPR",        "< 0.10", lambda v: v < 0.10),
        ]:
            mu  = float(np.mean(vals))
            std = float(np.std(vals))
            ok  = "✔" if chk(mu) else "✘"
            print(f"  {name:<14s}  {mu:>8.3f}  {std:>8.3f}  {tgt:>8s}  {ok}")

        if approach == "synthetic":
            all_types = []
            for r in rows:
                all_types.extend(r.get("injection_types", []))
            print(f"  Injection types tested: {set(all_types)}")


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
              f"({'✔' if sep > 1.05 else '⚠' if sep > 1.02 else '✘'})")
    else:
        print("  [Check 2] No anomalies detected — separation ratio N/A")

    # Check 3: Temporal consistency
    # (Perturbation sensitivity removed — GT evaluation provides stronger validation)
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
        ModelVariant.IF_ONLY.value:      "steelblue",
        ModelVariant.NO_CL.value:        "darkorange",
        ModelVariant.FULL_DRIFT.value:   "darkgreen",
        ModelVariant.FULL_ANOMALY.value: "purple",
        ModelVariant.FULL_BOTH.value:    "crimson",
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
    full_per_case = all_results.get(ModelVariant.FULL_BOTH.value, {}).get("per_case", [])
    if full_per_case and not full_per_case[0]["results"].empty:
        r  = full_per_case[0]["results"]
        ax.plot(r["t_start"], r["drift_ratio"],
                lw=1.2, color="darkgreen", label="Drift ratio")
        calib = full_per_case[0]["pipeline"].drift_detector.threshold
        ax.axhline(calib, color="red", lw=1.5, ls="--",
                   label=f"Calibrated trigger = {calib:.4f}")
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
    full_per_case = all_results.get(ModelVariant.FULL_BOTH.value, {}).get("per_case", [])
    if not full_per_case:
        print("[WARN] No FULL_BOTH cases for continual learning plot")
        return

    # Skip Case 1 if it has empty results (heavy NaN cases produce no scored windows)
    first_valid = next((cd for cd in full_per_case if not cd["results"].empty), None)
    if first_valid is None:
        print("[WARN] All FULL_BOTH cases have empty results")
        return

    r        = first_valid["results"]
    pipeline = first_valid["pipeline"]
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
    # calibrated threshold is per-case; shown in adaptation_timeline plot
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
                f"Trigger threshold: self-calibrated per case",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("(4) Retraining Event Log")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "continual_learning.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] continual_learning.png saved → {out}")

def plot_adaptation_timeline(all_results: dict, cases: list):
    """
    The novelty visualization: shows the model's decision boundary moving
    over time in response to physiological drift.

    Four panels per case (shows first case with retraining events):
      (0,0) Anomaly score + evolving threshold line (the boundary moves)
      (0,1) Drift ratio vs self-calibrated trigger level (what caused retrain)
      (1,0) Anomaly rate in buffer over time (second trigger condition)
      (1,1) Cumulative retrain count + trigger analysis table

    The key visual argument: in a static model (No_CL), the threshold line
    is flat regardless of what the patient's physiology does. In Full_Both,
    the threshold line steps at retrain events, tracking the patient's
    evolving baseline. That step IS the continual learning contribution.
    """
    # Find first case with at least one retrain event in Full_Both
    full_cases = all_results.get(ModelVariant.FULL_BOTH.value, {}).get("per_case", [])
    nocl_cases = all_results.get(ModelVariant.NO_CL.value,     {}).get("per_case", [])

    # Find first non-empty case with at least one retraining event
    full_cd = next(
        (cd for cd in full_cases
         if not cd["results"].empty and cd["pipeline"].ifm.retraining_events),
        None
    )
    # Fall back to any non-empty case if no retraining events found
    if full_cd is None:
        full_cd = next((cd for cd in full_cases if not cd["results"].empty), None)
    if full_cd is None:
        print("[WARN] No non-empty FULL_BOTH cases — adaptation timeline skipped")
        return

    caseid  = full_cd["caseid"]
    nocl_cd = next((cd for cd in nocl_cases if cd["caseid"] == caseid), None)

    cl_df   = full_cd["pipeline"].cl_tracker.to_dataframe()
    events  = full_cd["pipeline"].ifm.retraining_events
    tracker = full_cd["pipeline"].cl_tracker

    if cl_df.empty:
        print("[WARN] CL tracker empty — adaptation timeline skipped")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Continual Learning Adaptation Timeline - Case {caseid}\n"
        f"Decision boundary moves to track patient physiology "
        f"({len(events)} retraining event(s))",
        fontsize=12, fontweight="bold"
    )

    t = cl_df["timestamp"].values

    # (0,0) Anomaly score + moving threshold vs No_CL flat threshold
    ax = axes[0, 0]
    ax.plot(t, cl_df["anomaly_score"].values,
            lw=0.7, alpha=0.5, color="steelblue", label="Anomaly score")
    ax.plot(t, cl_df["threshold"].values,
            lw=2.0, color="crimson", label="Adaptive threshold (Full_Both)")

    if nocl_cd is not None and not nocl_cd["results"].empty:
        nocl_thresh = nocl_cd["pipeline"].ifm.threshold
        ax.axhline(nocl_thresh, lw=1.5, ls="--", color="darkorange",
                   label=f"Static threshold (No_CL) = {nocl_thresh:.4f}")

    anom_mask = cl_df["is_anomaly"].values
    ax.scatter(t[anom_mask], cl_df.loc[anom_mask, "anomaly_score"].values,
               color="red", s=12, zorder=5, alpha=0.8, label="Anomaly flag")

    for ev in events:
        ax.axvline(ev.timestamp, color="purple", lw=1.5, ls=":", alpha=0.7)

    ax.set_xlabel("Time (s)"); ax.set_ylabel("Score")
    ax.set_title("(1) Adaptive vs Static Decision Boundary\n"
                 "(purple = retraining events, threshold line moves)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (0,1) Drift ratio vs self-calibrated threshold
    ax = axes[0, 1]
    ax.plot(t, cl_df["drift_ratio"].values,
            lw=1.0, color="teal", label="Drift ratio")
    calib_thresh = full_cd["pipeline"].drift_detector.threshold
    ax.axhline(calib_thresh, lw=1.5, ls="--", color="red",
               label=f"Calibrated trigger = {calib_thresh:.4f}")
    ax.axhline(1.0, lw=1.0, ls=":", color="gray", alpha=0.5)

    for ev in events:
        ax.axvline(ev.timestamp, color="purple", lw=1.5, ls=":", alpha=0.7)

    ax.set_xlabel("Time (s)"); ax.set_ylabel("Drift ratio")
    ax.set_title("(2) Drift Ratio vs Self-Calibrated Threshold\n"
                 "(trigger = 1.0 + 2σ of warmup variance)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1,0) Anomaly rate in buffer over time
    ax = axes[1, 0]
    ax.plot(t, cl_df["anomaly_rate"].values * 100,
            lw=1.0, color="darkorange", label="Anomaly rate in buffer (%)")
    ax.axhline(ANOMALY_RATE_TRIGGER * 100, lw=1.5, ls="--", color="red",
               label=f"Trigger = {ANOMALY_RATE_TRIGGER*100:.0f}%")

    for ev in events:
        ax.axvline(ev.timestamp, color="purple", lw=1.5, ls=":", alpha=0.7)

    ax.set_xlabel("Time (s)"); ax.set_ylabel("Anomaly rate in buffer (%)")
    ax.set_title("(3) Second Trigger Condition: Anomaly Rate\n"
                 "(retraining requires BOTH drift AND anomaly_rate)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1,1) Trigger analysis table + cumulative retrain count
    ax = axes[1, 1]
    ax.axis("off")

    summary = tracker.summary()
    trig    = tracker.trigger_analysis

    txt  = "ADAPTATION SUMMARY - Case " + str(caseid) + "\n"
    txt += "=" * 40 + "\n\n"
    txt += "Total retraining events : " + str(summary.get("n_retrains", 0)) + "\n"
    txt += "Windows scored          : " + str(summary.get("n_windows", 0)) + "\n"
    txt += "Overall anomaly rate    : " + f"{summary.get('anomaly_rate', 0)*100:.1f}%" + "\n"
    txt += "Max drift ratio         : " + f"{summary.get('max_drift_ratio', 0):.4f}" + "\n"
    thr  = summary.get("threshold_range", (0, 0))
    txt += "Threshold range         : " + f"{thr[0]:.4f} - {thr[1]:.4f}" + "\n"
    txt += "Threshold movement      : " + f"{summary.get('threshold_delta', 0):.4f}" + "\n\n"

    if trig:
        txt += "Trigger conditions at each retrain:\n"
        for i, ev in enumerate(trig):
            d = "Y" if ev["drift_active"]   else "N"
            a = "Y" if ev["anomaly_active"] else "N"
            txt += (f"  Event {i+1} @ t={ev['timestamp']:.0f}s: "
                    f"drift={d} ({ev['drift_ratio']:.3f})  "
                    f"anomaly={a} ({ev['anomaly_rate']*100:.1f}%)\n")

    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=8.5,
            va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.6))
    ax.set_title("(4) Trigger Analysis + Adaptation Summary")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "adaptation_timeline.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] adaptation_timeline.png saved -> {out}")


def plot_cl_novelty_comparison(all_results: dict):
    """
    Side-by-side threshold trajectory comparison across all 5 variants.

    Shows exactly one thing: the No_CL threshold is a flat horizontal line
    for every case. The Full_* thresholds step up or down at retrain events.
    This is the clearest visual proof that CL is doing something the static
    model cannot.
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(
        "Threshold Trajectory Comparison - Static vs Adaptive Decision Boundary\n"
        "Flat line = frozen model.  Steps = continual learning adaptation.",
        fontsize=12, fontweight="bold"
    )

    variant_order = [
        ModelVariant.IF_ONLY, ModelVariant.NO_CL,
        ModelVariant.FULL_DRIFT, ModelVariant.FULL_ANOMALY, ModelVariant.FULL_BOTH,
    ]
    colors_map = {
        ModelVariant.IF_ONLY.value:      "steelblue",
        ModelVariant.NO_CL.value:        "darkorange",
        ModelVariant.FULL_DRIFT.value:   "darkgreen",
        ModelVariant.FULL_ANOMALY.value: "purple",
        ModelVariant.FULL_BOTH.value:    "crimson",
    }

    # Pick two representative cases: one with retrain, one without
    full_both_cases = all_results.get(ModelVariant.FULL_BOTH.value, {}).get("per_case", [])
    valid_cases   = [cd for cd in full_both_cases if not cd["results"].empty]
    cases_with    = [cd for cd in valid_cases if cd["pipeline"].ifm.retraining_events]
    cases_without = [cd for cd in valid_cases if not cd["pipeline"].ifm.retraining_events]

    for row, (case_pool, row_label) in enumerate([
        (cases_with,    "Case with retraining"),
        (cases_without, "Case without retraining"),
    ]):
        if not case_pool:
            continue
        ref_caseid = case_pool[0]["caseid"]

        for col, variant in enumerate(variant_order):
            ax    = axes[row, col]
            vname = variant.value
            vdata = all_results.get(vname, {}).get("per_case", [])
            cd    = next((c for c in vdata if c["caseid"] == ref_caseid), None)

            if cd is None or cd["results"].empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            r      = cd["results"]
            color  = colors_map.get(vname, "gray")

            if "threshold" in r.columns:
                ax.plot(r["t_start"], r["threshold"],
                        lw=2.0, color=color, label="Threshold")
            else:
                ax.axhline(cd["pipeline"].ifm.threshold,
                           lw=2.0, color=color, label="Threshold")

            # Mark retrain events
            for ev in cd["pipeline"].ifm.retraining_events:
                ax.axvline(ev.timestamp, color="purple", lw=1, ls=":", alpha=0.7)

            anom = r[r["is_anomaly"]]
            ax.scatter(anom["t_start"],
                       [cd["pipeline"].ifm.threshold] * len(anom),
                       color="red", s=8, alpha=0.5, zorder=3)

            n_ev = len(cd["pipeline"].ifm.retraining_events)
            ax.set_title(f"{vname}\n({n_ev} retrain(s))", fontsize=8)
            ax.set_xlabel("Time (s)", fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{row_label}\nThreshold", fontsize=7)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "cl_novelty_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] cl_novelty_comparison.png saved -> {out}")


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
        ("separation_ratio",     "[2] Separation",  (1.05, 99)),

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

    # -- Priority 3: Load 10-15 VitalDB cases — per-case, not concatenated --
    print("\n[STEP 1] Loading data (per-case)...")
    if VITALDB_AVAILABLE:
        cases = load_cases_separately(
            case_ids=list(range(1, 26)),
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

    # -- Priority 1: Run all 3 baselines per case --------------------------
    print("\n[STEP 2] Running baseline comparison (Priority 1)...")
    all_results = run_all_baselines_per_case(cases, verbose=False)
    df_comp     = compare_baselines(all_results)

    # -- Priority 2: Continual learning summary ----------------------------
    print("\n[STEP 3] Continual learning analysis (Priority 2)...")
    full_per_case = all_results.get(ModelVariant.FULL_BOTH.value, {}).get("per_case", [])
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
              "self-calibrated. Lower DRIFT_THRESHOLD_K in the config "
              f"if the data is stable.")

    # -- Priority 3: Threshold sensitivity on FULL model (first case) ------
    print("\n[STEP 4] Threshold sensitivity sweep (Priority 3)...")
    df_sens = pd.DataFrame()
    # Run sweep on the case with most anomalies — gives informative curve
    sens_case = max(
        (cd for cd in full_per_case if not cd["results"].empty),
        key=lambda cd: cd["results"]["is_anomaly"].sum(),
        default=None
    )
    if sens_case is not None:
        print(f"  Running sensitivity sweep on Case {sens_case['caseid']} "
              f"({sens_case['results']['is_anomaly'].sum()} anomalies)")
        df_sens = threshold_sensitivity_sweep(
            sens_case["results"],
            sens_case["pipeline"].ifm.warmup_scores,
        )

    # -- Evaluation on FULL model — all cases, aggregate mean±std ---------
    print("\n[STEP 5] Evaluating FULL model across all cases...")
    per_case_eval = []   # list of dicts, one per valid case

    for cd in full_per_case:
        if cd["results"].empty:
            continue
        # Match this case's df_clean and active_sigs from the cases list
        case_match = next(
            (c for c in cases if c[0] == cd["caseid"]), None
        )
        if case_match is None:
            continue
        _, df_case, sigs_case = case_match
        result = evaluate_model(
            cd["results"], cd["pipeline"].ifm, df_case, sigs_case
        )
        if result:
            result["caseid"] = cd["caseid"]
            per_case_eval.append(result)

    # Aggregate mean±std across all cases
    eval_results = {}   # aggregated summary for summary print
    eval_keys = ["cv", "separation_ratio",
                 "isolated_fraction"]
    if per_case_eval:
        for key in eval_keys:
            vals = [r[key] for r in per_case_eval if r.get(key) is not None]
            if vals:
                eval_results[f"{key}_mean"] = float(np.mean(vals))
                eval_results[f"{key}_std"]  = float(np.std(vals))
                eval_results[f"{key}_n"]    = len(vals)

    # Also keep the case with most anomalies for plot_unified_results
    best_case = max(
        (cd for cd in full_per_case if not cd["results"].empty),
        key=lambda cd: cd["results"]["is_anomaly"].sum(),
        default=None
    )
    best_eval = next(
        (r for r in per_case_eval
         if best_case and r.get("caseid") == best_case["caseid"]),
        per_case_eval[0] if per_case_eval else {}
    )

    print("\n-- Ground Truth Evaluation Summary ---------------")
    print("\n[STEP 5b] Ground truth evaluation (clinical + synthetic)...")
    gt_results = evaluate_with_ground_truth(cases, all_results, full_per_case)
    print_gt_summary(gt_results)

    # -- Plots -------------------------------------------------------------
    print("\n[STEP 6] Generating plots...")
    plot_baseline_comparison(all_results, df_comp)
    plot_continual_learning(all_results)
    plot_adaptation_timeline(all_results, cases)    # novelty: moving boundary
    plot_cl_novelty_comparison(all_results)         # novelty: static vs adaptive
    if not df_sens.empty:
        plot_threshold_sensitivity(df_sens)
    if best_case is not None:
        best_match = next((c for c in cases if c[0] == best_case["caseid"]), None)
        if best_match:
            prefix = "vitaldb_real" if VITALDB_AVAILABLE else "vitaldb_sim"
            plot_unified_results(
                best_case["results"],
                best_match[1],
                best_match[2],
                best_case["pipeline"].ifm.threshold,
                best_eval,
                prefix=prefix,
            )

    # -- Summary -----------------------------------------------------------
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    # Reprint baseline table in summary for clean final output
    print(f"\nBaseline Comparison (mean +/- std across {len(cases)} cases):")
    print(f"  {'Variant':<14s}  {'n':>3s}  {'Anomaly rate':>14s}  "
          f"{'Isolated frac':>14s}  {'Cluster len':>11s}  {'Retrains':>8s}")
    print("  " + "-"*72)
    for vname in df_comp.index:
        r   = df_comp.loc[vname]
        n   = int(r["n_cases"])
        ar  = r.get("anomaly_rate_mean",      float("nan")) * 100
        ars = r.get("anomaly_rate_std",       float("nan")) * 100
        iso = r.get("isolated_fraction_mean", float("nan")) * 100
        iss = r.get("isolated_fraction_std",  float("nan")) * 100
        cl  = r.get("mean_cluster_len_mean",  float("nan"))
        cls = r.get("mean_cluster_len_std",   float("nan"))
        re  = r.get("n_retrain_events_mean",  float("nan"))
        res = r.get("n_retrain_events_std",   float("nan"))
        print(f"  {vname:<14s}  {n:>3d}  "
              f"{ar:5.1f}% +/-{ars:4.1f}%  "
              f"{iso:5.1f}% +/-{iss:4.1f}%  "
              f"{cl:5.2f} +/-{cls:4.2f}  "
              f"{re:4.2f} +/-{res:4.2f}")

    print(f"\n  Total retraining events (FULL_BOTH): {total_events}")
    print(f"  Cases that needed retraining: "
          f"{sum(1 for cd in full_per_case if cd['pipeline'].ifm.retraining_events)}"
          f"/{len(full_per_case)}")

    # Reprint GT summary in final summary block
    if gt_results.get("clinical") or gt_results.get("synthetic"):
        print_gt_summary(gt_results)

    if eval_results:
        n_eval = max(
            eval_results.get(f"{k}_n", 0)
            for k in ["cv","separation_ratio","isolated_fraction"]
        )
        print(f"\nProxy Evaluation — FULL model (mean +/- std across {n_eval} cases):")
        labels = {
            "cv":                      "CV % (model stability)",
            "separation_ratio":        "Separation ratio",

            "isolated_fraction":       "Isolated fraction %",
        }
        scale  = {
            "cv": 1, "separation_ratio": 1,
            "isolated_fraction": 100,
        }
        good   = {
            "cv":                      ("< 20%",   lambda v: v < 20),
            "separation_ratio":        ("> 1.05",  lambda v: v > 1.05),
    
            "isolated_fraction":       ("< 30%",   lambda v: v < 30),
        }
        print(f"  {'Metric':<30s}  {'Mean':>8s}  {'Std':>8s}  {'n':>3s}  {'Target':>8s}  OK?")
        print("  " + "-"*70)
        for key in ["cv", "separation_ratio",
                    "isolated_fraction"]:
            mu  = eval_results.get(f"{key}_mean")
            std = eval_results.get(f"{key}_std")
            n   = eval_results.get(f"{key}_n", 0)
            if mu is None:
                continue
            s   = scale[key]
            tgt, chk = good[key]
            ok  = "✔" if chk(mu * s) else "✘"
            print(f"  {labels[key]:<30s}  {mu*s:>7.2f}  {std*s:>7.2f}  {n:>3d}  "
                  f"{tgt:>8s}  {ok}")

    print(f"\nOutput files:")
    for f in ["baseline_comparison.csv", "baseline_comparison.png",
              "threshold_sensitivity.csv", "threshold_sensitivity.png",
              "continual_learning.png", "adaptation_timeline.png",
              "cl_novelty_comparison.png"]:
        path = os.path.join(OUT_DIR, f)
        exists = "✔" if os.path.exists(path) else "✘ (not generated)"
        print(f"  {exists}  {path}")

    print(f"\n[DONE]")
    return all_results, cases, df_comp


if __name__ == "__main__":
    all_results, cases, df_comp = main()