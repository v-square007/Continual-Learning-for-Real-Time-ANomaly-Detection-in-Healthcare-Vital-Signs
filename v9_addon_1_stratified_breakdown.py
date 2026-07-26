# =============================================================================
# ADD-ON 1 — Event-Type Stratified GT Breakdown (Full_Both)
# -----------------------------------------------------------------------------
# Paste as a NEW CELL at the end of v9code_output.ipynb, AFTER the main cell.
# It reuses functions/constants already defined by the v9 cell above
# (fetch_case, CLINICAL_EVENT_TRACKS, WINDOW_SIZE_S, STEP_SIZE_S,
#  compute_gt_metrics, inject_synthetic_anomalies, StreamingPipeline,
#  ModelVariant, VITALDB_AVAILABLE, OUT_DIR) — do not redefine those.
#
# BEFORE running this, change the last line of the v9 cell from:
#     if __name__ == "__main__":
#         main()
# to:
#     if __name__ == "__main__":
#         all_results, cases, df_comp = main()
# so that `all_results` and `cases` exist in the notebook namespace.
#
# WHY THIS EXISTS
# The paper currently reports one pooled recall number against clinical GT
# (0.109) and one pooled recall against synthetic injection (0.117). Both
# numbers average over four/five structurally different event types. If IF
# recall is uniformly ~0.11 across all types, that supports a general
# "IF underweights any localized deviation" story. If it's near-zero for
# some types and much higher for others, that's a different, more specific
# story worth reporting instead.
# =============================================================================
from v9_pipeline import *  

def fetch_clinical_events_typed(caseid, df_clean, window_size=WINDOW_SIZE_S,
                                 step=STEP_SIZE_S):
    """
    Same source data as fetch_clinical_events, but returns one label Series
    PER event type instead of a single OR-combined series.
    """
    if not VITALDB_AVAILABLE:
        return {}

    n = len(df_clean)
    type_seconds = {name: np.zeros(n, dtype=int)
                     for name, _ in CLINICAL_EVENT_TRACKS.values()}

    for track, (event_name, condition) in CLINICAL_EVENT_TRACKS.items():
        actual_track = "Solar8000/HR" if "tachy" in track else track
        try:
            data = vitaldb.vital_recs(caseid, track_names=[actual_track],
                                       interval=1, return_timestamp=True)
            if data is None or len(data) == 0:
                continue
            df_sig = pd.DataFrame(data, columns=["time", "value"])
            df_sig["time"] = df_sig["time"].astype(float).round().astype(int)
            df_sig = df_sig.dropna(subset=["value"])
            for _, row in df_sig.iterrows():
                t = int(row["time"])
                if 0 <= t < n and condition(row["value"]):
                    type_seconds[event_name][t] = 1
        except Exception:
            continue

    type_labels = {}
    for event_name, seconds in type_seconds.items():
        labels, wid = {}, 0
        for start in range(0, n - window_size + 1, step):
            end = start + window_size
            labels[wid] = 1 if seconds[start:end].mean() > 0.10 else 0
            wid += 1
        s = pd.Series(labels)
        if s.sum() > 0:          # only keep types that actually occurred
            type_labels[event_name] = s
    return type_labels


def stratified_clinical_breakdown(cases, full_per_case):
    """Precision/recall/F1/FPR per clinical event type, mean across cases."""
    per_type = {}

    for caseid, df_clean, active_sigs in cases:
        cd = next((c for c in full_per_case
                   if c["caseid"] == caseid and not c["results"].empty), None)
        if cd is None:
            continue
        typed_labels = fetch_clinical_events_typed(caseid, df_clean)
        for event_name, gt_labels in typed_labels.items():
            m = compute_gt_metrics(cd["results"], gt_labels, f"clinical_{event_name}")
            if m:
                m["caseid"] = caseid
                per_type.setdefault(event_name, []).append(m)

    print("\n" + "=" * 65)
    print("  STRATIFIED CLINICAL GT BREAKDOWN — Full_Both, by event type")
    print("=" * 65)
    print(f"  {'Event type':<14s}  {'n cases':>7s}  {'Precision':>10s}  "
          f"{'Recall':>8s}  {'F1':>8s}  {'FPR':>8s}")
    print("  " + "-" * 62)

    rows = []
    for event_name, recs in per_type.items():
        key = f"clinical_{event_name}"
        prec = float(np.mean([r[f"{key}_precision"] for r in recs]))
        rec  = float(np.mean([r[f"{key}_recall"]    for r in recs]))
        f1   = float(np.mean([r[f"{key}_f1"]        for r in recs]))
        fpr  = float(np.mean([r[f"{key}_fpr"]       for r in recs]))
        print(f"  {event_name:<14s}  {len(recs):>7d}  {prec:>10.3f}  "
              f"{rec:>8.3f}  {f1:>8.3f}  {fpr:>8.3f}")
        rows.append({"event_type": event_name, "n_cases": len(recs),
                     "precision": prec, "recall": rec, "f1": f1, "fpr": fpr})

    df_strat = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "clinical_breakdown_by_type.csv")
    df_strat.to_csv(out, index=False)
    print(f"\n  Saved -> {out}")
    return df_strat


def stratified_synthetic_breakdown(cases, all_results):
    """
    Precision/recall/F1/FPR per synthetic injection type (hr_spike, spo2_drop,
    sbp_drop, rr_spike, hr_brady), mean across cases. Every injection is
    single-channel by construction (see inject_synthetic_anomalies), so this
    shows whether recall varies by WHICH channel/direction is perturbed, not
    by how many channels.
    """
    per_type = {}

    for caseid, df_clean, active_sigs in cases:
        try:
            df_inj, injected = inject_synthetic_anomalies(df_clean, active_sigs,
                                                            n_anomalies=20)
        except Exception:
            continue
        if not injected:
            continue

        pipeline_inj = StreamingPipeline(active_sigs, variant=ModelVariant.FULL_BOTH)
        results_inj  = pipeline_inj.run(df_inj, verbose=False)
        if results_inj.empty:
            continue

        by_type = {}
        for inj in injected:
            by_type.setdefault(inj.event_type, []).append(inj.window_id)

        for event_type, wids in by_type.items():
            wid_set = set(wids)
            gt = pd.Series({wid: (1 if wid in wid_set else 0)
                            for wid in results_inj["window_id"].values})
            m = compute_gt_metrics(results_inj, gt, f"synthetic_{event_type}")
            if m:
                m["caseid"] = caseid
                per_type.setdefault(event_type, []).append(m)

    print("\n" + "=" * 65)
    print("  STRATIFIED SYNTHETIC INJECTION BREAKDOWN — Full_Both, by type")
    print("=" * 65)
    print(f"  {'Injection type':<14s}  {'n obs':>6s}  {'Precision':>10s}  "
          f"{'Recall':>8s}  {'F1':>8s}  {'FPR':>8s}")
    print("  " + "-" * 60)

    rows = []
    for event_type, recs in per_type.items():
        key = f"synthetic_{event_type}"
        prec = float(np.mean([r[f"{key}_precision"] for r in recs]))
        rec  = float(np.mean([r[f"{key}_recall"]    for r in recs]))
        f1   = float(np.mean([r[f"{key}_f1"]        for r in recs]))
        fpr  = float(np.mean([r[f"{key}_fpr"]       for r in recs]))
        print(f"  {event_type:<14s}  {len(recs):>6d}  {prec:>10.3f}  "
              f"{rec:>8.3f}  {f1:>8.3f}  {fpr:>8.3f}")
        rows.append({"injection_type": event_type, "n_obs": len(recs),
                     "precision": prec, "recall": rec, "f1": f1, "fpr": fpr})

    df_strat = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "synthetic_breakdown_by_type.csv")
    df_strat.to_csv(out, index=False)
    print(f"\n  Saved -> {out}")
    return df_strat