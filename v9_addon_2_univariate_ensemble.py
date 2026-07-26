from v9_pipeline import *

# =============================================================================
# ADD-ON 2 -- Univariate Ensemble Arm ("Full_Both + Univariate")  [FIXED v2]
# -----------------------------------------------------------------------------
# Root cause of both previous bugs:
#  v1: z-score computed over ALL active_sigs, including waveform-derived
#      features with near-zero warmup MAD -> blew up.
#  v2 (this fix): reference distribution was the MAD-FILTERED 60-window
#      warmup itself, which is deliberately the most stable slice of the
#      case (that's select_warmup_mad's job) -> its MAD is artificially
#      tiny, so ordinary later drift reads as multi-sigma outlier no matter
#      the threshold.
#
# This version computes the per-channel median/MAD from the RAW physiological
# values over the first 50% of the case (row-level, not the hand-picked
# 60 windows), giving a realistic estimate of natural variance during the
# stable period. Also still restricted to the 6 physiologically-bounded
# channels only.
# =============================================================================

def compute_univariate_flags(df_clean, active_sigs, results_df,
                              z_threshold=3.0, window_size=WINDOW_SIZE_S,
                              step=STEP_SIZE_S):
    phys_sigs = [sig for sig in active_sigs if sig.split("_")[0] in PHYS_BOUNDS]
    if not phys_sigs:
        return pd.Series(0, index=results_df["window_id"].values)

    max_row = int(len(df_clean) * 0.50)
    med, mad_sd = {}, {}
    for sig in phys_sigs:
        v = df_clean[sig].iloc[:max_row].values.astype(float)
        v = v[~np.isnan(v)]
        base = sig.split("_")[0]
        lo, hi = PHYS_BOUNDS[base]
        # Floor on mad_sd at 2% of the channel's physiological range. Prevents
        # a channel that happens to be near-constant in the first half (BT is
        # the usual culprit -- it moves in 0.1 degC steps and can sit flat for
        # long stretches) from producing near-infinite z on any later tick and
        # dominating max_z across all channels.
        floor = 0.02 * (hi - lo)
        if len(v) == 0:
            med[sig], mad_sd[sig] = 0.0, floor
            continue
        m = np.median(v)
        mad = np.median(np.abs(v - m))
        med[sig]    = m
        mad_sd[sig] = max(mad * 1.4826, floor)

    flags, wid = {}, 0
    for start in range(0, len(df_clean) - window_size + 1, step):
        chunk = df_clean.iloc[start:start + window_size]
        max_z = 0.0
        for sig in phys_sigs:
            v = chunk[sig].values.astype(float)
            v = v[~np.isnan(v)]
            if len(v) == 0:
                continue
            z = abs((v.mean() - med[sig]) / mad_sd[sig])
            max_z = max(max_z, z)
        flags[wid] = max_z
        wid += 1

    z_series = pd.Series(flags)
    common = results_df.set_index("window_id").index.intersection(z_series.index)
    return (z_series.loc[common] > z_threshold).astype(int)


def sweep_univariate_ensemble(cases, all_results, thresholds=(2.0, 2.5, 3.0, 3.5, 4.0)):
    full_per_case = all_results.get(ModelVariant.FULL_BOTH.value, {}).get("per_case", [])

    print("\n" + "=" * 65)
    print("  UNIVARIATE ENSEMBLE -- THRESHOLD SWEEP (Full_Both OR per-channel z)")
    print("=" * 65)
    print(f"  {'z-threshold':>11s}  {'Precision':>10s}  {'Recall':>8s}  "
          f"{'F1':>8s}  {'FPR':>8s}")
    print("  " + "-" * 52)

    rows = []
    for z_thr in thresholds:
        recs = []
        for caseid, df_clean, active_sigs in cases:
            cd = next((c for c in full_per_case
                       if c["caseid"] == caseid and not c["results"].empty), None)
            if cd is None:
                continue

            uni_flag = compute_univariate_flags(df_clean, active_sigs,
                                                 cd["results"], z_threshold=z_thr)

            res = cd["results"].set_index("window_id").copy()
            res["univariate_flag"] = 0
            res.loc[uni_flag.index, "univariate_flag"] = uni_flag.values
            res["is_anomaly"] = (res["is_anomaly"].astype(int) |
                                  res["univariate_flag"]).astype(bool)
            res = res.reset_index()

            gt = fetch_clinical_events(caseid, df_clean)
            if gt.empty or gt.sum() == 0:
                continue
            m = compute_gt_metrics(res, gt, "ens")
            if m:
                recs.append(m)

        if not recs:
            continue
        prec = float(np.mean([r["ens_precision"] for r in recs]))
        rec  = float(np.mean([r["ens_recall"]    for r in recs]))
        f1   = float(np.mean([r["ens_f1"]        for r in recs]))
        fpr  = float(np.mean([r["ens_fpr"]       for r in recs]))
        print(f"  {z_thr:>11.1f}  {prec:>10.3f}  {rec:>8.3f}  {f1:>8.3f}  {fpr:>8.3f}")
        rows.append({"z_threshold": z_thr, "precision": prec, "recall": rec,
                     "f1": f1, "fpr": fpr})

    print(f"\n  Full_Both alone (no ensemble): Precision=0.164  Recall=0.109  "
          f"F1=0.080  FPR=0.059")

    df_sweep = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "univariate_ensemble_sweep.csv")
    df_sweep.to_csv(out, index=False)
    print(f"\n  Saved -> {out}")
    return df_sweep