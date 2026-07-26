from v9_pipeline import main   # your renamed v9 file, with main() returning values
from v9_addon_1_stratified_breakdown import stratified_clinical_breakdown, stratified_synthetic_breakdown
from v9_addon_2_univariate_ensemble import sweep_univariate_ensemble

all_results, cases, df_comp = main()
stratified_clinical_breakdown(cases, all_results.get("Full_Both", {}).get("per_case", []))
stratified_synthetic_breakdown(cases, all_results)
sweep_univariate_ensemble(cases, all_results)