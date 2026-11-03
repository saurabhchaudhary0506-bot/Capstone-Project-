import numpy as np
import pandas as pd
from src.shipment_prep import (
    clean_shipments, build_features, PROMISED_HOURS, FEATURES,
)

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# "before" measurements (after dedup, before any filling)
dedup = raw.drop_duplicates()
n_spell_before = dedup["service_type"].nunique()
miss_dist = int(dedup["distance_km"].isna().sum())
miss_wt = int((dedup["parcel_weight_kg"].isna()).sum())
sentinel = int((dedup["parcel_weight_kg"] > 100).sum())
miss_val = int(dedup["declared_value"].isna().sum())
miss_pay = int(dedup["payment_mode"].isna().sum())

# clean + features 
clean, medians = clean_shipments(raw)
n_spell_after = clean["service_type"].nunique()
feat = build_features(clean, hubs)

feat["is_delayed"] = (feat["delivery_hours"] > feat["promised_hours"]).astype(int)

# write modelling table
feat.to_csv("data/modelling_table.csv", index=False)

# hours outlier rule (1.5 x IQR)
q1 = feat["delivery_hours"].quantile(0.25)
q3 = feat["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n_out = int(((feat["delivery_hours"] < lower) | (feat["delivery_hours"] > upper)).sum())

check_cols = FEATURES + ["delivery_hours", "is_delayed"]

checks = {
    "raw_rows": len(raw),
    "duplicate_rows_removed": len(raw) - len(dedup),
    "rows_after_dedup": len(dedup),
    "missing_distance_km_before": miss_dist,
    "missing_parcel_weight_before": miss_wt,
    "sentinel_weight_rows": sentinel,
    "missing_declared_value_before": miss_val,
    "missing_payment_mode_before": miss_pay,
    "service_type_spellings_before": n_spell_before,
    "service_type_spellings_after": n_spell_after,
    "declared_value_mean_inr": feat["declared_value_inr"].mean(),
    "declared_value_median_inr": feat["declared_value_inr"].median(),
    "log_declared_value_mean": feat["log_declared_value"].mean(),
    "log_declared_value_median": feat["log_declared_value"].median(),
    "median_distance_km": feat["distance_km"].median(),
    "median_parcel_weight_kg": feat["parcel_weight_kg"].median(),
    "median_declared_value_inr": feat["declared_value_inr"].median(),
    "rows_after_hub_merge": len(feat),
    "nulls_in_features_and_target": int(feat[check_cols].isna().sum().sum()),
    "delayed_count": int(feat["is_delayed"].sum()),
    "delay_rate": feat["is_delayed"].mean(),
    "hours_q1": q1,
    "hours_q3": q3,
    "hours_lower_bound": lower,
    "hours_upper_bound": upper,
    "hours_outlier_count": n_out,
}
checks_df = pd.DataFrame({"metric": list(checks.keys()),
                          "value": list(checks.values())})
checks_df["value"] = checks_df["value"].astype(float).round(4)
checks_df.to_csv("outputs/part_a_checks.csv", index=False)

# delay by group
feat["distance_band"] = pd.cut(
    feat["distance_km"], bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"], right=True,
)

rows = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = feat.groupby(dim, observed=True).agg(
        shipments=("is_delayed", "size"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean"),
    ).reset_index()
    g.insert(0, "dimension", dim)
    g = g.rename(columns={dim: "group"})
    rows.append(g)
by_group = pd.concat(rows, ignore_index=True)
by_group[["delay_rate", "mean_hours"]] = by_group[["delay_rate", "mean_hours"]].round(4)
by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)

# correlations
corr = feat.select_dtypes(include="number").corr().round(4)
corr.to_csv("outputs/part_a_correlations.csv")

print(checks_df)
print(by_group)