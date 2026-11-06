import joblib
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

new = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
imp = pd.read_csv("models/impute_values.csv")
medians = dict(zip(imp["column"], imp["median"].astype(float)))
segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]

clean, _ = clean_shipments(new, medians)
feat = build_features(clean, hubs)
feat["delay_probability"] = model.predict_proba(feat[order])[:, 1].round(4)
feat["flagged"] = (feat["delay_probability"] >= threshold).astype(int)
feat = feat.merge(segments, on="hub_id", how="left")
feat["action"] = np.where(
    (feat["flagged"] == 1) & (feat["segment"] == "Escalate"), "Escalate to hub manager",
    np.where(feat["flagged"] == 1, "Proactive customer call", "No action"))

out = feat[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
out.to_csv("outputs/new_shipment_actions.csv", index=False)
print(len(out), "rows;", int(out["flagged"].sum()), "flagged;",
      int((out["action"] == "Escalate to hub manager").sum()), "escalations")