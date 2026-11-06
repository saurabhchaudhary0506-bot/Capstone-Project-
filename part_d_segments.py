import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

# ---- Task 1: hub profile (ascending hub_id) ----
prof = df.groupby("hub_id").agg(
    shipments=("hub_id", "size"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean"),
).reset_index().sort_values("hub_id").reset_index(drop=True)
prof.round(4).to_csv("outputs/hub_profile.csv", index=False)

# ---- Task 2: scale the four columns ----
cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
Xs = StandardScaler().fit_transform(prof[cols])


# ---- Task 3: separation evidence ----
def across_over_within(X, labels):
    across, within = [], []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            d = np.linalg.norm(X[i] - X[j])
            (within if labels[i] == labels[j] else across).append(d)
    return np.mean(across) / np.mean(within)


sep_rows = []
for K in [2, 3, 4, 5]:
    labels = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(Xs)
    sizes = sorted(np.bincount(labels), reverse=True)
    sep_rows.append({"K": K,
                     "across_over_within": across_over_within(Xs, labels),
                     "cluster_sizes": "-".join(str(int(s)) for s in sizes)})
sep = pd.DataFrame(sep_rows)
sep.round(4).to_csv("outputs/part_d_separation.csv", index=False)
print(sep.round(4))

# ---- Task 4: K = 3, name segments by delay rate ----
prof["cluster"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(Xs)
order = prof.groupby("cluster")["delay_rate"].mean().sort_values().index
names = dict(zip(order, ["Standard", "Watch", "Escalate"]))
prof["segment"] = prof["cluster"].map(names)

seg = prof[["hub_id", "segment"]].merge(
    hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
seg.to_csv("outputs/hub_segments.csv", index=False)

seg_prof = prof.groupby("segment").agg(
    hubs=("hub_id", "size"),
    mean_delivery_hours=("mean_delivery_hours", "mean"),
    delay_rate=("delay_rate", "mean"),
    mean_distance_km=("mean_distance_km", "mean"),
    mean_n_stops=("mean_n_stops", "mean"),
).reindex(["Standard", "Watch", "Escalate"]).reset_index()
seg_prof.round(4).to_csv("outputs/segment_profile.csv", index=False)

implied = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatch = seg[seg["segment"] != seg["hub_tier"].map(implied)]
mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

print(seg_prof.round(4))
print(mismatch)