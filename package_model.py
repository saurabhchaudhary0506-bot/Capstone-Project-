import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from src.shipment_prep import FEATURES, clean_shipments

cfg = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
max_depth = None if str(cfg["max_depth"]) == "unlimited" else int(cfg["max_depth"])
params = {"n_estimators": int(cfg["n_estimators"]), "max_depth": max_depth,
          "min_samples_leaf": int(cfg["min_samples_leaf"])}
threshold = float(cfg["threshold"])

df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df["is_delayed"]
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

model = Pipeline([
    ("selector", SelectKBest(f_classif, k=int(cfg["selector_k"]))),
    ("model", RandomForestClassifier(random_state=42, **params)),
]).fit(X_train, y_train)

joblib.dump(model, "models/delay_model.pkl")
pd.DataFrame({"threshold": [threshold]}).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

_, medians = clean_shipments(pd.read_csv("data/raw_shipments.csv"))
pd.DataFrame({"column": list(medians), "median": list(medians.values())}
             ).round(4).to_csv("models/impute_values.csv", index=False)

loaded = joblib.load("models/delay_model.pkl")
p_mem = model.predict_proba(X_test)[:, 1]
p_load = loaded.predict_proba(X_test)[:, 1]
pd.DataFrame({"metric": ["test_rows_scored", "reload_max_abs_probability_diff", "threshold"],
              "value": [len(X_test), float(np.abs(p_mem - p_load).max()), threshold]}
             ).round(4).to_csv("outputs/part_d_reload_check.csv", index=False)
print("saved; reload diff:", float(np.abs(p_mem - p_load).max()))