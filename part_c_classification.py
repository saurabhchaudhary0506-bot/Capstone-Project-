import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS

df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]

# Task 1: Split 
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
print(len(X_train), len(X_val), len(X_test))
print(y_train.sum(), y_val.sum(), y_test.sum())  


def cm_row(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)}


# Task 2: Blind baseline + Three Classifiers (Validation, 0.5 rule)
rows = [cm_row("blind_all_on_time", y_val, np.zeros(len(y_val), dtype=int))]

logit = Pipeline([("scaler", StandardScaler()),
                  ("model", LogisticRegression(max_iter=1000))]).fit(X_train, y_train)
rows.append(cm_row("logistic_regression", y_val, logit.predict(X_val)))

best_d, best_f1 = None, -1
for d in [2, 3, 4, 5, 6, 8, 10]:          
    t = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    f = f1_score(y_val, t.predict(X_val))
    if f > best_f1:
        best_d, best_f1 = d, f
tree = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=42).fit(X_train, y_train)
rows.append(cm_row(f"decision_tree_depth_{best_d}", y_val, tree.predict(X_val)))

rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
rows.append(cm_row("random_forest_default", y_val, rf.predict(X_val)))

models_df = pd.DataFrame(rows)
models_df.round(4).to_csv("outputs/part_c_models.csv", index=False)
print(models_df.round(4))

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Task 3: Which columns carry signal?
rf_all = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
skb = SelectKBest(f_classif, k="all").fit(X_train, y_train)
fs = pd.DataFrame({"feature": FEATURES,
                   "f_score": skb.scores_,
                   "rf_importance": rf_all.feature_importances_})
fs["f_rank"] = fs["f_score"].rank(ascending=False, method="first").astype(int)
fs["is_noise"] = fs["feature"].isin(NOISE_COLUMNS)
fs = fs[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
fs.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)
print(fs.sort_values("f_rank").round(4))
print("noise share of importance:", round(fs.loc[fs["is_noise"], "rf_importance"].sum(), 4))

sel_rows = []
for k in [5, 7, 10, 12, 17]:
    p = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                  ("model", RandomForestClassifier(n_estimators=100, random_state=42))])
    p.fit(X_train, y_train)
    sel_rows.append({"k": k, "val_f1": f1_score(y_val, p.predict(X_val))})
sel_df = pd.DataFrame(sel_rows)
sel_df.round(4).to_csv("outputs/part_c_selectk.csv", index=False)
print(sel_df.round(4))

# Task 4: Tuning on the 7 selected columns (selector fitted on train only)
selector = SelectKBest(f_classif, k=7).fit(X_train, y_train)
Xtr7 = selector.transform(X_train)
Xva7 = selector.transform(X_val)

grid_params = {"n_estimators": [50, 100, 150],
               "max_depth": [4, 6, 8],
               "min_samples_leaf": [1, 5, 10]}


def clean_params(d):
    return {k: (None if v is None else int(v)) for k, v in d.items()}


grid = GridSearchCV(RandomForestClassifier(random_state=42), grid_params, cv=5)
grid.fit(Xtr7, y_train)
rand = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid_params,
                          n_iter=9, cv=5, random_state=42)
rand.fit(Xtr7, y_train)
default7 = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr7, y_train)

grid_p = clean_params(grid.best_params_)
rand_p = clean_params(rand.best_params_)
default_p = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}

tuning = pd.DataFrame([
    {"search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 27 * 5,
     "best_params": str(grid_p), "best_cv_accuracy": grid.best_score_,
     "val_f1": f1_score(y_val, grid.best_estimator_.predict(Xva7))},
    {"search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 9 * 5,
     "best_params": str(rand_p), "best_cv_accuracy": rand.best_score_,
     "val_f1": f1_score(y_val, rand.best_estimator_.predict(Xva7))},
    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0,
     "total_fits": 1, "best_params": str(default_p), "best_cv_accuracy": np.nan,
     "val_f1": f1_score(y_val, default7.predict(Xva7))},
])
tuning.round(4).to_csv("outputs/part_c_tuning.csv", index=False)
print(tuning.round(4))

# Task 5: Choose the final configuration (ties: default, grid, randomized)
priority = ["default_forest_same_7_features", "grid_search", "randomized_search"]
best_f1 = tuning["val_f1"].max()
chosen_from = next(n for n in priority
                   if tuning.loc[tuning["search"] == n, "val_f1"].iloc[0] == best_f1)
chosen_params = {"default_forest_same_7_features": default_p,
                 "grid_search": grid_p, "randomized_search": rand_p}[chosen_from]

final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(random_state=42, **chosen_params)),
]).fit(X_train, y_train)
print("chosen:", chosen_from, chosen_params)

# Task 6: Alert threshold from validation probabilities
proba_val = final_model.predict_proba(X_val)[:, 1]
thr_rows = []
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    pred = (proba_val >= t).astype(int)
    r = cm_row("x", y_val, pred)
    r.pop("model")
    thr_rows.append({"threshold": t, **r})
thr = pd.DataFrame(thr_rows)
thr.round(4).to_csv("outputs/part_c_thresholds.csv", index=False)
print(thr.round(4))

ok = thr[thr["recall"] >= 0.80]
threshold = float(ok["threshold"].max()) if len(ok) else 0.1
print("chosen threshold:", threshold)

# Task 7: One test evaluation + final config 
proba_test = final_model.predict_proba(X_test)[:, 1]
pred_test = (proba_test >= threshold).astype(int)
r = cm_row("x", y_test, pred_test)
r.pop("model")
final_test = pd.DataFrame([{"threshold": threshold, **r}])
final_test.round(4).to_csv("outputs/part_c_final_test.csv", index=False)
print(final_test.round(4))

cfg = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": chosen_params["n_estimators"],
    "max_depth": "unlimited" if chosen_params["max_depth"] is None else chosen_params["max_depth"],
    "min_samples_leaf": chosen_params["min_samples_leaf"],
    "threshold": threshold,
    "chosen_from": chosen_from,
}])
cfg.round(4).to_csv("outputs/part_c_final_config.csv", index=False)