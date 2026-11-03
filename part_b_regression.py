import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES

df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["delivery_hours"]

# Task 1: split 840 / 180 / 180
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

# Task 2: scaling without leakage 
scaler = StandardScaler().fit(X_train)
train_scaled = scaler.transform(X_train)
test_scaled = scaler.transform(X_test)
checks = {
    "train_rows": len(X_train),
    "val_rows": len(X_val),
    "test_rows": len(X_test),
    "train_scaled_max_abs_mean": np.abs(train_scaled.mean(axis=0)).max(),
    "test_scaled_max_abs_mean": np.abs(test_scaled.mean(axis=0)).max(),
}
pd.DataFrame({"metric": list(checks), "value": list(checks.values())}).round(4).to_csv(
    "outputs/part_b_checks.csv", index=False)


def score(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}


rows = []  

# Task 3: baseline
base_pred = np.full(len(y_val), y_train.mean())
rows.append({"model": "mean_baseline", "split": "val", **score(y_val, base_pred)})

# Task 4: tree-depth diagnostic
depths = [1, 2, 3, 4, 6, 8, 12, None]
depth_rows = []
best_depth, best_val = None, np.inf
for d in depths:                      
    t = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
    tr = np.sqrt(mean_squared_error(y_train, t.predict(X_train)))
    va = np.sqrt(mean_squared_error(y_val, t.predict(X_val)))
    depth_rows.append({"max_depth": "unlimited" if d is None else d,
                       "train_rmse": tr, "val_rmse": va})
    if va < best_val:                 # strict "<" keeps the smaller depth on ties
        best_val, best_depth = va, d
pd.DataFrame(depth_rows).round(4).to_csv("outputs/part_b_tree_depth.csv", index=False)

# Task 5: four candidate models
models = {
    "linear_regression": Pipeline([("scaler", StandardScaler()),
                                   ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42),
}
val_rmse = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    s = score(y_val, m.predict(X_val))
    val_rmse[name] = s["RMSE"]
    rows.append({"model": name, "split": "val", **s})

# Task 6: learning-rate experiment
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    g = GradientBoostingRegressor(n_estimators=100, learning_rate=lr,
                                  random_state=42).fit(X_train, y_train)
    s = score(y_val, g.predict(X_val))
    lr_rows.append({"learning_rate": lr, "val_rmse": s["RMSE"], "val_r2": s["R2"]})
pd.DataFrame(lr_rows).round(4).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# Task 7: cross-validation on the two best, then final test
ranking = sorted(val_rmse, key=val_rmse.get)
top2 = ranking[:2]       
X_trval = pd.concat([X_train, X_val])
y_trval = pd.concat([y_train, y_val])
cv = KFold(n_splits=5, shuffle=True, random_state=42)   
cv_rows = []
for name in top2:
    sc = cross_val_score(models[name], X_trval, y_trval, cv=cv)
    cv_rows.append({"model": name, "mean_r2": sc.mean(), "std_r2": sc.std()})
pd.DataFrame(cv_rows).round(4).to_csv("outputs/part_b_cv.csv", index=False)

final_name = ranking[0]
rows.append({"model": final_name, "split": "test",
             **score(y_test, models[final_name].predict(X_test))})  

metrics = pd.DataFrame(rows, columns=["model", "split", "MAE", "MSE", "RMSE", "R2"])
metrics.round(4).to_csv("outputs/part_b_metrics.csv", index=False)

print(checks)
print(pd.DataFrame(depth_rows).round(4))
print(metrics.round(4))
print(pd.DataFrame(lr_rows).round(4))
print(pd.DataFrame(cv_rows).round(4))
print("final model:", final_name)