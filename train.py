"""Train, compare and save the hotel cancellation models.

Run:  python train.py
Needs data/hotel_reservations.csv (see README).
"""
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", message="Precision is ill-defined")  # baseline never predicts Canceled

SEED = 42
FINAL_CV = 5
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

# 1. Load and clean ------------------------------------------------------------------
df = pd.read_csv("data/hotel_reservations.csv")
rows_in_file = len(df)
df = df.drop_duplicates()   # BEFORE dropping Booking_ID
df = df.drop(columns=["Booking_ID", "arrival_year", "arrival_date"])
df = df[df["no_of_adults"] >= 1]
df = df[(df["no_of_weekend_nights"] + df["no_of_week_nights"]) >= 1]
log(f"Rows in file: {rows_in_file} | Rows after cleaning: {len(df)}")
if rows_in_file <= 5000:
    print("WARNING: 5,000 rows or fewer. This does NOT meet the project requirement.")

# 2. Target and features -------------------------------------------------------------
y = (df["booking_status"] == "Canceled").astype(int)   # 1 = Canceled
X = df.drop(columns="booking_status")
num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat = [c for c in X.columns if c not in num]

# 3. Split (stratified keeps the cancel rate equal in both parts) ---------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
log(f"Train rows: {len(X_train)} | Test rows: {len(X_test)} | Canceled share: {y.mean():.1%}")


# 4. Preprocessing + model in ONE pipeline --------------------------------------------
def make_model(classifier):
    prep = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
    return Pipeline([("prep", prep), ("clf", classifier)])


candidates = {
    "Baseline (always Not Canceled)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
    "Random Forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=3,
                                            class_weight="balanced", random_state=SEED, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
}

# 5. Cross-validation on the training set (the test set is not touched) ---------------
skf = StratifiedKFold(n_splits=FINAL_CV, shuffle=True, random_state=SEED)
cv_rows = []
for name, clf in candidates.items():
    scores = cross_validate(make_model(clone(clf)), X_train, y_train, cv=skf, n_jobs=1,
                            scoring={"f1": "f1", "recall": "recall", "precision": "precision",
                                     "roc_auc": "roc_auc"})
    row = {"Model": name}
    for key in ["f1", "recall", "precision", "roc_auc"]:
        v = scores[f"test_{key}"]
        row[f"CV {key.upper().replace('_', '-')}"] = f"{v.mean():.3f} +/- {v.std():.3f}"
        row[f"_{key}_mean"] = v.mean()
    cv_rows.append(row)
    log(f"CV {name}: F1 {row['_f1_mean']:.4f}")
cv_table = pd.DataFrame(cv_rows)
cv_table.drop(columns=[c for c in cv_table.columns if c.startswith("_")]).to_csv(
    "results/cv_results.csv", index=False)

# 6. Final fit on the training set, evaluate on the test set ---------------------------
results, saved, fitted, proba_by_model = [], [], {}, {}
for name, clf in candidates.items():
    model = make_model(clone(clf))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "PR-AUC": average_precision_score(y_test, proba)})
    fitted[name], proba_by_model[name] = model, proba
    log(f"Trained {name}")
    if name.startswith("Baseline"):
        continue                      # baseline is only for comparison

    slug = name.lower().replace(" ", "_")
    joblib.dump(model, f"models/{slug}.joblib", compress=3)
    saved.append({"label": name, "file": f"models/{slug}.joblib", "slug": slug})
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                           display_labels=["Not Canceled", "Canceled"]).plot(
        ax=ax, colorbar=False, cmap="Purples")
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(f"results/confusion_{slug}.png", dpi=120)
    plt.close(fig)

table = pd.DataFrame(results).round(4)
table.to_csv("results/model_comparison.csv", index=False)
print("\n", table.to_string(index=False))

# Best model = highest cross-validated F1 (chosen without looking at the test set)
real = cv_table[~cv_table["Model"].str.startswith("Baseline")]
best = real.sort_values("_f1_mean").iloc[-1]["Model"]
log(f"Best model by CV F1: {best}")

# 7. ROC and precision-recall curves --------------------------------------------------
colors = {"Logistic Regression": "#1f77b4", "KNN": "#2ca02c",
          "Random Forest": "#6a1b9a", "Gradient Boosting": "#ff7f0e"}
fig, ax = plt.subplots(figsize=(5, 4.5))
for name, p in proba_by_model.items():
    if name.startswith("Baseline"):
        continue
    fpr, tpr, _ = roc_curve(y_test, p)
    ax.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_test, p):.3f})", color=colors[name])
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curves")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig("results/roc_curves.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 4.5))
for name, p in proba_by_model.items():
    if name.startswith("Baseline"):
        continue
    prec, rec, _ = precision_recall_curve(y_test, p)
    ax.plot(rec, prec, label=f"{name} ({average_precision_score(y_test, p):.3f})", color=colors[name])
ax.axhline(y_test.mean(), color="k", ls="--", lw=0.8)
ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout(); fig.savefig("results/pr_curves.png", dpi=120); plt.close(fig)

# 8. Feature importance (Random Forest built-in) ---------------------------------------
rf = fitted["Random Forest"]
names = [n.split("__", 1)[1] for n in rf.named_steps["prep"].get_feature_names_out()]
imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names)
imp.sort_values(ascending=False).head(12).rename("importance").to_csv(
    "results/feature_importance.csv", index_label="feature")

# 9. Simple data insights for the app (cancel rate by group) ---------------------------
d = df.assign(canceled=y)


def rate(by):
    g = d.groupby(by, observed=True)["canceled"].agg(["size", "mean"])
    g.columns = ["bookings", "cancel_rate"]
    g = g.reset_index().rename(columns={by: "group"})
    g["group"] = g["group"].astype(str)
    return g.round(4)


rate("market_segment_type").to_csv("results/insights_segment.csv", index=False)
rate("arrival_month").to_csv("results/insights_month.csv", index=False)
d["lead_bucket"] = pd.cut(d["lead_time"], [-1, 7, 30, 90, 180, 1000],
                          labels=["0-7 days", "8-30", "31-90", "91-180", "181+"])
rate("lead_bucket").to_csv("results/insights_lead_time.csv", index=False)
d["requests_bucket"] = d["no_of_special_requests"].clip(upper=3).map(
    {0: "0", 1: "1", 2: "2", 3: "3+"})
rate("requests_bucket").to_csv("results/insights_requests.csv", index=False)

# 10. Save what the app needs ----------------------------------------------------------
reference = {c: float(X[c].median()) for c in num}
reference.update({c: X[c].mode().iloc[0] for c in cat})
info = {"best_model": best, "models": saved, "columns": list(X.columns),
        "numeric": num, "categorical": cat,
        "categories": {c: sorted(X[c].astype(str).unique().tolist()) for c in cat},
        "reference": reference,
        "rows_used": len(df), "canceled_share": round(float(y.mean()), 4)}
Path("models/info.json").write_text(json.dumps(info, indent=2, default=str))
log("Saved models/, results/ and models/info.json")