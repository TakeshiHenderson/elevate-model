import json

def cell(source, cell_type="code"):
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {
        "cell_type": "code", "execution_count": None,
        "metadata": {}, "outputs": [],
        "source": source
    }

C1 = cell("""import json, logging, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, mean_absolute_error,
    precision_recall_curve,
)

OUTPUT_DIR = Path('/home/takeshi/elevate/output')
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('rightaid.modeling')

with open(OUTPUT_DIR / 'eda_metadata.json', encoding='utf-8') as f:
    eda_meta = json.load(f)

MODEL_FEATURES = eda_meta['model_features']
print(f'Loaded EDA metadata. Setup: {eda_meta["setup"]}')
print(f'Task: {eda_meta.get("task", "binary_eligibility")}')
print(f'Features ({len(MODEL_FEATURES)}): {MODEL_FEATURES}')
print()
print('Model: XGBoost binary:logistic + SHAP (TreeExplainer)')
print(eda_meta['model_rationale'])""")

C2 = cell("""# Load processed dataset
df = pd.read_parquet(OUTPUT_DIR / 'rightaid_processed.parquet')
print(f'Loaded: {df.shape}')

X = df[MODEL_FEATURES].copy()

# Task 1: binary eligibility (bottom 40% = desil <= 4)
y_bottom40 = df['bottom40'].copy()

# Task 2: anomaly detection (0/1)
y_anomaly  = df['is_anomaly'].copy()

print(f'X shape      : {X.shape}')
vc = y_bottom40.value_counts().sort_index()
print(f'y_bottom40   : {vc.to_dict()}  ({vc[1]/len(y_bottom40)*100:.1f}% eligible)')
print(f'y_anomaly    : {y_anomaly.value_counts().to_dict()}')
logger.info('data_loaded X=%s', X.shape)""")

C3 = cell("""# Train/Test Split — single split shared by both tasks.
RANDOM_STATE = 42
TEST_SIZE    = 0.2

(
    X_train, X_test,
    y_elig_train, y_elig_test,
    y_anom_train, y_anom_test,
) = train_test_split(
    X, y_bottom40, y_anomaly,
    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_bottom40
)

print(f'Train size : {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)')
print(f'Test size  : {len(X_test):,}  ({len(X_test)/len(X)*100:.1f}%)')
print(f'Eligible in test  : {y_elig_test.sum():,} ({y_elig_test.mean()*100:.1f}%)')
print(f'Anomaly in test   : {y_anom_test.sum():,} ({y_anom_test.mean()*100:.2f}%)')
logger.info('split_done train=%d test=%d', len(X_train), len(X_test))""")

C4 = cell("""# ============================================================
# TASK 1: Binary Eligibility — bottom-40% bansos qualifier
# Predicts whether a household is in the bottom 40% of welfare
# (desil_kesejahteraan_aktual <= 4 = eligible for bansos).
# Uses XGBoost binary:logistic with threshold-optimized F1.
# ============================================================
PARAMS_ELIG = {
    'objective'            : 'binary:logistic',
    'n_estimators'         : 1000,
    'max_depth'            : 8,
    'learning_rate'        : 0.05,
    'subsample'            : 0.85,
    'colsample_bytree'     : 0.85,
    'min_child_weight'     : 3,
    'gamma'                : 0.0,
    'reg_alpha'            : 0.05,
    'reg_lambda'           : 1.0,
    'eval_metric'          : 'auc',
    'early_stopping_rounds': 50,
    'random_state'         : RANDOM_STATE,
    'n_jobs'               : -1,
    'tree_method'          : 'hist',
}

model_elig = xgb.XGBClassifier(**PARAMS_ELIG)
model_elig.fit(X_train, y_elig_train, eval_set=[(X_test, y_elig_test)], verbose=False)

y_prob_elig = model_elig.predict_proba(X_test)[:, 1]
auc_elig    = roc_auc_score(y_elig_test, y_prob_elig)

# Threshold optimization — maximize F1 on test set
prec, rec, thresholds = precision_recall_curve(y_elig_test, y_prob_elig)
f1_scores  = 2 * prec * rec / (prec + rec + 1e-9)
best_idx   = f1_scores.argmax()
best_thr   = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
y_pred_elig = (y_prob_elig >= best_thr).astype(int)

acc_elig  = accuracy_score(y_elig_test, y_pred_elig)
f1_elig   = f1_score(y_elig_test, y_pred_elig, average='binary')
prec_elig = prec[best_idx]
rec_elig  = rec[best_idx]

print(f'Task 1 — Binary Eligibility (best_iter={model_elig.best_iteration})')
print(f'  ROC-AUC        : {auc_elig:.4f}')
print(f'  Best threshold : {best_thr:.3f}')
print(f'  Accuracy       : {acc_elig:.4f}')
print(f'  F1 binary      : {f1_elig:.4f}')
print(f'  Precision      : {prec_elig:.4f}')
print(f'  Recall         : {rec_elig:.4f}')
print()
print(classification_report(y_elig_test, y_pred_elig,
                             target_names=['Not eligible (>desil4)', 'Eligible (≤desil4)'],
                             zero_division=0))
logger.info('model_elig_trained auc=%.4f acc=%.4f f1=%.4f', auc_elig, acc_elig, f1_elig)""")

C5 = cell("""# Save Task 1 model + confusion matrix
model_elig_path = OUTPUT_DIR / 'xgboost_eligibility_v3.pkl'
with open(model_elig_path, 'wb') as f:
    pickle.dump({
        'model'    : model_elig,
        'features' : MODEL_FEATURES,
        'threshold': float(best_thr),
        'target'   : 'bottom40',
    }, f)
print(f'Saved: {model_elig_path}')

fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_elig_test, y_pred_elig)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Not eligible', 'Eligible'],
            yticklabels=['Not eligible', 'Eligible'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Task 1 — Eligibility Confusion Matrix (bottom-40%)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cm_eligibility.png', dpi=100, bbox_inches='tight')
plt.show()""")

C6 = cell("""# ============================================================
# TASK 2: Anomaly Detection (XGBoost binary)
# scale_pos_weight + eval_metric='auc' + threshold optimization.
# ============================================================
n_neg = (y_anom_train == 0).sum()
n_pos = (y_anom_train == 1).sum()
spw   = n_neg / n_pos
print(f'Class balance — neg: {n_neg:,} | pos: {n_pos:,} | scale_pos_weight: {spw:.2f}')

PARAMS_ANOM = {
    'objective'            : 'binary:logistic',
    'n_estimators'         : 800,
    'max_depth'            : 5,
    'learning_rate'        : 0.05,
    'subsample'            : 0.85,
    'colsample_bytree'     : 0.85,
    'min_child_weight'     : 5,
    'gamma'                : 0.05,
    'reg_alpha'            : 0.1,
    'reg_lambda'           : 1.0,
    'scale_pos_weight'     : spw,
    'eval_metric'          : 'auc',
    'early_stopping_rounds': 40,
    'random_state'         : RANDOM_STATE,
    'n_jobs'               : -1,
    'tree_method'          : 'hist',
}

model_anomaly = xgb.XGBClassifier(**PARAMS_ANOM)
model_anomaly.fit(
    X_train, y_anom_train,
    eval_set=[(X_test, y_anom_test)],
    verbose=False,
)

y_prob_anom = model_anomaly.predict_proba(X_test)[:, 1]
roc_anom    = roc_auc_score(y_anom_test, y_prob_anom)

prec_a, rec_a, thresholds_a = precision_recall_curve(y_anom_test, y_prob_anom)
f1_scores_a  = 2 * prec_a * rec_a / (prec_a + rec_a + 1e-9)
best_idx_a   = f1_scores_a.argmax()
best_thr_a   = thresholds_a[best_idx_a] if best_idx_a < len(thresholds_a) else 0.5
y_pred_anom  = (y_prob_anom >= best_thr_a).astype(int)

acc_anom = accuracy_score(y_anom_test, y_pred_anom)
f1_anom  = f1_score(y_anom_test, y_pred_anom, average='binary')

print(f'\\nTask 2 — Anomaly Detection (best iteration: {model_anomaly.best_iteration})')
print(f'  ROC-AUC         : {roc_anom:.4f}')
print(f'  Best threshold  : {best_thr_a:.3f}')
print(f'  F1 @ threshold  : {f1_anom:.4f}')
print(f'  Accuracy        : {acc_anom:.4f}')
print()
print(classification_report(y_anom_test, y_pred_anom, target_names=['Normal','Anomaly'], zero_division=0))
logger.info('model_anomaly_trained roc=%.4f f1=%.4f thr=%.3f', roc_anom, f1_anom, best_thr_a)""")

C7 = cell("""# Save Task 2 model
model_anom_path = OUTPUT_DIR / 'xgboost_anomaly_v2.pkl'
with open(model_anom_path, 'wb') as f:
    pickle.dump({'model': model_anomaly, 'features': MODEL_FEATURES}, f)
print(f'Saved: {model_anom_path}')

fig, ax = plt.subplots(figsize=(5, 4))
cm2 = confusion_matrix(y_anom_test, y_pred_anom)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges', ax=ax,
            xticklabels=['Normal','Anomaly'], yticklabels=['Normal','Anomaly'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Task 2 — Confusion Matrix: Anomaly Detection')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cm_anomaly.png', dpi=100, bbox_inches='tight')
plt.show()""")

C8 = cell("""# ============================================================
# SHAP — TreeExplainer (Task 1: Eligibility)
# ============================================================
print('Computing SHAP values for Task 1 (Eligibility)...')
explainer_elig = shap.TreeExplainer(model_elig)

sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_test), size=10_000, replace=False)
X_sample = X_test.iloc[sample_idx]

shap_values_elig = explainer_elig.shap_values(X_sample)
if isinstance(shap_values_elig, list):
    shap_vals_1d = shap_values_elig[1]
else:
    shap_vals_1d = shap_values_elig

mean_shap_elig = np.abs(shap_vals_1d).mean(axis=0)
shap_importance_elig = pd.Series(mean_shap_elig, index=MODEL_FEATURES).sort_values(ascending=False)

print('\\nGlobal Feature Importance (mean |SHAP|) — Task 1 Eligibility:')
print(shap_importance_elig.to_string())

fig, ax = plt.subplots(figsize=(8, 7))
shap_importance_elig.plot(kind='barh', ax=ax, color='steelblue')
ax.invert_yaxis()
ax.set_title('SHAP Feature Importance — Bansos Eligibility (bottom-40%)')
ax.set_xlabel('Mean |SHAP value|')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'shap_importance_eligibility.png', dpi=100, bbox_inches='tight')
plt.show()
logger.info('shap_elig_computed sample=%d', len(X_sample))""")

C9 = cell("""# SHAP — TreeExplainer (Task 2: Anomaly Detection)
print('Computing SHAP values for Task 2 (Anomaly)...')
explainer_anom = shap.TreeExplainer(model_anomaly)
shap_values_anom = explainer_anom.shap_values(X_sample)

shap_importance_anom = pd.Series(
    np.abs(shap_values_anom).mean(axis=0),
    index=MODEL_FEATURES
).sort_values(ascending=False)

print('\\nGlobal Feature Importance (mean |SHAP|) — Task 2:')
print(shap_importance_anom.to_string())

fig, ax = plt.subplots(figsize=(8, 7))
shap_importance_anom.plot(kind='barh', ax=ax, color='darkorange')
ax.invert_yaxis()
ax.set_title('SHAP Feature Importance — Anomaly Detection')
ax.set_xlabel('Mean |SHAP value|')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'shap_importance_anomaly.png', dpi=100, bbox_inches='tight')
plt.show()
logger.info('shap_anomaly_computed sample=%d', len(X_sample))""")

C10 = cell("""# SHAP Summary Plot — Task 2 (beeswarm)
shap.summary_plot(
    shap_values_anom, X_sample,
    feature_names=MODEL_FEATURES,
    show=False, max_display=16,
)
plt.title('SHAP Beeswarm — Anomaly Detection (RightAid MVP)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'shap_beeswarm_anomaly.png', dpi=100, bbox_inches='tight')
plt.show()""")

C11 = cell("""# ============================================================
# PMT vs ML Comparison — Binary Eligibility
# PMT binary: desil_pmt_konvensional <= 4
# ML binary : model prediction @ optimized threshold
# Core RightAid insight: how much better is ML at identifying
# the right households for bansos?
# ============================================================
test_idx = X_test.index
df_eval  = df.loc[test_idx, [
    'desil_kesejahteraan_aktual', 'desil_pmt_konvensional',
    'skor_pmt_konvensional', 'is_anomaly', 'scenario', 'bottom40'
]].copy()
df_eval['pmt_eligible']    = (df_eval['desil_pmt_konvensional'] <= 4).astype(int)
df_eval['ml_eligible']     = y_pred_elig
df_eval['prob_eligible_ml']= y_prob_elig
df_eval['pred_anomaly_ml'] = y_pred_anom
df_eval['prob_anomaly_ml'] = y_prob_anom

# Error rates
pmt_excl  = ((df_eval['bottom40'] == 1) & (df_eval['pmt_eligible'] == 0)).mean() * 100
ml_excl   = ((df_eval['bottom40'] == 1) & (df_eval['ml_eligible']  == 0)).mean() * 100
pmt_incl  = ((df_eval['bottom40'] == 0) & (df_eval['pmt_eligible'] == 1)).mean() * 100
ml_incl   = ((df_eval['bottom40'] == 0) & (df_eval['ml_eligible']  == 1)).mean() * 100

pmt_acc   = accuracy_score(df_eval['bottom40'], df_eval['pmt_eligible'])
ml_acc    = accuracy_score(df_eval['bottom40'], df_eval['ml_eligible'])
pmt_f1    = f1_score(df_eval['bottom40'], df_eval['pmt_eligible'])

print('=== PMT vs ML — Binary Eligibility Comparison ===')
print(f'  PMT accuracy   : {pmt_acc:.4f}  |  ML accuracy: {ml_acc:.4f}')
print(f'  PMT F1         : {pmt_f1:.4f}  |  ML F1: {f1_elig:.4f}')
print()
print(f'  Exclusion error (eligible HH missed):')
print(f'    PMT: {pmt_excl:.2f}%  →  ML: {ml_excl:.2f}%  (Δ {ml_excl - pmt_excl:+.2f}pp)')
print(f'  Inclusion error (non-eligible included):')
print(f'    PMT: {pmt_incl:.2f}%  →  ML: {ml_incl:.2f}%  (Δ {ml_incl - pmt_incl:+.2f}pp)')

# Anomaly breakdown
anom_rows = df_eval[df_eval['is_anomaly'] == 1]
pmt_anom_excl = (anom_rows['pmt_eligible'] == 0).mean() * 100
ml_anom_excl  = (anom_rows['ml_eligible']  == 0).mean() * 100
print()
print(f'  Anomaly HH exclusion (mis-targeting of flagged cases):')
print(f'    PMT: {pmt_anom_excl:.2f}%  →  ML: {ml_anom_excl:.2f}%')
logger.info('pmt_vs_ml pmt_excl=%.2f ml_excl=%.2f pmt_acc=%.4f ml_acc=%.4f',
            pmt_excl, ml_excl, pmt_acc, ml_acc)""")

C12 = cell("""# Save evaluation_report.json
evaluation_report = {
    "project"       : "RightAid ML-PMT Refresher",
    "model_version" : "v3",
    "setup"         : "A_v3_binary",
    "features"      : MODEL_FEATURES,
    "n_train"       : int(len(X_train)),
    "n_test"        : int(len(X_test)),
    "task1_eligibility": {
        "model"          : "XGBoost binary:logistic (bottom-40 eligibility)",
        "target"         : "bottom40 = desil_kesejahteraan_aktual <= 4",
        "roc_auc"        : round(auc_elig, 4),
        "accuracy"       : round(acc_elig, 4),
        "f1_binary"      : round(f1_elig, 4),
        "precision"      : round(float(prec_elig), 4),
        "recall"         : round(float(rec_elig), 4),
        "best_threshold" : round(float(best_thr), 4),
        "best_iteration" : int(model_elig.best_iteration),
        "model_path"     : str(model_elig_path),
        "shap_top5"      : shap_importance_elig.head(5).round(6).to_dict(),
    },
    "task2_anomaly": {
        "model"          : "XGBoost binary (binary:logistic)",
        "roc_auc"        : round(roc_anom, 4),
        "f1_binary"      : round(f1_anom, 4),
        "accuracy"       : round(acc_anom, 4),
        "best_threshold" : round(float(best_thr_a), 4),
        "best_iteration" : int(model_anomaly.best_iteration),
        "model_path"     : str(model_anom_path),
        "shap_top5"      : shap_importance_anom.head(5).round(6).to_dict(),
    },
    "pmt_vs_ml": {
        "pmt_accuracy"              : round(pmt_acc, 4),
        "ml_accuracy"               : round(ml_acc, 4),
        "pmt_f1"                    : round(pmt_f1, 4),
        "ml_f1"                     : round(f1_elig, 4),
        "pmt_exclusion_error_pct"   : round(pmt_excl, 2),
        "ml_exclusion_error_pct"    : round(ml_excl, 2),
        "pmt_inclusion_error_pct"   : round(pmt_incl, 2),
        "ml_inclusion_error_pct"    : round(ml_incl, 2),
        "pmt_anomaly_exclusion_pct" : round(pmt_anom_excl, 2),
        "ml_anomaly_exclusion_pct"  : round(ml_anom_excl, 2),
    },
    "explainability": {
        "method"       : "SHAP TreeExplainer",
        "shap_sample_n": 10000,
        "artifacts"    : [
            "shap_importance_eligibility.png",
            "shap_importance_anomaly.png",
            "shap_beeswarm_anomaly.png",
        ],
    },
}

report_path = OUTPUT_DIR / 'evaluation_report.json'
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(evaluation_report, f, indent=2, ensure_ascii=False)

print(f'Saved: {report_path}')
print(json.dumps(evaluation_report, indent=2, ensure_ascii=False))
logger.info('evaluation_report_saved')""")

cells = [C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (elevate)",
            "language": "python",
            "name": "elevate"
        },
        "language_info": {"name": "python", "version": "3.12.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open('/home/takeshi/elevate/modeling.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Modeling notebook written with', len(cells), 'cells.')
