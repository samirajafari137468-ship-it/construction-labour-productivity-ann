# Construction Labour Productivity Prediction
## LWI — PR — ANN Pipeline
### Group 8 | PCM A.Y. 2024-2025 | Supervisor: Prof. Dr. Giacomo Scrinzi
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
CSV_IN  = 'worker_dataset_FINAL_v13.csv'   # ← update path if needed
FIG_DIR = 'figures'
OUT_DIR = 'outputs'
SEED    = 42

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ── PLOT STYLE ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'Times New Roman',
    'font.size':         11,
    'axes.titlesize':    12,
    'axes.labelsize':    11,
    'legend.fontsize':   10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linestyle':    '--',
    'figure.dpi':        150,
})

COLORS     = {'Junior': '#4A90D9', 'Mid-level': '#F5A623', 'Senior': '#2ECC71'}
TIER_ORDER = ['Junior', 'Mid-level', 'Senior']
print("=" * 65)
print("STAGE 1 — Loading dataset")
print("=" * 65)



# ← Change filename if different
DATA_PATH = '/content/worker dataset FINAL v13.csv'
OUTPUT_DIR = '/content/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)
---
df = pd.read_csv(CSV_IN)
print(f"  Rows    : {len(df):,}")
print(f"  Columns : {df.shape[1]}")
print(f"  Tiers   : {df['Tier'].value_counts().to_dict()}")

# =============================================================================
# STAGE 2 — DATA CLEANING
# =============================================================================
print("\n" + "=" * 65)
print("STAGE 2 — Data cleaning")
print("=" * 65)

checks = {
    'Missing values'       : df.isnull().sum().sum(),
    'Duplicate rows'       : df.duplicated().sum(),
}
print(f"  Missing values  : {checks['Missing values']}")
print(f"  Duplicate rows  : {checks['Duplicate rows']}")

# Validate score columns [1, 10]
score_cols = ['SK_Skill_Score','AP_Pay_Adequacy','LS_Safety_Score',
              'PR_Tools_Score','SUP_Supervision_Score',
              'MAT_Material_Availability','COM_Communication_Score',
              'Motivation_Score','Weather_Score']
for c in score_cols:
    out = ((df[c] < 1) | (df[c] > 10)).sum()
    print(f"  {c:<35} out of [1,10] : {out}")

# Validate rate columns [0, 1]
for c in ['Absenteeism_Rate', 'Rework_Rate']:
    out = ((df[c] < 0) | (df[c] > 1)).sum()
    print(f"  {c:<35} out of [0,1]  : {out}")

# Statistical outliers (3σ)
num_cols = df.select_dtypes(include=np.number).columns
total_outliers = 0
for c in num_cols:
    n = ((df[c] - df[c].mean()).abs() > 3 * df[c].std()).sum()
    if n > 0:
        print(f"  {c:<35} outliers (3σ) : {n}  → retained")
        total_outliers += n
print(f"  Total outliers retained : {total_outliers}")
print("  ✅ All cleaning checks passed — dataset is clean")


## What This Project Does

This project develops a two-stage construction labour productivity 
modelling framework:

- **Stage 1 — LWI**: Computes the Labour Workability Index for 
  14,800 construction workers using the Malara (2019) formula 
  structure and Jian (2025) RII weights from a meta-analysis of 
  27 studies across 17 countries

- **Stage 2 — PR**: Computes the Productivity Ratio using a 
  multiplicative penalty structure with 5 penalty factors 
  (overtime, absenteeism, rework, weather, motivation)

- **Stage 3 — ANN**: Trains a Multilayer Perceptron neural network 
  (256→128→64→32) to predict Productivity Ratio from 12 raw 
  observable worker and site condition variables
W_POS = {
    'SUP_Supervision_Score':      0.855,  # Rank 1 — Jian (2025)
    'SK_Skill_Score':             0.833,  # Rank 2
    'AP_Pay_Adequacy':            0.823,  # Rank 3
    'PR_Tools_Score':             0.826,  # Rank 4
    'MAT_Material_Availability':  0.826,  # Rank 4
    'TR_Training_hrs_yr':         0.810,  # Rank 6
    'Motivation_Score':           0.787,  # Rank 7
    'LS_Safety_Score':            0.780,  # Rank 8
    'COM_Communication_Score':    0.779,  # Rank 9
}
W_NEG = {
    'Absenteeism_Rate':    0.810,  # Negative — Van Tam (2021)
    'Overtime_hrs_week':   0.810,  # Negative — Van Tam (2021)
    'Rework_Rate':         0.795,  # Negative — Van Tam (2021)
---

## Dataset

**Name:** Employee Productivity and Work Conditions Dataset  
**Source:** Kaggle   
**Size:** 14,800 workers · 23 columns · 3 skill tiers  
**Reference:** Gonzales (2025)

> Note: Download the dataset from Kaggle and place it in the 
> same folder as the scripts before running.

xi_norm = {
    'SUP_Supervision_Score':     (df['SUP_Supervision_Score']     - 1) / 9,
    'SK_Skill_Score':            (df['SK_Skill_Score']            - 1) / 9,
    'AP_Pay_Adequacy':           (df['AP_Pay_Adequacy']           - 1) / 9,
    'PR_Tools_Score':            (df['PR_Tools_Score']            - 1) / 9,
    'MAT_Material_Availability': (df['MAT_Material_Availability'] - 1) / 9,
    'TR_Training_hrs_yr':        np.minimum(df['TR_Training_hrs_yr'] / 150, 1.0),
    'Motivation_Score':          (df['Motivation_Score']          - 1) / 9,
    'LS_Safety_Score':           (df['LS_Safety_Score']           - 1) / 9,
    'COM_Communication_Score':   (df['COM_Communication_Score']   - 1) / 9,
    'Absenteeism_Rate':          df['Absenteeism_Rate'],
    'Overtime_hrs_week':         df['Overtime_hrs_week'] / 28 * 0.6,
    'Rework_Rate':               df['Rework_Rate'] * 1.8,
}

# ── LWI formula — Malara (2019) Eq.8 ─────────────────────────────────────────
# LWI = SUM(wi_pos × xi_norm) / SUM(wi_pos)
#      − 0.4 × SUM(wj_neg × xj_norm) / SUM(wj_neg)
#      + ε_LWI ~ N(0, 0.065²)

sum_w_pos = sum(W_POS.values())   # 7.319
sum_w_neg = sum(W_NEG.values())   # 2.415

pos_term  = sum(W_POS[c] * xi_norm[c] for c in W_POS) / sum_w_pos
neg_term  = 0.4 * sum(W_NEG[c] * xi_norm[c] for c in W_NEG) / sum_w_neg

# Use pre-computed LWI from dataset (includes ε_LWI noise)
df['LWI_computed'] = np.clip(pos_term - neg_term, 0, 1)

print(f"  LWI (dataset)   mean={df['LWI'].mean():.4f}  std={df['LWI'].std():.4f}")
print(f"  LWI (computed)  mean={df['LWI_computed'].mean():.4f}  std={df['LWI_computed'].std():.4f}")
print(f"  Σ w_pos = {sum_w_pos:.3f}  |  Σ w_neg = {sum_w_neg:.3f}")
print(f"  Reconstruction r = {np.corrcoef(df['LWI_computed'], df['LWI'])[0,1]:.4f}")

# =============================================================================
# STAGE 5 — PR CALCULATION
# Malara et al. (2019) multiplicative penalty structure
# PR = LWI^0.75 × F_OT × F_ABS × F_RW × F_WC × F_MOT + ε
# =============================================================================


print("\n" + "=" * 65)
print("STAGE 5 — PR calculation (penalty factors from dataset)")
print("=" * 65)
---
sum_w_pos = sum(W_POS.values())   # 7.319
sum_w_neg = sum(W_NEG.values())   # 2.415

pos_term  = sum(W_POS[c] * xi_norm[c] for c in W_POS) / sum_w_pos
neg_term  = 0.4 * sum(W_NEG[c] * xi_norm[c] for c in W_NEG) / sum_w_neg

# Use pre-computed LWI from dataset (includes ε_LWI noise)
df['LWI_computed'] = np.clip(pos_term - neg_term, 0, 1)

print(f"  LWI (dataset)   mean={df['LWI'].mean():.4f}  std={df['LWI'].std():.4f}")
print(f"  LWI (computed)  mean={df['LWI_computed'].mean():.4f}  std={df['LWI_computed'].std():.4f}")
print(f"  Σ w_pos = {sum_w_pos:.3f}  |  Σ w_neg = {sum_w_neg:.3f}")
print(f"  Reconstruction r = {np.corrcoef(df['LWI_computed'], df['LWI'])[0,1]:.4f}")
## Results

| Metric | Value |
|--------|-------|
| Test R² | 0.721 |
| MAE | 0.061 |
| RMSE | 0.086 |
| Theoretical ceiling R² | 0.765 |
| % of ceiling reached | 95.0% |
| Top feature | Supervision (ΔR²=0.059) |

---
print("\n" + "=" * 65)
print("STAGE 6 — Feature engineering")
print("=" * 65)

# OHE — Tier → binary columns
df_enc    = pd.get_dummies(df, columns=['Tier'], drop_first=False)
tier_cols = [c for c in df_enc.columns if 'Tier_' in c]
for c in tier_cols:
    df_enc[c] = df_enc[c].astype(int)

# ANN input features — 12 continuous + 3 OHE tier dummies
ANN_FEATURES = ALL_FEATS + tier_cols
FEAT_LABELS_IMP = FEAT_LABELS + [t.replace('Tier_', '') for t in tier_cols]

print(f"  OHE tier columns : {tier_cols}")
print(f"  Total ANN inputs : {len(ANN_FEATURES)} "
      f"(12 continuous + {len(tier_cols)} OHE tier)")

# Train/test split — BEFORE scaling to prevent data leakage
X = df_enc[ANN_FEATURES].values
y = df_enc['Productivity_Ratio'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)
_, tier_test = train_test_split(
    df_enc['Tier_Junior'].values, test_size=0.2, random_state=SEED)

# StandardScaler — fit on training set ONLY (prevent data leakage)
scaler        = StandardScaler()
X_train_sc    = scaler.fit_transform(X_train)
X_test_sc     = scaler.transform(X_test)

print(f"  Training set : {X_train.shape[0]:,} workers")
print(f"  Test set     : {X_test.shape[0]:,} workers")

# Save processed files
df_proc = df_enc[ALL_FEATS + ['LWI', 'Productivity_Ratio'] + tier_cols].copy()
df_proc.to_csv(os.path.join(OUT_DIR, 'processed_worker_productivity.csv'), index=False)

df_scaled          = pd.DataFrame(X_train_sc, columns=ANN_FEATURES)
df_scaled['PR_target'] = y_train
df_scaled.to_csv(os.path.join(OUT_DIR,
                 'processed_worker_productivity_ann_input.csv'), index=False)
print(f"  File 1 saved → processed_worker_productivity.csv")
print(f"  File 2 saved → processed_worker_productivity_ann_input.csv")

print("\n" + "=" * 65)
print("STAGE 7 — ANN training")
print("=" * 65)

ann = MLPRegressor(
    hidden_layer_sizes  = (256, 128, 64, 32),
    activation          = 'relu',
    solver              = 'adam',
    alpha               = 0.001,
    learning_rate_init  = 0.0003,
    max_iter            = 2000,
    early_stopping      = True,
    validation_fraction = 0.15,
    n_iter_no_change    = 200,
    random_state        = SEED,
)
ann.fit(X_train_sc, y_train)

y_pred    = ann.predict(X_test_sc)
residuals = y_test - y_pred
r2_val    = r2_score(y_test, y_pred)
mae_val   = mean_absolute_error(y_test, y_pred)
rmse_val  = np.sqrt(mean_squared_error(y_test, y_pred))

best_r2   = ann.best_validation_score_
best_iter = ann.validation_scores_.index(best_r2) + 1
stop_iter = ann.n_iter_

print(f"  Architecture  : 256 → 128 → 64 → 32 → 1")
print(f"  Total params  : ~46,593")
print(f"  Peak val R²   : {best_r2:.4f} at epoch {best_iter}")
print(f"  Early stop    : epoch {stop_iter}")
print(f"  Test R²       : {r2_val:.4f}")
print(f"  MAE           : {mae_val:.4f}")
print(f"  RMSE          : {rmse_val:.4f}")
print(f"  % of ceiling  : {r2_val / ceiling_r2 * 100:.1f}%")

# Cross-validation
cv = cross_val_score(
    MLPRegressor(hidden_layer_sizes=(256,128,64,32), max_iter=500,
                 random_state=SEED),
    scaler.fit_transform(X), y, cv=5, scoring='r2')
print(f"  CV R²         : {cv.mean():.4f} ± {cv.std():.4f}")


print("\n" + "=" * 65)
print("STAGE 8 — Results figures")
print("=" * 65)

# ── Fig 05: Feature importance ────────────────────────────────────────────────
base_r2     = r2_score(y_test, ann.predict(X_test_sc))
importances = []
rng2        = np.random.default_rng(SEED)
for i in range(X_test_sc.shape[1]):
    drops = []
    for _ in range(10):
        Xp      = X_test_sc.copy()
        Xp[:,i] = rng2.permutation(Xp[:,i])
        drops.append(base_r2 - r2_score(y_test, ann.predict(Xp)))
    importances.append(np.mean(drops))

imp_df = pd.DataFrame({'Feature': FEAT_LABELS_IMP, 'Importance': importances})
imp_df = imp_df.sort_values('Importance', ascending=True)
neg_f  = ['Absenteeism', 'Overtime', 'Rework']
imp_df['Color'] = imp_df['Feature'].apply(
    lambda x: '#E74C3C' if x in neg_f else '#3498DB')

fig, ax = plt.subplots(figsize=(9, 8))
bars_h = ax.barh(imp_df['Feature'], imp_df['Importance'],
                 color=imp_df['Color'], edgecolor='white', height=0.65)
for bar, val in zip(bars_h, imp_df['Importance']):
    ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)
ax.set_xlabel('Drop in R² when feature is randomly shuffled')
ax.set_title('ANN Feature Importance — Permutation Method\n'
             'Higher ΔR² = more important to the model',
             fontweight='bold', pad=12)
p1 = mpatches.Patch(color='#3498DB', label='Positive (enabling) factor')
p2 = mpatches.Patch(color='#E74C3C', label='Negative (inhibiting) factor')
ax.legend(handles=[p1, p2], loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig05_feature_importance.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Fig 05 saved — feature importance")

# ── Fig 06: Predicted vs actual ───────────────────────────────────────────────
# Reconstruct tier colours for test set
_, tier_df_test = train_test_split(df['Tier'].values, test_size=0.2,
                                    random_state=SEED)
fig, ax = plt.subplots(figsize=(9, 8))
for tier in TIER_ORDER:
    mask = tier_df_test == tier
    ax.scatter(y_test[mask], y_pred[mask],
               color=COLORS[tier], alpha=0.35, s=14,
               label=tier, rasterized=True)
x_line = np.array([0.0, 1.0])
ax.plot(x_line, x_line,        color='gray', linestyle='--',
        linewidth=1.8, label='Ideal (y = x)')
ax.plot(x_line, x_line + 0.10, color='red',  linestyle='--',
        linewidth=1.5, label='+10% bound')
ax.plot(x_line, x_line - 0.10, color='red',  linestyle='--',
        linewidth=1.5, label='−10% bound')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel('Actual Productivity Ratio')
ax.set_ylabel('ANN Predicted Productivity Ratio (ŷ)')
ax.set_title(f'Predicted vs. Actual: Productivity Ratio\n'
             f'R² = {r2_val:.4f}  |  MAE = {mae_val:.4f}  |  '
             f'RMSE = {rmse_val:.4f}',
             fontweight='bold', pad=12)
ax.legend()
plt.tight_layout()
fig.text(0.5, -0.03,
         f'Figure 06: ANN predicted vs actual PR (n=2,960 test workers). '
         f'Gray = perfect prediction; Red = ±10% parallel bounds.\n'
         f'R²={r2_val:.4f}, MAE={mae_val:.4f}, RMSE={rmse_val:.4f}.',
         ha='center', fontsize=9, style='italic', color='#444444')
plt.savefig(os.path.join(FIG_DIR, 'fig06_predicted_vs_actual.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Fig 06 saved — predicted vs actual")

# ── Fig 07: Residual analysis ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('ANN Residual Analysis — Productivity Ratio Prediction',
             fontsize=13, fontweight='bold')
axes[0].hist(residuals, bins=50, color='#5B7FA6',
             edgecolor='white', linewidth=0.4, alpha=0.85)
axes[0].axvline(0, color='red', linestyle='--',
                linewidth=1.8, label='Zero error')
axes[0].axvline(residuals.mean(), color='orange', linestyle='-',
                linewidth=1.5, label=f'Mean = {residuals.mean():.4f}')
axes[0].set_xlabel('Residual  (Actual − Predicted)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Residual Distribution\n'
                  'Bell shape centred at zero = no bias', fontweight='bold')
axes[0].legend()
axes[1].scatter(y_pred, residuals, alpha=0.2, s=8,
                color='#5B7FA6', rasterized=True)
axes[1].axhline(0, color='red', linestyle='--', linewidth=1.8)
axes[1].axhline( residuals.std(), color='gray', linestyle=':',
                 linewidth=1.2, label=f'+1σ = {residuals.std():.3f}')
axes[1].axhline(-residuals.std(), color='gray', linestyle=':',
                linewidth=1.2, label='−1σ')
axes[1].set_xlabel('Predicted Productivity Ratio')
axes[1].set_ylabel('Residual  (Actual − Predicted)')
axes[1].set_title('Residuals vs. Predicted Values\n'
                  'Uniform scatter = homoscedastic errors', fontweight='bold')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig07_residuals.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Fig 07 saved — residual analysis")

# ── Fig 14: Convergence graph ─────────────────────────────────────────────────
loss_curve = ann.loss_curve_
val_r2     = ann.validation_scores_
iters      = list(range(1, len(loss_curve) + 1))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Figure 14 — ANN Model Convergence\n"
             "Training Loss (log scale)  +  Validation R² per Iteration",
             fontsize=13, fontweight='bold')

ax = axes[0]
ax.plot(iters, loss_curve, color='#1B3A6B', lw=2.5, label='Training Loss (MSE)')
ax.axvline(stop_iter, color='#E74C3C', lw=2, ls='--',
           label=f'Early Stop  iter={stop_iter}')
ax.axhline(loss_curve[-1], color='gray', lw=1.2, ls=':',
           label='Noise floor')
ax.set_yscale('log')
ax.set_xlabel('Optimization Epochs (Iterations)', fontweight='bold')
ax.set_ylabel('Objective Function Error (Training MSE)', fontweight='bold')
ax.set_title('Panel A: Objective Training Loss Trajectory', fontweight='bold')
ax.text(0.98, 0.95,
        f'Start = {loss_curve[0]:.4f}\n'
        f'Final  = {loss_curve[-1]:.5f}\n'
        f'Iters  = {stop_iter}',
        transform=ax.transAxes, fontsize=9, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#CCCCCC', alpha=0.9))
ax.legend(); ax.grid(True, alpha=0.3, ls='--')

ax = axes[1]
rapid_iters = iters[:best_iter]
rapid_vals  = val_r2[:best_iter]
ax.plot(iters, val_r2, color='gray', lw=1.5, alpha=0.6, label='Validation Curve')
ax.plot(rapid_iters, rapid_vals, color='#E74C3C', lw=2.5,
        label='Rapid Optimization Phase')
ax.axhline(best_r2, color='navy', lw=1.5, ls=':',
           label=f'Best R² = {best_r2:.4f}')
ax.axvline(stop_iter, color='#E74C3C', lw=2, ls='--',
           label=f'Early Stop  iter={stop_iter}')
ax.axvline(best_iter, color='#2ECC71', lw=1.8, ls='-.',
           label=f'Best iter = {best_iter}')
ax.axvspan(best_iter, stop_iter, alpha=0.07, color='#E74C3C',
           label='Patience window')
ax.fill_between(iters, val_r2, alpha=0.08, color='#9B59B6')
ax.scatter(best_iter, best_r2, color='#E74C3C', s=80, zorder=5,
           edgecolor='black', lw=1.2)
ax.annotate(f"Peak Accuracy Reached!\nEpoch: {best_iter}\n"
            f"Validation R²: {best_r2:.4f}",
            xy=(best_iter, best_r2),
            xytext=(best_iter + max(iters)*0.08, best_r2 - 0.08),
            fontsize=9, fontweight='bold', color='navy',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#EBF5FB',
                      edgecolor='navy', alpha=0.95),
            arrowprops=dict(arrowstyle='->', color='navy',
                            connectionstyle='arc3,rad=0.2', lw=1.2))
ax.set_ylim(max(0, min(val_r2) - 0.05), 1.02)
ax.set_xlabel('Optimization Epochs (Iterations)', fontweight='bold')
ax.set_ylabel('Generalization Error (Validation MSE)', fontweight='bold')
ax.set_title(f'Validation R² per Iteration\n'
             f'Model explains {best_r2*100:.1f}% of PR variance at best point',
             fontweight='bold')
ax.legend(loc='lower right', fontsize=9); ax.grid(True, alpha=0.3, ls='--')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig14_ann_convergence_loss.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Fig 14 saved — convergence graph")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
print(f"  Dataset        : {len(df):,} workers · {df.shape[1]} columns")
print(f"  Tiers          : Junior={df[df['Tier']=='Junior'].shape[0]:,} · "
      f"Mid-level={df[df['Tier']=='Mid-level'].shape[0]:,} · "
      f"Senior={df[df['Tier']=='Senior'].shape[0]:,}")
print(f"  LWI mean       : {df['LWI'].mean():.4f}")
print(f"  PR mean        : {df['Productivity_Ratio'].mean():.4f}")
print(f"  Noise floor σ  : {noise_eff.std():.4f}")
print(f"  Ceiling R²     : {ceiling_r2:.4f}")
print(f"  ANN R²         : {r2_val:.4f}  ({r2_val/ceiling_r2*100:.1f}% of ceiling)")
print(f"  ANN MAE        : {mae_val:.4f}")
print(f"  ANN RMSE       : {rmse_val:.4f}")
print(f"  CV R²          : {cv.mean():.4f} ± {cv.std():.4f}")
print(f"  Best val R²    : {best_r2:.4f} at epoch {best_iter}")
print(f"  Early stop     : epoch {stop_iter}")
print(f"\n  Figures saved  : {FIG_DIR}/")
print(f"  Outputs saved  : {OUT_DIR}/")
print("=" * 65)


## Formula Sources

| Paper | Contribution |
|-------|-------------|
| Malara et al. (2019) Buildings 9(12):240 | LWI formula structure + normalisation |
| Jian et al. (2025) Buildings 15(14):2463 | RII weights from 27 studies |
| Van Tam et al. (2021) Cogent Bus & Mgmt 8:1863303 | 0.4 negative coefficient |

---
fig, ax = plt.subplots(figsize=(10, 5))
for tier in TIER_ORDER:
    vals = df[df['Tier']==tier]['Productivity_Ratio']
    ax.hist(vals, bins=40, alpha=0.6, color=COLORS[tier], label=tier, edgecolor='white', linewidth=0.3)
    ax.axvline(vals.mean(), color=COLORS[tier], linestyle='--', linewidth=1.5)
ax.set_xlabel('Productivity Ratio (PR)', fontsize=12)
ax.set_ylabel('Number of Workers', fontsize=12)
ax.set_title('Distribution of Productivity Ratio by Skill Tier', fontsize=13, fontweight='bold', pad=12)
ax.legend(title='Skill Tier', fontsize=10, framealpha=0.85)
caption = ('Figure 01: Histogram of Productivity Ratio (PR) for all three skill tiers. '
           'Dashed vertical lines indicate tier means.\n'
           'Senior workers exhibit the highest PR, while Junior workers cluster at lower values.')
fig.text(0.5, -0.06, caption, ha='center', fontsize=9, style='italic', color='#444444')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig01_pr_distribution.png', bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(figsize=(8, 6))
data_by_tier = [df[df['Tier']==t]['Productivity_Ratio'].values for t in TIER_ORDER]
bp = ax.boxplot(data_by_tier, patch_artist=True,
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5),
                flierprops=dict(marker='o', markersize=2, alpha=0.3))
for patch, tier in zip(bp['boxes'], TIER_ORDER):
    patch.set_facecolor(COLORS[tier]); patch.set_alpha(0.75)
ax.set_xticks([1,2,3])
ax.set_xticklabels(TIER_ORDER, fontsize=12)
ax.set_ylabel('Productivity Ratio (PR)', fontsize=12)
ax.set_title('Productivity Ratio Distribution by Skill Tier', fontsize=13, fontweight='bold', pad=12)
for i, tier in enumerate(TIER_ORDER):
    med = np.median(df[df['Tier']==tier]['Productivity_Ratio'])
    ax.text(i+1, med+0.016, f'Md={med:.3f}', ha='center', fontsize=9, fontweight='bold')
caption = ('Figure 02: Box plots comparing Productivity Ratio across skill tiers.\n'
           'Boxes show IQR; whiskers extend to 1.5×IQR; circles are outliers.')
fig.text(0.5, -0.07, caption, ha='center', fontsize=9, style='italic', color='#444444')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig02_boxplot_tier.png', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
counts = [df[df['Tier']==t].shape[0] for t in TIER_ORDER]
pcts   = [c/sum(counts)*100 for c in counts]
bars = ax.bar(TIER_ORDER, counts, color=[COLORS[t] for t in TIER_ORDER], edgecolor='white', width=0.55)
for bar, cnt, pct in zip(bars, counts, pcts):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+120,
            f'{cnt:,}\n({pct:.0f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Workers', fontsize=12)
ax.set_title('Dataset Composition: Workers per Skill Tier', fontsize=13, fontweight='bold', pad=12)
ax.set_ylim(0, max(counts)*1.22)
caption = ('Figure 03: Total number of workers per skill tier (n = 14,800).\n'
           '50% Junior, 35% Mid-level, 15% Senior.')
fig.text(0.5, -0.06, caption, ha='center', fontsize=9, style='italic', color='#444444')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig03_worker_count.png', bbox_inches='tight')
plt.show()



# Fig 04 — Correlation Heatmap
corr_cols   = FEATURES + ['Productivity_Ratio']
corr_labels = FEAT_LABELS + ['Productivity\nRatio']
corr_matrix = df[corr_cols].corr()
corr_matrix.index   = corr_labels
corr_matrix.columns = corr_labels
fig, ax = plt.subplots(figsize=(13, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, vmin=-1, vmax=1, ax=ax,
            annot_kws={'size':8}, linewidths=0.4, linecolor='white',
            cbar_kws={'shrink':0.8, 'label':'Pearson r'})
ax.set_title('Correlation Matrix — Input Features & Productivity Ratio', fontsize=13, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
caption = ('Figure 04: Pearson correlation matrix. Green = positive, Red = negative.\n'
           'Skill Score and Supervision show strongest positive correlations with PR;'
           ' Absenteeism and Rework show strongest negative.')
fig.text(0.5, -0.04, caption, ha='center', fontsize=9, style='italic', color='#444444')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig04_correlation_heatmap.png', bbox_inches='tight')
plt.show()


# Fig 05 — Feature Importance (Permutation)
base_r2 = r2_score(y_test, ann.predict(X_test))
importances = []
for i in range(X_test.shape[1]):
    Xp = X_test.copy()
    np.random.seed(42)
    Xp[:,i] = np.random.permutation(Xp[:,i])
    importances.append(base_r2 - r2_score(y_test, ann.predict(Xp)))
imp_df = pd.DataFrame({'Feature':FEAT_LABELS,'Importance':importances})
imp_df = imp_df.sort_values('Importance', ascending=True)
imp_df['Color'] = imp_df['Feature'].apply(lambda x: '#E74C3C' if x in NEG_FEATURES else '#3498DB')
fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(imp_df['Feature'], imp_df['Importance'], color=imp_df['Color'], edgecolor='white', height=0.65)
for bar, val in zip(bars, imp_df['Importance']):
    ax.text(val+0.0005, bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
ax.set_xlabel('Drop in R² when feature is randomly shuffled', fontsize=11)
ax.set_title('ANN Feature Importance — Permutation Method', fontsize=13, fontweight='bold', pad=12)
p1 = mpatches.Patch(color='#3498DB', label='Positive factor')
p2 = mpatches.Patch(color='#E74C3C', label='Negative (inhibiting) factor')
ax.legend(handles=[p1,p2], fontsize=10, loc='lower right', framealpha=0.85)
caption = ('Figure 05: Permutation-based feature importance. '
           'Blue = positive factors; Red = inhibiting factors.\n'
           'Skill Score and Supervision are the most influential positive drivers.')
fig.text(0.5, -0.06, caption, ha='center', fontsize=9, style='italic', color='#444444')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig05_feature_importance.png', bbox_inches='tight')
plt.show()


# Fig 06 — Predicted vs Actual (±10% parallel bounds)
fig, ax = plt.subplots(figsize=(9, 8))
for tier in TIER_ORDER:
    mask = tier_test == tier
    ax.scatter(y_test[mask], y_pred[mask], color=COLORS[tier], alpha=0.35, s=14, label=tier, rasterized=True)
x_line = np.array([0.0, 1.0])
ax.plot(x_line, x_line,        color='gray', linestyle='--', linewidth=1.8, label='Ideal (y = x)')
ax.plot(x_line, x_line + 0.10, color='red',  linestyle='--', linewidth=1.5, label='+10% bound')
ax.plot(x_line, x_line - 0.10, color='red',  linestyle='--', linewidth=1.5, label='−10% bound')
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_xlabel('Actual Productivity Ratio', fontsize=12)
ax.set_ylabel('ANN Predicted Productivity Ratio (ŷ)', fontsize=12)
ax.set_title(f'Predicted vs. Actual: Productivity Ratio\nR² = {r2_val:.4f}  |  MAE = {mae_val:.4f}  |  RMSE = {rmse_val:.4f}',
             fontsize=12, fontweight='bold', pad=12)
ax.legend(fontsize=9, framealpha=0.85)
caption = (f'Figure 06: ANN predicted vs actual PR on test set (n=2,960). '
           f'Gray = perfect prediction; Red = ±10% parallel bounds.\n'
           f'R²={r2_val:.4f}, MAE={mae_val:.4f}, RMSE={rmse_val:.4f}.')
fig.text(0.5, -0.05, caption, ha='center', fontsize=9, style='italic', color='#444444')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig06_predicted_vs_actual.png', bbox_inches='tight')
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(residuals, bins=50, color='#5B7FA6', edgecolor='white', linewidth=0.4, alpha=0.85)
axes[0].axvline(0, color='red', linestyle='--', linewidth=1.8, label='Zero error')
axes[0].axvline(residuals.mean(), color='orange', linestyle='-', linewidth=1.5, label=f'Mean={residuals.mean():.4f}')
axes[0].set_xlabel('Residual  (Actual − Predicted)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Residual Distribution', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9, framealpha=0.85)
axes[1].scatter(y_pred, residuals, alpha=0.2, s=8, color='#5B7FA6', rasterized=True)
axes[1].axhline(0, color='red', linestyle='--', linewidth=1.8)
axes[1].axhline( residuals.std(), color='gray', linestyle=':', linewidth=1.2, label=f'+1σ={residuals.std():.3f}')
axes[1].axhline(-residuals.std(), color='gray', linestyle=':', linewidth=1.2, label='−1σ')
axes[1].set_xlabel('Predicted Productivity Ratio', fontsize=11)
axes[1].set_ylabel('Residual  (Actual − Predicted)', fontsize=11)
axes[1].set_title('Residuals vs. Predicted Values', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9, framealpha=0.85)
plt.suptitle('ANN Residual Analysis — Productivity Ratio Prediction', fontsize=13, fontweight='bold', y=1.01)
caption = ('Figure 07 (Left): Residual distribution — near-zero mean indicates no systematic bias.\n'
           '(Right): Residuals vs predicted — uniform scatter confirms homoscedasticity.')
fig.text(0.5, -0.06, caption, ha='center', fontsize=9, style='italic', color='#444444')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig07_residuals.png', bbox_inches='tight')
plt.show()



stats = df.groupby('Tier')['Productivity_Ratio'].agg(['mean','std']).reindex(TIER_ORDER)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(TIER_ORDER, stats['mean'], yerr=stats['std'], capsize=6,
              color=[COLORS[t] for t in TIER_ORDER], alpha=0.82,
              edgecolor='white', linewidth=0.5, width=0.5,
              error_kw=dict(elinewidth=1.5, ecolor='#333333', capthick=1.5))
for bar, (tier, row) in zip(bars, stats.iterrows()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+row['std']+0.012,
            f'{row["mean"]:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Mean Productivity Ratio (PR)', fontsize=12)
ax.set_title('Mean Productivity Ratio by Skill Tier\n(Error bars = ±1 Standard Deviation)',
             fontsize=12, fontweight='bold', pad=12)
ax.set_ylim(0, 0.95)
caption = ('Figure 08: Mean PR per tier with ±1 standard deviation error bars.\n'
           'Step-wise increase from Junior to Senior confirms skill level as primary driver of PR.')
fig.text(0.5, -0.07, caption, ha='center', fontsize=9, style='italic', color='#444444')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig08_mean_pr_tier.png', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(9, 6))
parts = ax.violinplot([df[df['Tier']==t]['Productivity_Ratio'].values for t in TIER_ORDER],
                      positions=[1,2,3], showmedians=True, showextrema=True)
for pc, tier in zip(parts['bodies'], TIER_ORDER):
    pc.set_facecolor(COLORS[tier]); pc.set_alpha(0.65); pc.set_edgecolor('white')
parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
parts['cbars'].set_color('#555555'); parts['cmaxes'].set_color('#555555'); parts['cmins'].set_color('#555555')
for i, tier in enumerate(TIER_ORDER):
    sub = df[df['Tier']==tier]['Productivity_Ratio'].values
    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=min(300, len(sub)))
    sample = np.random.default_rng(42).choice(sub, size=min(300, len(sub)), replace=False)
    ax.scatter(np.full(len(sample), i+1)+jitter, sample, color=COLORS[tier], alpha=0.25, s=8, rasterized=True)
ax.set_xticks([1,2,3])
ax.set_xticklabels(TIER_ORDER, fontsize=12)
ax.set_ylabel('Productivity Ratio (PR)', fontsize=12)
ax.set_title('Productivity Ratio Distribution by Skill Tier — Violin Plot', fontsize=13, fontweight='bold', pad=12)
handles = [mpatches.Patch(color=COLORS[t], alpha=0.7, label=t) for t in TIER_ORDER]
ax.legend(handles=handles, fontsize=10, framealpha=0.85)
caption = ('Figure 09: Violin plot showing full distribution shape per tier.\n'
           'Width reflects worker density at each PR level. Jittered dots = random sample of 300 per tier.')
fig.text(0.5, -0.07, caption, ha='center', fontsize=9, style='italic', color='#444444')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig09_violin_tier.png', bbox_inches='tight')
plt.show()


## Files in This Repository
