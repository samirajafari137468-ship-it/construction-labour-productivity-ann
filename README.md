# Construction Labour Productivity Prediction
## LWI — PR — ANN Pipeline
### Group 8 | PCM A.Y. 2024-2025 | Supervisor: Prof. Dr. Giacomo Scrinzi

---

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

---

## Dataset

**Name:** Employee Productivity and Work Conditions Dataset  
**Source:** Kaggle   
**Size:** 14,800 workers · 23 columns · 3 skill tiers  
**Reference:** Gonzales (2025)

> Note: Download the dataset from Kaggle and place it in the 
> same folder as the scripts before running.

---

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

## Formula Sources

| Paper | Contribution |
|-------|-------------|
| Malara et al. (2019) Buildings 9(12):240 | LWI formula structure + normalisation |
| Jian et al. (2025) Buildings 15(14):2463 | RII weights from 27 studies |
| Van Tam et al. (2021) Cogent Bus & Mgmt 8:1863303 | 0.4 negative coefficient |

---

## Files in This Repository
