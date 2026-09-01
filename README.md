# NovaBank: Predictive Retention at Scale

**Quantic MSBA — Analytics Methods and Frameworks Project**
August 2026

## Project Summary
Developed a predictive churn framework for NovaBank to shift from reactive to proactive customer retention. Built and compared two classification models, delivered a threshold-based decision framework, and provided a business-ready pilot plan.

## Results
| Model | AUC-ROC | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline: Logistic Regression | 0.679 | 0.301 | 0.613 | 0.404 |
| **Improved: Random Forest** | **0.704** | **0.374** | **0.491** | **0.425** |

**Recommended threshold: 0.45** — captures 63% of churners, targeting ~813 of 2,000 customers per cohort.

## Files
| File | Description |
|---|---|
| `NovaBank_Churn_Notebook.py` | Full reproducible analysis — paste into Google Colab and Run All |
| `eda_plots.png` | 6-panel EDA: age, products, geography, balance, activity, correlations |
| `model_diagnostics.png` | ROC curve comparison + Random Forest feature importance |
| `decision_framework.png` | Confusion matrix + Precision-Recall threshold trade-off |

## How to Run
1. Open [Google Colab](https://colab.research.google.com)
2. Upload `NovaBank_Churn_Notebook.py`
3. Run All cells (Runtime → Run All)
4. All outputs reproduce with `random_state=42`

## Key Findings
- **Age 45+** customers churn at ~2× the base rate
- **3–4 products held** drives 58–60% churn (vs 16–17% for 1–2 products)
- **Germany segment** churns at 27% vs 16–18% for France/Spain
- **Inactive members** show significantly elevated churn

## Business Impact (Top-20% Policy)
- Base Case (20% churn): ~$320K net annual value
- Stress Case (35% churn): ~$630K net annual value
- ROI-positive in every scenario tested

## AI Usage
Claude (Anthropic) used for code scaffolding and memo structure.
All analytical decisions, threshold choices, and business recommendations: Author's own.

---
*Dataset: Synthetic NovaBank dataset (n=10,000, 20.0% churn rate) matching Quantic project specification.*
*Models: scikit-learn 1.x · Python 3.x · random_state=42 throughout*
