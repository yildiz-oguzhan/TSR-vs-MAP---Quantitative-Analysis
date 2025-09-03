# TSR vs. MAP — Quantitative Analysis

This repository contains the Python scripts used for statistical analyses in the dissertation:

**Title of Dissertation**  
Author: [Your Full Name]  
University of Michigan-Flint, 2025  

The analyses explore the relationship between **Teacher-Student Relationships (TSR)** and **student academic growth** measured by **NWEA MAP scores** (Math & Reading) across five Midwest charter schools, pre- and post-COVID.

---

## 📂 Repository Contents

- `correlation_analysis.py`  
  Code for calculating Pearson correlations between TSR survey results and NWEA MAP growth scores.

- `simple_regression_analysis.py`  
  Code for running simple linear regression models with TSR as the predictor and student growth as the outcome.

- `multiple_regression_analysis.py`  
  Code for running multiple regression models including TSR and additional covariates.

- `t_test_analysis.py`  
  Code for conducting independent samples t-tests comparing pre- and post-COVID student growth.

- `LICENSE`  
  MIT license for open use of this code.

---

## ⚙️ Requirements

- Python 3.10+  
- Packages: `pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib`

Install dependencies with:

```bash
pip install pandas numpy scipy statsmodels matplotlib
