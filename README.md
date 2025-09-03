This repository contains the Python scripts used for statistical analyses in the study:

**Title of Dissertation**  
Author: Oguzhan Yildiz 
University of Michigan-Flint, 2025  

The analyses explore the relationship between **Teacher-Student Relationships (TSR)** and **student academic growth** measured by **NWEA MAP scores** (Math & Reading) across five Midwest charter schools, pre- and post-COVID.

---

## ⚙️ Requirements

- Python 3.10+  
- Packages: `pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib`

Install dependencies with:

```bash
pip install pandas numpy scipy statsmodels matplotlib

Correlation Analysis

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load your dataset (adjust the path as needed)
df = pd.read_csv("your_data.csv")

# Initialize a dictionary to store correlations
results = []

# Get unique schools and subjects
schools = df['School'].unique()
subjects = ['Math', 'Reading']
tsr_items = ['TSR_Overall', 'TSR_Q1', 'TSR_Q2', 'TSR_Q3', 'TSR_Q4']

# Loop through each school and subject
for school in schools:
    for subject in subjects:
        school_data = df[(df['School'] == school) & (df['Subject'] == subject)]
        for item in tsr_items:
            if item in school_data.columns:
                # Drop NA values
                data = school_data[[item, 'RIT_Score']].dropna()
                if not data.empty:
                    r, p = pearsonr(data[item], data['RIT_Score'])
                    results.append({
                        'School': school,
                        'Subject': subject,
                        'TSR_Item': item,
                        'Correlation': r,
                        'p_value': p
                    })

# Convert results to DataFrame
cor_df = pd.DataFrame(results)

# Optional: Save to CSV
cor_df.to_csv("tsr_rit_correlations.csv", index=False)

# Pivot for heatmap visualization
heatmap_data = cor_df.pivot_table(
    index=['School'],
    columns=['TSR_Item', 'Subject'],
    values='Correlation'
)

# Plot heatmap
plt.figure(figsize=(16, 8))
sns.heatmap(heatmap_data, annot=True, cmap='RdBu_r', center=0, linewidths=.5)
plt.title("Correlation between TSR Items and RIT Scores")
plt.tight_layout()
plt.show()

# =======================================================
# Paired T-Test Analysis (Fall 2019 vs. Pandemic)
# =======================================================

import pandas as pd
from scipy.stats import ttest_rel

# Load your dataset
df = pd.read_csv("your_data.csv")

# Make sure your dataset has these columns:
# 'Student_ID', 'Term', 'TSR_Rating', 'Math_RIT', 'Reading_RIT'

# Label terms
df['Period'] = df['Term'].apply(lambda x: 'Fall2019' if x == 'Fall 2019' else (
    'Pandemic' if x in ['Winter 2020', 'Spring 2020', 'Fall 2020', 'Winter 2021', 'Spring 2021'] else 'Other'))

# Filter to Fall 2019 and Pandemic periods
filtered_df = df[df['Period'].isin(['Fall2019', 'Pandemic'])]

# Group by student and period to get means
grouped = filtered_df.groupby(['Student_ID', 'Period']).agg({
    'TSR_Rating': 'mean',
    'Math_RIT': 'mean',
    'Reading_RIT': 'mean'
}).reset_index()

# Pivot so each student has two columns: one for Fall2019 and one for Pandemic
pivot = grouped.pivot(index='Student_ID', columns='Period')

# Drop any rows with missing values in any comparison
pivot = pivot.dropna()

# Extract paired samples
tsr_fall = pivot['TSR_Rating']['Fall2019']
tsr_pandemic = pivot['TSR_Rating']['Pandemic']

math_fall = pivot['Math_RIT']['Fall2019']
math_pandemic = pivot['Math_RIT']['Pandemic']

reading_fall = pivot['Reading_RIT']['Fall2019']
reading_pandemic = pivot['Reading_RIT']['Pandemic']

# Run paired t-tests
tsr_t, tsr_p = ttest_rel(tsr_fall, tsr_pandemic)
math_t, math_p = ttest_rel(math_fall, math_pandemic)
reading_t, reading_p = ttest_rel(reading_fall, reading_pandemic)

# =======================================================
# Simple Regression Analysis
# =======================================================

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load your dataset
df = pd.read_csv("your_data.csv")

# Optional: Drop missing values for relevant columns
df = df.dropna(subset=[
    'TSR_Rating', 'Math_RIT', 'Reading_RIT',
    'TSR_Q1', 'TSR_Q2', 'TSR_Q3', 'TSR_Q4'
])

# Function to run and summarize regression
def run_regression(dep_var, indep_var):
    model = ols(f"{dep_var} ~ {indep_var}", data=df).fit()
    print(f"Regression: {dep_var} ~ {indep_var}")
    print(model.summary())
    print("\n")

# ▶️ Regression: Math RIT ~ TSR Rating Average
run_regression('Math_RIT', 'TSR_Rating')

# ▶️ Regression: Reading RIT ~ TSR Rating Average
run_regression('Reading_RIT', 'TSR_Rating')

# ▶️ Individual TSR Item Regressions (Math)
for q in ['TSR_Q1', 'TSR_Q2', 'TSR_Q3', 'TSR_Q4']:
    run_regression('Math_RIT', q)

# ▶️ Individual TSR Item Regressions (Reading)
for q in ['TSR_Q1', 'TSR_Q2', 'TSR_Q3', 'TSR_Q4']:
    run_regression('Reading_RIT', q)

# =======================================================
# Multiple Regression Analysis
# =======================================================

import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_excel("Final_NWEA_Panorama_Full_Fixed_Final.xlsx")

# Rename relevant columns
df = df.rename(columns={
    'TSR Rating Avg': 'TSR_Rating',
    'TSR Q1 Avg': 'TSR_Q1',
    'TSR Q2 Avg': 'TSR_Q2',
    'TSR Q3 Avg': 'TSR_Q3',
    'TSR Q4 Avg': 'TSR_Q4',
    'Mean RIT': 'Mean_RIT',
    'Subject': 'Subject'
})

# Filter for Math and Reading
math_df = df[df['Subject'] == 'Math'].dropna(subset=['TSR_Q1', 'TSR_Q2', 'TSR_Q3', 'TSR_Q4', 'Mean_RIT'])
reading_df = df[df['Subject'] == 'Reading'].dropna(subset=['TSR_Q1', 'TSR_Q2', 'TSR_Q3', 'TSR_Q4', 'Mean_RIT'])

# Standardize
scaler = StandardScaler()
math_std = pd.DataFrame(scaler.fit_transform(math_df[['TSR_Q1', 'TSR_Q2', 'TSR_Q3', 'TSR_Q4', 'Mean_RIT']]),
                        columns=['TSR_Q1', 'TSR_Q2', 'TSR_Q3', 'TSR_Q4', 'RIT'])
reading_std = pd.DataFrame(scaler.fit_transform(reading_df[['TSR_Q1', 'TSR_Q2', 'TSR_Q3', 'TSR_Q4', 'Mean_RIT']]),
                           columns=['TSR_Q1', 'TSR_Q2', 'TSR_Q3', 'TSR_Q4', 'RIT'])

# Run regression function
def run_regression(dep_var_data, predictors):
    X = dep_var_data[predictors]
    y = dep_var_data['RIT']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())

# Run models
predictors = ['TSR_Q1', 'TSR_Q2', 'TSR_Q3', 'TSR_Q4']
run_regression(math_std, predictors)
run_regression(reading_std, predictors)
