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
