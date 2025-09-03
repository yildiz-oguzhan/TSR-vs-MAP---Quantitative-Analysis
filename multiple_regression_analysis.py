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
