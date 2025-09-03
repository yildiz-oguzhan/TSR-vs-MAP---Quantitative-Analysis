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

# Display results
print("TSR Rating Avg (Fall 2019 vs. Pandemic): t = {:.2f}, p = {:.3f}".format(tsr_t, tsr_p))
print("Math RIT (Fall 2019 vs. Pandemic): t = {:.2f}, p = {:.3f}".format(math_t, math_p))
print("Reading RIT (Fall 2019 vs. Pandemic): t = {:.2f}, p = {:.3f}".format(reading_t, reading_p))
