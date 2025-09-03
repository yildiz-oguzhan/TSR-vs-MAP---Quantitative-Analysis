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
