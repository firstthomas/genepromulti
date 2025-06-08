import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv("tuning_results.csv")

# Step 1: Max test score per experiment
max_scores_per_experiment = (
    df.groupby(['crossover_rate_subtree', 'crossover_rate_node_level', 'experiment'])['test_score']
    .max()
    .reset_index()
)

# Step 2: Median of these max scores per (subtree, node_level) combination
median_of_max_scores = (
    max_scores_per_experiment
    .groupby(['crossover_rate_subtree', 'crossover_rate_node_level'])['test_score']
    .median()
    .reset_index()
)

# Step 3: Pivot for heatmap
heatmap_data = median_of_max_scores.pivot(
    index='crossover_rate_node_level',
    columns='crossover_rate_subtree',
    values='test_score'
)

# Step 4: Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
plt.title("Median of Best Test Scores per Experiment")
plt.xlabel("Crossover Rate Subtree")
plt.ylabel("Crossover Rate Node Level")
plt.tight_layout()
plt.savefig("heatmap_median_best_scores.png")
plt.show()

