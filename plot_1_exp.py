import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
df_node = pd.read_csv("test_scores_node_crossover copy.csv")
df_subtree = pd.read_csv("test_scores_subtree_crossover copy.csv")

# Group by generation
grouped_node = df_node.groupby("generation")["test_score"]
grouped_subtree = df_subtree.groupby("generation")["test_score"]

# Compute statistics for node crossover
median_node = grouped_node.median()
std_dev_node = grouped_node.std()

# Compute statistics for subtree crossover
median_subtree = grouped_subtree.median()
std_dev_subtree = grouped_subtree.std()

# Plot the median lines
plt.plot(median_node.index, median_node.values, label="Node Crossover - median", color="black", linewidth=2)
plt.plot(median_subtree.index, median_subtree.values, label="Baseline - median", color="blue", linewidth=2)

# Plot ±1 standard deviation as shaded areas
plt.fill_between(median_node.index, median_node - std_dev_node, median_node + std_dev_node,
                 color="gray", alpha=0.3, label="Node Crossover ±1 Std Dev")
plt.fill_between(median_subtree.index, median_subtree - std_dev_subtree, median_subtree + std_dev_subtree,
                 color="skyblue", alpha=0.3, label="Baseline ±1 Std Dev")

# Labels and styling
plt.xlabel("Generation")
plt.ylabel("Test Score")
plt.title("Test Score Statistics per Generation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.ylim(-1500, 1000)  # Set y-axis limits
# plt.xlim(0, 20)  # Set y-axis limits

plt.savefig("test_scores_comparison_median.png", dpi=300)  # Save the figure
plt.show()
