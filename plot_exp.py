import matplotlib.pyplot as plt
import csv

def load_scores_from_csv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip the header
        return [float(row[1]) for row in reader if row]

def plot_comparison_scores(scores1, scores2, label1, label2):
    generations1 = list(range(1, len(scores1) + 1))
    generations2 = list(range(1, len(scores2) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(generations1, scores1, marker='o', label=label1)
    plt.plot(generations2, scores2, marker='s', label=label2)

    plt.title('Average Test Score per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Average Test Score')
    plt.grid(True)
    plt.legend()
    plt.show()

# Load the scores
scores_baseline = load_scores_from_csv('results_baseline_2/avg_test_scores_baseline.csv')
scores_crossover = load_scores_from_csv('results_node_level_crossover/avg_test_scores_baseline.csv')

# Plot them
plot_comparison_scores(scores_baseline, scores_crossover,
                       label1='Baseline',
                       label2='Node-Level Crossover')
