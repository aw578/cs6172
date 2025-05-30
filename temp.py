import matplotlib.pyplot as plt

# Test names and their respective accuracy rates
tests = [
    "api_testing",
    "assignment",
    "assignment_simple",
    "assignment_simple_2",
    "backup_restore",
    "compilation_workflow",
    "database_workflow",
    "data_processing",
    "docker_workflow",
    "error_handling",
    "file_operations",
    "git_workflow",
    "network_troubleshooting",
    "python_development",
    "sample",
    "simple",
    "system_monitoring",
    "web_development",
]

accuracy_rates = [
    3 / 5,
    6 / 13,
    4 / 6,
    3 / 9,
    0 / 4,
    0 / 8,
    0 / 3,
    2 / 4,
    3 / 6,
    2 / 17,
    7 / 20,
    5 / 9,
    1 / 3,
    0 / 4,
    0 / 2,
    3 / 10,
    0,  # system_monitoring has 0/0 predictions, treated as 0 accuracy
    3 / 5,
]

# Plotting
plt.figure(figsize=(12, 6))
plt.barh(tests, accuracy_rates, color="skyblue")
plt.xlabel("Accuracy Rate")
plt.title("Accuracy Rates of Synthesized Programs")
plt.xlim(0, 1)
plt.grid(axis="x", linestyle="--", alpha=0.7)

for i, v in enumerate(accuracy_rates):
    plt.text(v + 0.01, i, f"{v:.2%}", va="center", fontsize=8)

plt.tight_layout()
plt.show()
