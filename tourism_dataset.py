import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load Dataset
df = pd.read_csv("tourism_dataset.csv")

# Numerical Columns
data = df[["Visitors", "Rating", "Revenue"]]

# Normalize Data
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(
    scaler.fit_transform(data),
    columns=data.columns
)

# Convert Wide Format to Long Format
plot_data = scaled_data.reset_index().melt(
    id_vars="index",
    var_name="Feature",
    value_name="Value"
)

# Plot
plt.figure(figsize=(12,6))
sns.lineplot(
    data=plot_data,
    x="index",
    y="Value",
    hue="Feature",
    marker="o"
)

plt.title("Tourism Dataset - Multi Line Plot", fontsize=16)
plt.xlabel("Record Number")
plt.ylabel("Normalized Value")
plt.grid(True)
plt.legend(title="Features")
plt.tight_layout()

plt.show()
