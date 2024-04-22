import matplotlib.pyplot as plt

# Define file paths
file_paths = ['alpha_ours.txt', 'alpha_siamrpn++.txt', 'alpha_siammask.txt', 'alpha_dimp.txt', 'alpha_tomp.txt', 'alpha_mixformer.txt', 'alpha_stark.txt']

# Define Alpha values list
alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Initialize Mean Accuracy lists
mean_accuracies_ours = []
mean_accuracies_mixformer = []
mean_accuracies_stark = []

# Define different colors
colors = [(0/255, 255/255, 0/255),  # Green
          (255/255, 0/255, 0/255),  # Red
          (0/255, 0/255, 255/255),  # Blue
          (255/255, 255/255, 0/255),  # Yellow
          (255/255, 0/255, 255/255),  # Magenta
          (0/255, 255/255, 255/255),  # Cyan
          (128/255, 0/255, 128/255),  # Purple
          (0/255, 128/255, 128/255)]  # Teal

# Plotting line chart for each file
for i, file_path in enumerate(file_paths):
    # Read file content
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Initialize Mean Accuracy list for the file
        file_mean_accuracies = []
        for line in lines:
            # Split line data and extract Mean Accuracy value
            parts = line.split(', ')
            if len(parts) > 1:
                accuracy_str = parts[1].split('Mean Accuracy:')[1].strip()
                accuracy = float(accuracy_str)
                file_mean_accuracies.append(accuracy)

        # Extract type identifier from file name, remove the ".txt" extension from the end
        label = file_path.split(".")[-2].rstrip('.txt').split("_")[-1]

        # Plot line chart
        plt.plot(alpha_values, file_mean_accuracies, label=label, color=colors[i], marker='o')

# Add title and labels
plt.xticks(alpha_values, [f"{alpha:.1f}" for alpha in alpha_values], rotation=45)  # Rotate the x-axis labels for better readability

# Add legend
plt.legend()

# Save the plot as PDF
plt.savefig('alpha.pdf')

# Display the plot
plt.show()
