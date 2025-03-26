import os
import matplotlib.pyplot as plt

def plot_data_distribution(dataset_dir, title="Data Distribution", figsize=(10, 5)):
    class_counts = {}
    for class_name in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_path):
            count = len([
                f for f in os.listdir(class_path)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            ])
            class_counts[class_name] = count

    # Plot
    plt.figure(figsize=figsize)
    plt.bar(class_counts.keys(), class_counts.values(), edgecolor='black')
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    # plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
