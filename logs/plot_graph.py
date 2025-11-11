import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# --- Configuration ---
window = 50
drop_nums = [1, 3, 5]
seeds = [500, 1000, 1500] 
colors = {1: "blue", 3: "green", 5: "red"} 
plt.style.use('seaborn-v0_8-whitegrid')

# --- Step 1: Automatically Find and Configure Files ---
data_config = {}
print("Searching for files...")
for dn in drop_nums:
    file_pattern = f"ARNACDD_Hopper-v4_dropnum_{dn}_seed_*_Adam_3rd/{dn}_*_merged.csv"
    found_files = glob.glob(file_pattern)

    if not found_files:
        print(f"No files found for drop_num = {dn} using pattern: {file_pattern}")
        continue
    
    data_config[dn] = {f"seed_{s}": file for s, file in zip(seeds, found_files)}

if not data_config or not any(data_config.values()):
    print("\nCould not find any files. Please ensure the script is in the correct directory and the naming is exact.")
    exit()
print("...Files found and configured successfully.")

# --- Common Processing Function ---
def process_and_plot(data_config, plot_type):
    """Processes data and creates the plots based on plot_type (separate/merged)."""
    
    all_results = {} 
    
    for drop_num, seeds_config in data_config.items():
        all_scores = []
        iterations = None 
        
        for label, file in seeds_config.items():
            try:
                df = pd.read_csv(file)
                df = df.sort_values("epoch")
                
                # Moving average smoothing (window = 50)
                smoothed = df["score"].rolling(window, center=True, min_periods=1).mean()
                all_scores.append(smoothed.values)
                
                if iterations is None:
                    iterations = df["epoch"].values
            except Exception as e:
                print(f"Error reading or processing file {file}: {e}")
                continue

        if not all_scores:
            continue

        all_scores = np.array(all_scores)
        mean_curve = np.nanmean(all_scores, axis=0)
        std_curve = np.nanstd(all_scores, axis=0)
        
        all_results[drop_num] = (iterations, mean_curve, std_curve)

    if plot_type == 'separate':
        for drop_num, (iterations, mean_curve, std_curve) in all_results.items():
            plt.figure(figsize=(12,7), facecolor="white")
            
            shading_factor = 0.5 if drop_num == 1 else 1.0
            
            plt.plot(iterations, mean_curve, label=f"drop_num = {drop_num}",
                     color=colors[drop_num], linewidth=2)
            
            lower_bound = mean_curve - (shading_factor * std_curve)
            upper_bound = mean_curve + (shading_factor * std_curve)
            
            plt.fill_between(iterations, lower_bound, upper_bound,
                             alpha=0.2, color=colors[drop_num])
            
            plt.xlabel("No. of Iterations", fontsize=12)
            plt.ylabel("Score", fontsize=12)
            plt.title(f"Hopper-v4 Performance (drop_num = {drop_num})", fontsize=14)
            plt.ylim(0)
            plt.xlim(0)
            plt.legend()
            plt.savefig(f"Hopper-v4_dropnum_{drop_num}.jpg")
            plt.show()

    elif plot_type == 'merged':
        plt.figure(figsize=(12,7), facecolor="white")
        
        for drop_num, (iterations, mean_curve, std_curve) in all_results.items():
            shading_factor = 0.5 if drop_num == 1 else 1.0
            
            plt.plot(iterations, mean_curve, label=f"drop_num = {drop_num}",
                     color=colors[drop_num], linewidth=2)
            
            lower_bound = mean_curve - (shading_factor * std_curve)
            upper_bound = mean_curve + (shading_factor * std_curve)

            plt.fill_between(iterations, lower_bound, upper_bound,
                             alpha=0.2, color=colors[drop_num])
            
        plt.xlabel("No. of Iterations", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.title("Hopper-v4 Performance", fontsize=14)
        plt.ylim(0)
        plt.xlim(0)
        plt.legend()
        plt.savefig("Hopper-v4.jpg")
        plt.show()


# --- Execute Tasks ---
print("-" * 30)

# Task A: Create 3 separate graphs
print("Executing Task A: Generating 3 separate graphs with conditional shading...")
process_and_plot(data_config, plot_type='separate')

# Task B: Create 1 single merged graph
print("\nExecuting Task B: Generating 1 merged graph with conditional shading...")
process_and_plot(data_config, plot_type='merged')

print("-" * 30)
print("All plotting tasks complete.")