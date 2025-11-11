import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# --- Configuration ---
window = 50
algos = ['NPG', 'PG', 'PPO', 'TRPO'] 
seeds = [500, 1000, 1500] 

colors = {"NPG": "blue", "PG": "green", "PPO": "red", "TRPO": "purple"} 
plt.style.use('seaborn-v0_8-whitegrid')

# --- Step 1: Automatically Find and Configure Files ---
data_config = {}
print("Searching for files...")
for algo in algos:
    file_pattern = f"{algo}_Hopper-v4_seed_*_2nd/{algo}_*_merged.csv"
    found_files = glob.glob(file_pattern)

    if not found_files:
        print(f"No files found for algo = {algo} using pattern: {file_pattern}")
        continue
    
    if len(found_files) != len(seeds):
        print(f"Found {len(found_files)} files for {algo}, expected {len(seeds)}.")

    found_files.sort() 
    
    data_config[algo] = {f"seed_{s}": file for s, file in zip(seeds, found_files)}

if not data_config or not any(data_config.values()):
    print("\nCould not find any files. Please ensure the script is in the correct directory and the naming is exact.")
    exit()
print("...Files found and configured successfully.")

# --- Common Processing Function ---
def process_and_plot(data_config, plot_type):
    """Processes data and creates the plots based on plot_type (separate/merged)."""
    
    # 1. Processing and Storage (Same logic, no change needed here)
    all_results = {} 
    
    for algo, seeds_config in data_config.items():
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
        
        all_results[algo] = (iterations, mean_curve, std_curve)
        
    if plot_type == 'separate':
        for algo, (iterations, mean_curve, std_curve) in all_results.items():
            plt.figure(figsize=(12,7), facecolor="white")
            
            shading_factor = 0.50 if algo == 'NPG' else 1.0
            
            plt.plot(iterations, mean_curve, label=f"{algo}",
                     color=colors[algo], linewidth=2) 
            
            lower_bound = mean_curve - (shading_factor * std_curve)
            upper_bound = mean_curve + (shading_factor * std_curve)
            
            plt.fill_between(iterations, lower_bound, upper_bound,
                             alpha=0.2, color=colors[algo])
            
            plt.xlabel("No. of Iterations", fontsize=12)
            plt.ylabel("Score", fontsize=12)
            plt.title(f"Hopper-v4 Performance ({algo})", fontsize=14)
            plt.ylim(0)
            plt.xlim(0)
            plt.legend()
            plt.savefig(f"Hopper-v4_algo_{algo}.jpg")
            plt.show()

    elif plot_type == 'merged':
        plt.figure(figsize=(12,7), facecolor="white")
        
        for algo, (iterations, mean_curve, std_curve) in all_results.items():
            
            shading_factor = 0.50 if algo == 'NPG' else 1.0
            
            plt.plot(iterations, mean_curve, label=f"{algo}",
                     color=colors[algo], linewidth=2)
            
            lower_bound = mean_curve - (shading_factor * std_curve)
            upper_bound = mean_curve + (shading_factor * std_curve)

            plt.fill_between(iterations, lower_bound, upper_bound,
                             alpha=0.2, color=colors[algo])
            
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

# Task A: Create 4 separate graphs (one for each algo)
print("Executing Task A: Generating 4 separate graphs...")
process_and_plot(data_config, plot_type='separate')

# Task B: Create 1 single merged graph
print("\nExecuting Task B: Generating 1 merged graph...")
process_and_plot(data_config, plot_type='merged')

print("-" * 30)
print("All plotting tasks complete. Total 5 graphs generated (4 separate + 1 merged).")