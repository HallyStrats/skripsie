import os
import pandas as pd
import glob

# Initialize an empty list to store DataFrames
summary_dfs = []

# Define the base directory to search
base_dir = "../output_data/"

# Use glob to find all simulation_results.xlsx files recursively
file_pattern = os.path.join(base_dir, "**", "simulation_results.xlsx")
simulation_files = glob.glob(file_pattern, recursive=True)

# Iterate over each simulation_results.xlsx file
for file_path in simulation_files:
    try:
        # Read the 'Summary' sheet from the Excel file
        df = pd.read_excel(file_path, sheet_name='Summary')
        
        # Optionally, add a column to identify the simulation (e.g., folder name)
        simulation_id = os.path.basename(os.path.dirname(file_path))
        df['Simulation ID'] = simulation_id
        
        # Append the DataFrame to the list
        summary_dfs.append(df)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Combine all summary DataFrames into one
if summary_dfs:
    combined_df = pd.concat(summary_dfs, ignore_index=True)
    
    # Save the combined DataFrame to a new Excel file
    combined_df.to_excel("simulation_summary.xlsx", index=False)
    print("Combined summary saved to simulation_summary.xlsx")
else:
    print("No simulation_results.xlsx files found.")