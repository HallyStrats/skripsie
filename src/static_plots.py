import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import shutil
from datetime import datetime, timedelta, time  # Import datetime modules

# Load the data from the Excel file
file_path = '../output_data/simulation_results.xlsx'

# Read the relevant sheets
battery_df = pd.read_excel(file_path, sheet_name='Battery States')
trip_df = pd.read_excel(file_path, sheet_name='Trips')
charger_df = pd.read_excel(file_path, sheet_name='Charger States')
bike_df = pd.read_excel(file_path, sheet_name='Bike States')

# Combine 'Date' and 'Time' columns in all DataFrames to create a proper datetime object
battery_df['Date'] = pd.to_datetime(battery_df['Date'], errors='coerce')
battery_df['DateTime'] = pd.to_datetime(battery_df['Date'].dt.strftime('%Y-%m-%d') + ' ' + battery_df['Time'].astype(str))

trip_df['Date'] = pd.to_datetime(trip_df['Date'], errors='coerce')

charger_df['Date'] = pd.to_datetime(charger_df['Date'], errors='coerce')
charger_df['DateTime'] = pd.to_datetime(charger_df['Date'].dt.strftime('%Y-%m-%d') + ' ' + charger_df['Time'].astype(str))

# ----------------- Original Battery SOC Plot -----------------

# Create a directory name based on the input data
start_date = trip_df['Date'].min().strftime('%Y-%m-%d')
end_date = trip_df['Date'].max().strftime('%Y-%m-%d')
bikes_used = trip_df['Allocated Bike ID'].nunique()  # Number of unique bikes used
batteries_used = battery_df['Battery ID'].nunique()
chargers_used = charger_df['Charger ID'].nunique()

# Get the current time and format it as a string
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

# Define the folder name
folder_name = f"{start_date}_{bikes_used}bikes_{batteries_used}batteries_{chargers_used}chargers_{current_time}"
output_folder_path = f"../output_data/{folder_name}"

# Create the folder if it doesn't exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Plot over the entire time period
plt.figure(figsize=(12, 6))

# Plot over the entire time period
fig, ax = plt.subplots(figsize=(12, 6))

# Iterate through each battery in the data
for battery_id in battery_df['Battery ID'].unique():
    battery_data = battery_df[battery_df['Battery ID'] == battery_id]
    ax.plot(battery_data['DateTime'], battery_data['SOC (kWh)'], label=f'Battery {battery_id}', linewidth=1)

# Add plot details
first_entry_date = battery_df['Date'].min().date()
ax.set_title(f'Battery SOC starting from {first_entry_date}', fontsize=14)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('SOC (kWh)', fontsize=12)
ax.grid(True)

# Set y-axis limits from 0 to 6 kWh
ax.set_ylim(0, 6)

# Set x-axis limits from start to end of data
start_datetime = battery_df['DateTime'].min()
end_datetime = battery_df['DateTime'].max()
ax.set_xlim(start_datetime, end_datetime)

# Format the x-axis to show dates and times
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

# Rotate the x-axis labels for better readability
fig.autofmt_xdate()

# Shrink the plot area to make room for the legend
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

# Place the legend to the right of the plot
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, fontsize='small')

# Save the plot with higher DPI
plt.savefig(f'{output_folder_path}/battery_soc_overall.png', dpi=300, bbox_inches='tight')
plt.close()

 # ----------------- New Graph: Cumulative Distance Travelled by Each Bike per Day -----------------
# Group the bike state data by date
for day, group in bike_df.groupby(bike_df['Date'].dt.date):
    plt.figure(figsize=(10, 6))
    
    # Combine Date and Time columns to create a proper datetime object for x-axis
    group['DateTime'] = pd.to_datetime(group['Date'].astype(str) + ' ' + group['Time'].astype(str))
    
    # Iterate through each bike in the day's data
    for bike_id in group['Bike ID'].unique():
        bike_data = group[group['Bike ID'] == bike_id]
        
        # Plot cumulative distance for each bike using the DateTime column
        plt.plot(bike_data['DateTime'], bike_data['Daily Distance Travelled'], label=f'Bike {bike_id}')
    
    # Add plot details
    plt.title(f'Cumulative Distance Travelled per Bike on {day}')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Distance (km)')
    plt.ylim(0,250)
    plt.legend()

    # Reduce the number of ticks by showing them every hour
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Set interval to 1 hour
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Format to show hours and minutes

    # Rotate the x-axis labels for better readability
    plt.gcf().autofmt_xdate()

    # Save the plot
    plt.savefig(f'{output_folder_path}/cumulative_distance_{day}.png')
    plt.close()

# ----------------- New Graph 1: Box and Whisker for Kilometers Driven -----------------

for day, group in trip_df.groupby(trip_df['Date'].dt.date):
    plt.figure(figsize=(10, 6))
    
    # Get the unique bike IDs and sort them
    sorted_bike_ids = sorted(group['Allocated Bike ID'].unique())
    
    # Prepare the data for the boxplot in the sorted order
    bike_kms = [group[group['Allocated Bike ID'] == bike_id]['Distance'] for bike_id in sorted_bike_ids]
    
    # Create the boxplot with sorted bike IDs
    plt.boxplot(bike_kms, labels=[f'Bike {bike_id}' for bike_id in sorted_bike_ids])
    plt.title(f'Kilometers Driven per Bike on {day}')
    plt.xlabel('Bike ID')
    plt.ylabel('Kilometers Driven')
    plt.grid(True)
    
    # Set y-axis limits from 0 to 50 km
    plt.ylim(0, 40)

    # Save the plot
    plt.savefig(f'{output_folder_path}/kms_driven_{day}.png')
    plt.close()

# ----------------- New Graph 2: Cumulative Power Usage by Each Charger -----------------

# Plot over the entire time period
plt.figure(figsize=(10, 6))

# Pivot the data to have DateTime as index and Charger IDs as columns
charger_pivot = charger_df.pivot_table(
    index='DateTime',
    columns='Charger ID',
    values='Total daily usage',
    aggfunc='first',
    fill_value=0
)

# Plot each charger's power usage
for charger_id in charger_pivot.columns:
    plt.plot(charger_pivot.index, charger_pivot[charger_id], label=f'Charger {charger_id}')

# Calculate cumulative usage over time
cumulative_usage = charger_pivot.sum(axis=1)

# Plot the total grid impact as a background line
plt.plot(
    cumulative_usage.index,
    cumulative_usage.values,
    label='Total Energy Consumption',
    linestyle='--',
    color='black',
    linewidth=1
)

# Add plot details
first_entry_date = charger_df['Date'].min().date()
plt.title(f'Cumulative Energy Consumption from {first_entry_date}')
plt.xlabel('Time')
plt.ylabel('Energy Consumption (kWh)')
plt.legend()
plt.grid(True)

# Set y-axis limits from 0 to 40 kWh
plt.ylim(0, 150)

# Format the x-axis to show dates and times
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

# Rotate the x-axis labels for better readability
plt.gcf().autofmt_xdate()

# Save the plot
plt.savefig(f'{output_folder_path}/charger_usage_overall.png', dpi=300)
plt.close()

# ----------------- New Graph 3: Total Grid Impact per Day (Bar Plot) -----------------

# daily_impact = charger_df.groupby(charger_df['Date'].dt.date)['Total daily usage'].sum()
# plt.figure(figsize=(10, 6))
# daily_impact.plot(kind='bar')

# # Add plot details
# plt.title('Total Grid Impact per Day')
# plt.xlabel('Day')
# plt.ylabel('Total Power Usage (kWh)')
# plt.grid(True)

# # Format the x-axis to show only days
# plt.xticks(rotation=45)

# # Save the plot
# plt.savefig(f'{output_folder_path}/total_grid_impact_per_day.png')
# plt.close()

# ----------------- Updated Plot: Number of Trips vs. Total Minutes Delayed per Day -----------------

# # Group by date and calculate the number of trips and total delay in minutes
# trip_summary = trip_df.groupby(trip_df['Date'].dt.date).agg(
#     total_trips=('Trip ID', 'count'),
#     total_delay_minutes=('Trip Delay (minutes)', 'sum')  # Sum of trip delays per day
# ).reset_index()

# # Now gather the number of bikes, batteries, and chargers used each day
# bikes_used = trip_df['Allocated Bike ID'].nunique()
# batteries_used = battery_df['Battery ID'].nunique()
# chargers_used = charger_df['Charger ID'].nunique()

# # Create a bar plot for total trips per day
# plt.figure(figsize=(10, 6))

# # Plot total trips per day as a bar chart
# plt.bar(trip_summary['Date'], trip_summary['total_trips'], label='Total Trips', color='blue')

# # Overlay the total delay in minutes as a line plot
# plt.plot(trip_summary['Date'], trip_summary['total_delay_minutes'], label='Total Delay (Minutes)', color='orange', marker='o', linewidth=2)

# # Add plot details
# plt.title(f'Trips vs. Total Minutes Delayed per Day\nBikes: {bikes_used}, Batteries: {batteries_used}, Chargers: {chargers_used}')
# plt.xlabel('Day')
# plt.ylabel('Number of Trips / Total Delay (Minutes)')
# plt.legend()
# plt.grid(axis='y')  # Only horizontal grid lines

# # Format the x-axis to show only days
# plt.gca().xaxis.set_major_locator(mdates.DayLocator())
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# # Rotate the x-axis labels for better readability
# plt.xticks(rotation=45)

# # Save the plot
# plt.savefig(f'{output_folder_path}/trips_vs_total_delay.png')
# plt.close()

# ----------------- New Graph: Total Grid Load over Time -----------------

# Plot over the entire time period
plt.figure(figsize=(10, 6))

# Pivot the data to have DateTime as index and Charger IDs as columns for 'Current usage'
charger_usage_pivot = charger_df.pivot_table(
    index='DateTime',
    columns='Charger ID',
    values='Current usage',
    aggfunc='first',
    fill_value=0
)

# Plot each charger's current usage
for charger_id in charger_usage_pivot.columns:
    plt.plot(charger_usage_pivot.index, 60*charger_usage_pivot[charger_id], label=f'Charger {charger_id}')

# Calculate total grid load over time by summing the usage of all chargers at each time point
total_grid_load = 60*charger_usage_pivot.sum(axis=1)

# Plot the total grid load as a dotted line
plt.plot(
    total_grid_load.index,
    total_grid_load.values,
    label='Total Grid Load',
    linestyle='--',
    color='black',
    linewidth=1
)

# Add plot details
first_entry_date = charger_df['Date'].min().date()
plt.title(f'Total Grid Load from {first_entry_date}')
plt.xlabel('Time')
plt.ylabel('Grid load (kW)')
plt.legend()
plt.grid(True)

# Set y-axis limits based on the data
plt.ylim(0, 35*1.1)

# Format the x-axis to show dates and times
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

# Rotate the x-axis labels for better readability
plt.gcf().autofmt_xdate()

# Save the plot
plt.savefig(f'{output_folder_path}/total_grid_load.png', dpi=300)
plt.close()

# ----------------- Move the input file to the output folder -----------------

# Move the simulation results file into the newly created folder
shutil.move(file_path, f'{output_folder_path}/simulation_results.xlsx')