import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the Excel file
file_path = 'simulation_summary.xlsx'
summary_df = pd.read_excel(file_path)

# ----------------- Plot 1: Number of Trips vs. Fleet Size -----------------
plt.figure(figsize=(10, 6))
plt.scatter(summary_df['Fleet Size'], summary_df['Number of Trips'], color='blue')
plt.title('Number of Trips vs. Fleet Size')
plt.xlabel('Fleet Size')
plt.ylabel('Number of Trips')
plt.grid(True)
plt.savefig('../output_data/number_of_trips_vs_fleet_size.png')
plt.close()

# ----------------- Plot 2: Avg Trips per Bike vs. Battery Pool Size -----------------
# Calculate Avg Trips per Bike (Fleet Size / Number of Trips)
summary_df['Avg Trips per Bike'] = summary_df['Number of Trips'] / summary_df['Fleet Size']

plt.figure(figsize=(10, 6))
plt.scatter(summary_df['Avg Trips per Bike'], summary_df['Battery Pool Size'] , color='green')
plt.title('Avg Trips per Bike vs. Battery Pool Size')
plt.ylabel('Battery Pool Size')
plt.xlabel('Avg Trips per Bike')
plt.grid(True)
plt.savefig('../output_data/avg_trips_per_bike_vs_battery_pool_size.png')
plt.close()

# ----------------- Plot 3: Avg Trips per Bike vs. Total Trip Delay -----------------
plt.figure(figsize=(10, 6))
plt.scatter(summary_df['Avg Trips per Bike'], summary_df['Total Trip Delay'], color='purple')
plt.title('Avg Trips per Bike vs. Total Trip Delay')
plt.ylabel('Total Trip Delay (minutes)')
plt.xlabel('Avg Trips per Bike')
plt.grid(True)
plt.savefig('../output_data/avg_trips_per_bike_vs_total_trip_delay.png')
plt.close()

# ----------------- Plot 4: Avg Batteries per Charger vs. Total Charging Delay -----------------
# Calculate Avg Batteries per Charger (Battery Pool Size / Chargers)
summary_df['Avg Batteries per Charger'] = summary_df['Battery Pool Size'] / summary_df['Chargers']

plt.figure(figsize=(10, 6))
plt.scatter(summary_df['Avg Batteries per Charger'], summary_df['Total Charging Delay'], color='orange')
plt.title('Avg Batteries per Charger vs. Total Charging Delay')
plt.ylabel('Total Charging Delay (minutes)')
plt.xlabel('Avg Batteries per Charger')
plt.grid(True)
plt.savefig('../output_data/avg_batteries_per_charger_vs_total_charging_delay.png')
plt.close()

# ----------------- Plot 5: Pairplot of Key Variables (with Avg Trip and Avg Charging Delay) -----------------
# Calculate Average Trip Delay and Average Charging Delay
summary_df['Avg Trip Delay'] = summary_df['Total Trip Delay'] / summary_df['Number of Trips']
summary_df['Avg Charging Delay'] = summary_df['Total Charging Delay'] / summary_df['Chargers']

# Select the columns for the pairplot
pairplot_columns = ['Number of Trips', 'Fleet Size', 'Avg Trip Delay', 'Battery Pool Size', 'Chargers', 'Avg Charging Delay']

# Generate the pairplot
sns.pairplot(summary_df[pairplot_columns], diag_kind='kde')
plt.savefig('../output_data/pairplot_avg_delays.png')
plt.show()