import pandas as pd
from datetime import timedelta, datetime, time
import logging
from helper_functions import load_trained_model, load_scaler, parse_coordinates, predict_trip_details, calculate_battery_usage, update_battery_soc
import openpyxl  # Make sure this library is installed
import numpy as np
import os
import argparse

BIKE_ALLOCATION = 1
MAX_DELAY = 30

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the Excel data into a variable named 'data'
file_path = '../resources/charging_profile_data.xlsx'  # Provide the correct path if the file is in a different location
charging_data = pd.read_excel(file_path)

# Create a DataFrame from the data
df_charging = pd.DataFrame(charging_data)

# Classes for Trips, Bikes, Batteries, and Chargers
class Trip:
    def __init__(self, trip_id):
        self.trip_id = trip_id
        self.date = None
        self.time = None
        self.duration = 0
        self.distance = 0
        self.elev = 0
        self.alloc_bike_id = 0
        self.battery_usage_per_minute = 0

class Bike:
    def __init__(self, bike_id):
        self.bike_id = bike_id
        self.battery = None  # Reference to an allocated Battery object
        self.is_available = True
        self.range = 6
        self.last_trip_end_time = None  # End time of the last trip
        self.total_distance_travelled = 0  # Track total distance traveled
        self.allocated_trip_id = None  # Track the current trip allocated to the bike
        self.battery_usage_per_minute = 0  # Store battery usage per minute for the current trip
        self.allocated_battery_id = 0
        self.battery_swap = None

class Battery:
    def __init__(self, battery_id):
        self.battery_id = battery_id
        self.current_charge_kwh = 6  # Full charge (6 kWh)
        self.is_charging = False
        self.swaps = 0
        self.allocated_bike_id = None  # The bike using this battery
        self.status = 'available'
        self.charging_start_time = None
        self.allocated_charger_id = 0
        self.prev_charge_kwh = 6
        self.daily_usage = 0

class Charger:
    def __init__(self,charger_id):
        self.charger_id = charger_id
        self.charging = False
        self.allocated_battery_id = 0
        self.battery_current_soc = 0
        self.daily_usage = 0
        self.battery_prev_soc = 0
        self.prev_charging_status = False
        self.current_usage = 0

def parse_args():
    parser = argparse.ArgumentParser(description="Run the bike simulation with specified fleet size, battery pool size, and number of chargers.")
    
    # Add arguments for fleet size, pool size, and chargers
    parser.add_argument('--fleet_size', type=int, default=2, help="Size of the bike fleet")
    parser.add_argument('--pool_size', type=int, default=4, help="Size of the battery pool")
    parser.add_argument('--chargers', type=int, default=1, help="Number of chargers")
    
    args = parser.parse_args()
    return args

def run_simulation(trip_orders_file, fleet_size, pool_size, chargers, model_file='../resources/trained_model.pkl', scaler_file='../resources/output_scaler.pkl'):
    try:
        # Load trip data
        trip_orders = pd.read_csv(trip_orders_file)
        trip_orders['datetime'] = pd.to_datetime(trip_orders['datetime'])
        trip_orders['destination'] = trip_orders['destination'].apply(parse_coordinates)

        # Load model and scaler
        model = load_trained_model(model_file)
        scaler = load_scaler(scaler_file)

        # Initialize tracking lists
        trip_data = []
        bike_state_data = []  # New list to store bike states
        battery_state_data = []  # List to store battery states
        charger_state_data = [] # List to store charger states
        summary_data = []

        # Initialise bikes and batteries
        bikes = [Bike(bike_id = i) for i in range(1, fleet_size + 1)]  # Create the bikes
        batteries = [Battery(battery_id = i) for i in range(1, pool_size + 1)]  # Create the batteries
        chargers = [Charger(charger_id = i) for i in range(1,chargers +1 )]

        # First loop: Trip allocation
        trip_orders['date'] = trip_orders['datetime'].dt.date
        days = trip_orders['date'].unique()

        for day in days:
            daily_trips = trip_orders[trip_orders['date'] == day]
            problems = allocate_trips(daily_trips, model, scaler, trip_data, bikes)
            if problems == 2:
                return problems
            missed_battery_swaps = 0

        # Use trip_data directly instead of reading from Excel
        trip_df = pd.DataFrame(trip_data, columns=['Trip ID', 'Date', 'Time', 'Distance', 'Duration', 'Elevation', 'Allocated Bike ID', 'Trip Successful', 'Battery Usage (kWh)', 'Battery Usage Per Minute (kWh)', 'Bike Mileage after Trip','Trip Delay (minutes)', 'Trip Efficiency (kWh per KM)'])

        # Simulate bike states for each day
        for day in days:
            day_trips = trip_df[trip_df['Date'] == day]
            simulate_bike_states_for_day(day, day_trips, bikes, bike_state_data)

        # Convert bike state data to DataFrame for writing to Excel
        bike_state_df = pd.DataFrame(bike_state_data, columns=['Date', 'Time', 'Bike ID', 'Is Available', 'Remaining Range (kWh)', 'Daily Distance Travelled', 'Battery Swap'])

        # Simulate battery states for each day
        for day in days:
            missed_battery_swaps = [0]  # Initialize missed battery swaps counter for the day
            problems = simulate_battery_states_for_day(day, batteries, bikes, chargers, bike_state_df, battery_state_data, charger_state_data, missed_battery_swaps)
            if problems > 0:
                return problems
        # Convert battery state data to DataFrame for writing to Excel
        battery_state_df = pd.DataFrame(battery_state_data, columns=['Date', 'Time', 'Battery ID', 'Status', 'SOC (kWh)', 'Swaps', 'Allocated Bike ID', 'Allocated Charger ID','Daily Usage (kWh)'])

        # for day in days:
        #     simulate_charger_states_for_day(day, chargers, battery_state_df, charger_state_data)

        charger_state_df = pd.DataFrame(charger_state_data, columns=['Date', 'Time', 'Charger ID', 'Charging', 'Allocated battery ID', 'Current battery SOC', 'Total daily usage', 'Current usage'])

        for day in days:
            total_trips = len(trip_df[trip_df['Date'] == day])
            total_trip_delay = trip_df[trip_df['Date'] == day]['Trip Delay (minutes)'].sum()
            if len(batteries)<len(chargers):
                total_charging_delay = 0
            else:
                total_charging_delay = battery_state_df[(battery_state_df['Date'] == day) & (battery_state_df['Status'] == "waiting")].shape[0]

            summary_data.append([
                day,
                total_trips,
                len(bikes),
                total_trip_delay,
                len(batteries),
                missed_battery_swaps[0],
                len(chargers),
                total_charging_delay,
                MAX_DELAY
            ])

        summary_data_df = pd.DataFrame(summary_data, columns = ['Date', 'Number of Trips', 'Fleet Size', 'Total Trip Delay', 'Battery Pool Size', 'Missed Battery Swaps', 'Chargers', 'Total Charging Delay', 'Max Delay'])
        # Define the output path for the Excel file
        output_path = f'../output_data/simulation_results.xlsx'

        # Ensure the directory exists
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        # Write all three DataFrames (Trips, Bike States, Battery States) to the Excel file
        with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
            summary_data_df.to_excel(writer, sheet_name = 'Summary', index=False)
            trip_df.to_excel(writer, sheet_name='Trips', index=False)  # Write trip data to 'Trips' sheet
            bike_state_df.to_excel(writer, sheet_name='Bike States', index=False)  # Write bike state data to 'Bike States' sheet
            battery_state_df.to_excel(writer, sheet_name='Battery States', index=False)  # Write bike state data to 'Battery States' sheet
            charger_state_df.to_excel(writer, sheet_name = 'Charger States', index=False)



        logger.info("Simulation completed and simulation_results.xlsx populated successfully.")
        
    except Exception as e:
        logger.error(f"Error during simulation: {e}", exc_info=True)
    return 0


def simulate_bike_states_for_day(day, day_trips, bikes, bike_state_data):
    """
    Simulate the bike states using data from the `trip_data` for a specific day.
    The bike should be available between trips and unavailable during trips.
    """
    # Pre-filter the trip data by bike for efficiency
    trips_by_bike = {bike_id: day_trips[day_trips['Allocated Bike ID'] == bike_id] for bike_id in day_trips['Allocated Bike ID'].unique()}

    start_time = datetime.combine(day, time(8, 0))  # Start at 8 AM
    end_time = datetime.combine(day, time(21, 0))  # End at 9 PM

    for bike in bikes:
        bike.range = 6
        bike.total_distance_travelled = 0

    current_time = start_time

    while current_time <= end_time:
        print(f"Simulating bike states for: {current_time}")
        for bike in bikes:
            bike.battery_swap = None
            # Get the trips for this bike
            bike_trips = trips_by_bike.get(bike.bike_id, pd.DataFrame())

            if not bike_trips.empty:
                # Check if the current time falls within any of the bike's trips
                bike_on_trip = False
                for _, trip in bike_trips.iterrows():
                    trip_start_time = datetime.combine(day, trip['Time'])  # Start time of the trip
                    trip_end_time = trip_start_time + timedelta(minutes=np.ceil(trip['Duration']))  # End time of the trip
                    trip_battery_usage_per_min = trip['Battery Usage Per Minute (kWh)']
                    trip_distance_per_min = trip['Distance'] / trip['Duration']

                    if trip_start_time <= current_time < trip_end_time:
                        # The bike is on a trip during this time
                        bike.is_available = False
                        bike.range -= trip_battery_usage_per_min
                        bike.total_distance_travelled += trip_distance_per_min
                        bike_on_trip = True
                        break

                if not bike_on_trip:
                    # The bike is available
                    bike.is_available = True
            else:
                bike.is_available = True
            # Bike Swap
            if bike.is_available == True and bike.range < 1.2:
                bike.range = 6
                bike.battery_swap = 1

            # Record the state of the bike
            bike_state_data.append([day, current_time.time(), bike.bike_id, bike.is_available, bike.range, bike.total_distance_travelled, bike.battery_swap])

        current_time += timedelta(minutes=1)  # Increment by 1 minute

def find_min_swaps(batteries):
    """
    This function finds and returns the minimum number of swaps among available batteries.
    
    Parameters:
    batteries (list): A list of battery objects, where each object has attributes like `status` and `swaps`.

    Returns:
    int: The minimum number of swaps among available batteries, or float('inf') if no available battery is found.
    """
    min_swaps = float('inf')  # Start with an infinitely large number

    # Loop through the batteries to find the minimum swaps for available batteries
    for new_battery in batteries:
        if new_battery.status == 'available' and new_battery.swaps < min_swaps:
            min_swaps = new_battery.swaps

    print("Minimum swaps of available batteries: ", min_swaps)
    return min_swaps

def simulate_battery_states_for_day(day, batteries, bikes, chargers, bike_state_df, battery_state_data, charger_state_data, missed_battery_swaps):
    start_time = datetime.combine(day, time(8, 0))  # Start at 8 AM
    end_time = datetime.combine(day, time(21, 0))  # End at 9 PM
    current_time = start_time

    for battery in batteries:
        battery.current_charge_kwh = 6
        battery.daily_usage = 0

    for charger in chargers:
        charger.daily_usage = 0

    # Allocate initial batteries to bikes at the start of the day
    allocate_batteries_to_bikes(bikes, batteries)

    while current_time <= end_time:
        print(f"Simulating battery states for: {current_time}")

        # Keep track of bikes that have already swapped in this time step
        bikes_swapped = set()

        # Iterate through all batteries, regardless of whether they are allocated or not
        for battery in batteries:
            charger_available = 0
            battery_swap = False
            if battery.current_charge_kwh < 0:
                return 2

            # Find the bike associated with the battery, if any
            bike = next((b for b in bikes if b.allocated_battery_id == battery.battery_id), None)

            # If the battery is allocated to a bike
            if bike is not None:
                # Skip if this bike has already swapped in this time step
                if bike.bike_id in bikes_swapped:
                    continue
                # Find the corresponding bike state for this time in the bike_state_df
                bike_state_at_time = bike_state_df[
                    (bike_state_df['Bike ID'] == bike.bike_id) & 
                    (bike_state_df['Time'] == current_time.time()) &
                    (bike_state_df['Date'] == day)
                ]

                if not bike_state_at_time.empty:
                    # Check if the bike is in use (battery depleting)
                    if bike_state_at_time['Is Available'].values[0] == False and battery.allocated_bike_id > 0:
                        battery.status = 'depleting'
                        battery.current_charge_kwh = bike_state_at_time['Remaining Range (kWh)'].values[0]
                    
                    # Check if the bike is available (battery not depleting)
                    elif bike_state_at_time['Is Available'].values[0] == True and battery.allocated_bike_id > 0:
                        battery.status = 'in-use'

                    # Handle battery swap: if a swap happens and the bike is available
                    if bike_state_at_time['Battery Swap'].values[0] == 1 and bike_state_at_time['Is Available'].values[0] == True:
                        # Only deallocate the current battery (which is associated with this bike) and mark it as charging
                        if battery.allocated_bike_id == bike.bike_id:
                            for charger in chargers:
                                if charger.allocated_battery_id == 0 and charger.charging == False:
                                    charger_available = 1
                                    charger.charging = True
                                    battery.allocated_charger_id = charger.charger_id
                                    charger.allocated_battery_id = battery.battery_id
                                    break
                                else: charger_available = 0

                            if charger_available == 1:
                                battery.status = 'charging'

                            else: battery.status = 'waiting'

                            battery.allocated_bike_id = 0  # Deallocate the battery from the bike
                            bike.allocated_battery_id = 0  # Reset the bike's allocated battery
                            battery.swaps += 1

                            min_swaps = find_min_swaps(batteries)

                            # Find a new available battery to allocate
                            for new_battery in batteries:
                                print(f"Battery {new_battery.battery_id} has {new_battery.swaps} swaps")
                                if new_battery.status == 'available' or new_battery.status == 'avail and charging' and new_battery.swaps == min_swaps:  # Ensure only one available battery is selected
                                    battery_swap = True
                                    new_battery.allocated_bike_id = bike.bike_id  # Assign new battery to the bike
                                    bike.allocated_battery_id = new_battery.battery_id  # Update bike's battery ID
                                    new_battery.status = 'in-use'  # Set the new battery as in-use
                                    print(f"Assigned battery {new_battery.battery_id} at {current_time} with {new_battery.swaps} swaps")
                                    break  # Exit the loop after assigning one battery
                            if battery_swap:
                                bikes_swapped.add(bike.bike_id)  # Mark bike as swapped
                                print(f"Checkout this time: {current_time} on {day}")

                            if not battery_swap:
                                missed_battery_swaps[0] += 1
                                # print(f"No available battery for swap at {current_time} on {day}. Incremented missed battery swaps to {missed_battery_swaps}")
                                print("Battery Swap Missed!!!")
                                return 1

            else:
                # If the battery is not allocated to any bike, check its status
                if battery.status == 'charging' or battery.status == 'avail and charging':
                    battery.current_charge_kwh = update_battery_soc(battery.current_charge_kwh, df_charging)
                    if battery.current_charge_kwh >= 4.8:  # Full charge threshold
                        battery.status = 'avail and charging'  # Once fully charged, mark as available
                        if battery.current_charge_kwh >= 6:  # Full charge threshold
                            battery.status = 'available'  # Once fully charged, mark as available
                            battery.allocated_charger_id = 0
                            battery.current_charge_kwh = 6.0  # Cap the charge at full
                            for charger in chargers:
                                if charger.allocated_battery_id == battery.battery_id:
                                    charger.charging = False
                                    charger.allocated_battery_id = 0
                                    break

                if battery.current_charge_kwh == 6:
                    battery.status = 'available'  # Set to available if not charging
            
            if battery.status == 'depleting':
                battery.daily_usage += (battery.prev_charge_kwh - battery.current_charge_kwh)
            
            battery.prev_charge_kwh = battery.current_charge_kwh

            # Record the state of the battery, whether it's allocated or not
            battery_state_data.append([
                day, current_time.time(), battery.battery_id, battery.status, battery.current_charge_kwh,
                battery.swaps, battery.allocated_bike_id or 0, battery.allocated_charger_id, battery.daily_usage
            ])

        update_charger_status(day, current_time, chargers, batteries, charger_state_data)

        current_time += timedelta(minutes=1)  # Increment by 1 minute

    current_time = datetime.combine(day, time(21, 0))+timedelta(minutes=1)

    # Overnight charging loop from 9 PM to 7:59 AM next day
    current_time = datetime.combine(day, time(21, 0)) + timedelta(minutes=1)
    overnight_end_time = datetime.combine(day + timedelta(days=1), time(7, 59))

    while current_time <= overnight_end_time:
        print(f"Simulating battery states for: {current_time}")
        for battery in batteries:
            battery.allocated_bike_id = 0
            charger_available = 0

            if battery.allocated_charger_id > 0:
                battery.status = 'charging'
            elif battery.allocated_charger_id == 0:
                if battery.current_charge_kwh == 6:
                    battery.status = 'available'
                else:
                    # Try to allocate a charger
                    for charger in chargers:
                        if charger.allocated_battery_id == 0 and not charger.charging:
                            charger_available = 1
                            charger.charging = True
                            battery.allocated_charger_id = charger.charger_id
                            charger.allocated_battery_id = battery.battery_id
                            battery.status = 'charging'
                            break
                    if charger_available == 0:
                        battery.status = 'waiting'

            if battery.status == 'charging':
                battery.current_charge_kwh = update_battery_soc(battery.current_charge_kwh, df_charging)
                if battery.current_charge_kwh >= 6:  # Full charge threshold
                    battery.status = 'available'  # Once fully charged, mark as available
                    battery.allocated_charger_id = 0
                    battery.current_charge_kwh = 6.0  # Cap the charge at full
                    for charger in chargers:
                        if charger.allocated_battery_id == battery.battery_id:
                            charger.charging = False
                            charger.allocated_battery_id = 0
                            break

            if battery.status == 'available':
                battery.allocated_charger_id = 0

            # Record the state of the battery
            battery_state_data.append([
                current_time.date(), current_time.time(), battery.battery_id, battery.status, battery.current_charge_kwh,
                battery.swaps, battery.allocated_bike_id or 0, battery.allocated_charger_id, battery.daily_usage
            ])

        update_charger_status(current_time.date(), current_time, chargers, batteries, charger_state_data)

        # Increment current_time by 1 minute
        current_time += timedelta(minutes=1)

    for charger in chargers:
        if charger.charging == True:
            return 3

    return 0


def allocate_batteries_to_bikes(bikes, batteries):
    """
    Allocate batteries to bikes, prioritizing batteries with the least number of swaps.
    """
    # Sort the batteries by the number of swaps (least swaps first)
    available_batteries = sorted([b for b in batteries if b.status == 'available'], key=lambda b: b.swaps)

    # Iterate through bikes and assign the batteries with the least swaps
    for i, bike in enumerate(bikes):
        if i < len(available_batteries):  # Ensure we don't exceed the number of available batteries
            bike.range = 6
            bike.allocated_battery_id = available_batteries[i].battery_id  # Assign the battery ID to the bike
            available_batteries[i].allocated_bike_id = bike.bike_id  # Track which bike is using the battery
            available_batteries[i].status = 'in-use'  # Set the battery's status to in-use
        else:
            break  # Stop allocating if there are no more available batteries

def allocate_trips(daily_trips, model, scaler, trip_data, bikes):

    for i, trip in daily_trips.iterrows():
        trip_successful = False
        delay_time = 0
        trip_id = i + 1
        trip_date = trip['date']
        trip_datetime = trip['datetime']

        while not trip_successful and delay_time <= MAX_DELAY:
            # Use the updated trip_datetime
            trip_distance, trip_duration, trip_elev = predict_trip_details(
                model, scaler, trip['destination'], trip_datetime
            )

            allocated_bike_id = allocate_bike_to_trip(trip_datetime, trip_duration, trip_distance, bikes)

            if allocated_bike_id == 0:
                trip_datetime += timedelta(minutes=1)
                delay_time += 1

                # Check if any bikes will become available within the remaining delay time
                next_available_time = min((b.last_trip_end_time for b in bikes if b.last_trip_end_time), default=None)
                if next_available_time is None or next_available_time > trip_datetime + timedelta(minutes=MAX_DELAY - delay_time):
                    print(f"No bikes will become available for trip {trip_id} within delay time.")
                    break
            else:
                trip_successful = True
                bike = next(b for b in bikes if b.bike_id == allocated_bike_id)
                bike_mileage = bike.total_distance_travelled

                battery_usage = calculate_battery_usage(trip_distance, trip_elev, trip_duration, total_weight_kg=150)
                battery_usage_per_minute = battery_usage / trip_duration

                bike.allocated_trip_id = trip_id
                bike.battery_usage_per_minute = battery_usage_per_minute

        if not trip_successful:
            print(f"Trip {trip_id} could not be allocated a bike within {MAX_DELAY} minutes delay.")
            return 2
            # Handle trip failure logic here if necessary

        trip_data.append([
            trip_id, trip_date, trip_datetime.time(), trip_distance,
            trip_duration, trip_elev, allocated_bike_id, trip_successful,
            battery_usage, battery_usage_per_minute, bike_mileage, delay_time, battery_usage/trip_distance
        ])

    return 0

def allocate_bike_to_trip(trip_time, trip_duration, trip_distance, bikes):
    min_mileage = float('inf')
    selected_bike = None

    for bike in bikes:
        # Availability check without is_available
        if bike.last_trip_end_time is None or bike.last_trip_end_time <= trip_time:
            if bike.total_distance_travelled < min_mileage and BIKE_ALLOCATION == 1:
                min_mileage = bike.total_distance_travelled
                selected_bike = bike
            elif BIKE_ALLOCATION == 0:
                selected_bike = bike


    if selected_bike is not None:
        selected_bike.last_trip_end_time = trip_time + timedelta(minutes=trip_duration)
        selected_bike.total_distance_travelled += trip_distance
        return selected_bike.bike_id

    print("No available bike for the trip.")
    return 0

def update_charger_status(day, current_time, chargers, batteries, charger_state_data):

    for charger in chargers:
        for battery in batteries:
            if battery.allocated_charger_id == charger.charger_id:
                charger.charging = True
                charger.allocated_battery_id = battery.battery_id
                charger.battery_current_soc = battery.current_charge_kwh
                break

            else:
                charger.charging = False
                charger.allocated_battery_id = 0
                charger.battery_current_soc = 0

        if charger.charging == True:
            if charger.battery_prev_soc <= charger.battery_current_soc:
                charger.current_usage = charger.battery_current_soc - charger.battery_prev_soc
                charger.daily_usage += (charger.battery_current_soc - charger.battery_prev_soc)
            if charger.prev_charging_status == False:
                charger.current_usage = 0
                charger.daily_usage -= charger.battery_current_soc
        else:
            charger.current_usage = 0
        charger.battery_prev_soc = charger.battery_current_soc
        charger.prev_charging_status = charger.charging

        charger_state_data.append([
            day, current_time.time(), charger.charger_id, charger.charging, charger.allocated_battery_id, charger.battery_current_soc, charger.daily_usage, 1.1111*charger.current_usage
        ])
    return charger_state_data

if __name__ == "__main__":
    args = parse_args()
    current_pool_size = args.pool_size
    current_fleet_size = args.fleet_size
    current_chargers = args.chargers
    problem = 1  # Initialize to 1 to enter the loop

    while problem > 0:
        print(f"Running simulation with battery pool size {current_pool_size}")
        problem = run_simulation('../input_data/trip_orders.csv', fleet_size=current_fleet_size, pool_size=current_pool_size, chargers=current_chargers)
        if problem == 1:
            current_pool_size += 1
            print(f"Missed battery swap occurred. Retrying simulation with battery pool size {current_pool_size}")
        if problem == 2:
            current_fleet_size += 1
            current_pool_size += 1
            print(f"Trip delay too high. Retrying simulation with fleet size {current_fleet_size}")
        if problem == 3:
            current_chargers += 1
            print(f"Charger delay too high. Retrying simulation with {current_chargers} chargers")
