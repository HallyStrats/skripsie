# helper_functions.py
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import joblib
import logging
from datetime import datetime, timedelta

# Constants
BATTERY_CAPACITY_KWH = 6  # kWh
RECHARGE_TIME_PER_KWH = 20  # Minutes per kWh for recharging
CHARGER_POWER_KW = 7  # kW
CHARGING_EFFICIENCY_LOSS = 0.1  # 10% efficiency loss

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_datetime(datetime_str):
    try:
        return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        logger.error(f"Invalid datetime format: {datetime_str}")
        return None

# Load the Excel data into a variable named 'data'
file_path = '../resources/charging_profile_data.xlsx'  # Provide the correct path if the file is in a different location
charging_data = pd.read_excel(file_path)

# Create a DataFrame from the data
df_charging = pd.DataFrame(charging_data)

# Define the lookup function
def calculate_charge_time(initial_soc: float, target_soc_80: float = 4.8, target_soc_100: float = 6.0) -> dict:
    """
    Calculate the charging time required to reach 80% (4.8 kWh) and 100% (6.0 kWh) from an initial SoC.

    Args:
    initial_soc (float): Initial state of charge in kWh.
    target_soc_80 (float): Target SoC for 80% (default is 4.8 kWh).
    target_soc_100 (float): Target SoC for 100% (default is 6.0 kWh).

    Returns:
    dict: Dictionary with time to 80% and 100% SoC.
    """
    # Interpolation function for time vs. SoC
    interp_func = interp1d(df_charging['SoC (kWh)'], df_charging['Time (minutes)'], kind='linear', fill_value='extrapolate')
    
    # Get the time for initial SoC, 80% SoC, and 100% SoC
    initial_time = interp_func(initial_soc)
    time_to_80 = interp_func(target_soc_80)
    time_to_100 = interp_func(target_soc_100)
    
    # Calculate time needed to charge from initial SoC to 80% and 100%
    time_needed_to_80 = max(time_to_80 - initial_time, 0)
    time_needed_to_100 = max(time_to_100 - initial_time, 0)
    
    return {
        'Time to 80% (minutes)': time_needed_to_80,
        'Time to 100% (minutes)': time_needed_to_100
    }

def update_battery_soc(prev_soc, df_charging):
    """
    Update the battery state of charge (SoC) based on the previous SoC
    using interpolation from the charging profile in df_charging.

    Parameters:
    prev_soc (float): The previous state of charge in kWh.
    df_charging (pd.DataFrame): DataFrame containing 'Time (minutes)' and 'SoC (kWh)' columns.

    Returns:
    float: The updated state of charge in kWh.
    """
    # Ensure the DataFrame is sorted by SoC
    df_charging = df_charging.sort_values('SoC (kWh)').reset_index(drop=True)
    
    # Extract SoC and Time values
    soc_values = df_charging['SoC (kWh)'].values
    time_values = df_charging['Time (minutes)'].values
    
    # Create interpolation functions
    soc_to_time = interp1d(soc_values, time_values, fill_value='extrapolate')
    time_to_soc = interp1d(time_values, soc_values, fill_value='extrapolate')
    
    # Find the current time corresponding to the previous SoC
    current_time = soc_to_time(prev_soc)
    
    # Add one minute to the current time
    next_time = current_time + 1.0  # Assuming time steps are in minutes
    
    # Find the next SoC corresponding to the next time
    next_soc = time_to_soc(next_time)
    
    # Ensure SoC does not exceed the maximum value in the lookup table
    max_soc = soc_values[-1]
    next_soc = min(next_soc, max_soc)
    
    return next_soc

def initiate_recharge(battery, current_time):
    """
    Initiate the recharging process for a battery starting from its current SoC.

    Args:
    battery (Battery): The battery object to recharge.
    current_time (datetime): The current time when charging is initiated.
    """
    # Start charging any battery that is not fully charged exactly at current_time
    if battery.current_charge_kwh < BATTERY_CAPACITY_KWH:  # Battery is not fully charged
        battery.is_charging = True
        battery.charge_start_time = current_time  # Set the start time for charging
        # Calculate charge end time based on the current SoC
        battery.charge_end_time = current_time + timedelta(
            minutes=calculate_charge_time(battery.current_charge_kwh)['Time to 100% (minutes)']
        )
        logger.info(f"Battery {battery.battery_id} started charging at {current_time.time()} from {battery.current_charge_kwh:.2f} kWh.")

def update_recharge_status(battery, current_time):
    """
    Update the state of charge (SoC) for a battery that is charging based on the elapsed time.

    Args:
    battery (Battery): The battery object being charged.
    current_time (datetime): The current time.
    """
    if battery.is_charging:
        # Calculate the elapsed time correctly from the charge start time
        elapsed_minutes = (current_time - battery.charge_start_time).total_seconds() / 60
        
        # Update the SoC incrementally from the current SoC
        if elapsed_minutes > 0:
            initial_soc = battery.current_charge_kwh  # Use the current SoC as the starting point
            updated_soc = update_battery_soc(initial_soc, elapsed_minutes)
            battery.current_charge_kwh = updated_soc
        
        # Stop charging when fully charged
        if current_time >= battery.charge_end_time:
            battery.is_charging = False
            battery.current_charge_kwh = BATTERY_CAPACITY_KWH  # Fully charged
            logger.info(f"Battery {battery.battery_id} is now fully charged at {current_time.time()}.")

def parse_coordinates(coord_str):
    coord_str = coord_str.strip("()")
    lat, long = map(float, coord_str.split(", "))
    return lat, long

def load_trained_model(model_file='trained_model.pkl'):
    model = joblib.load(model_file)
    logger.info(f"Trained model loaded from {model_file}.")
    return model

def load_scaler(scaler_file='output_scaler.pkl'):
    scaler = joblib.load(scaler_file)
    logger.info(f"Scaler loaded from {scaler_file}.")
    return scaler

def extract_datetime_features(datetime_obj):
    hour = datetime_obj.hour
    day_of_week = datetime_obj.weekday()  # Monday = 0, Sunday = 6
    return hour, day_of_week

def predict_trip_details(model, scaler, destination_coordinates, datetime_obj):
    try:
        hour, day_of_week = extract_datetime_features(datetime_obj)
        destination_lat, destination_lon = destination_coordinates

        input_features = pd.DataFrame({
            'hour': [hour],
            'day_of_week': [day_of_week],
            'destination_lat': [destination_lat],
            'destination_lon': [destination_lon]
        })

        # Debugging: Print input features before prediction
        logger.info(f"Predicting trip details for input: {input_features.to_dict(orient='list')}")

        # Set n_jobs=1 to avoid parallel overhead
        model.set_params(n_jobs=1)

        # Predict
        predicted_values = model.predict(input_features)

        # Inverse transform the predicted values to original scale
        predicted_values_original = scaler.inverse_transform(predicted_values)
        predicted_distance_km = predicted_values_original[0][0]
        predicted_duration_min = predicted_values_original[0][1]
        predicted_elev_diff_m = predicted_values_original[0][2]

        if predicted_distance_km < 0 or predicted_duration_min < 0:
            logger.warning(f"Invalid prediction: distance {predicted_distance_km} km, duration {predicted_duration_min} min. Setting values to 0.")
            predicted_distance_km = max(0, predicted_distance_km)
            predicted_duration_min = max(0, predicted_duration_min)
        
        # Debugging: Print prediction results
        logger.info(f"Prediction results: Distance {predicted_distance_km} km, Duration {predicted_duration_min} min, Elevation Diff {predicted_elev_diff_m} m.")

        return 2*predicted_distance_km, 2*predicted_duration_min, predicted_elev_diff_m
    except Exception as e:
        logger.error(f"Error predicting trip details: {e}")
        return None, None, None

def calculate_battery_usage(distance_km, elev_difference_m, duration_min, total_weight_kg):
    try:
        base_usage_per_km = 0.08  # kWh/km
        uphill_increase_factor = 8.0  # 800% increase in usage uphill
        downhill_recuperation = 0.5  # 50% of energy recuperated downhill
        weight_threshold_kg = 200  # kg
        weight_adjustment_factor = 0.004  # 0.4% increase/decrease in usage per kg above/below the threshold
        duration_usage_per_minute = 0.01  # kWh/min for duration

        battery_usage = distance_km * base_usage_per_km

        if elev_difference_m > 0:
            battery_usage += elev_difference_m * uphill_increase_factor * base_usage_per_km / 1000
        elif elev_difference_m < 0:
            battery_usage -= abs(elev_difference_m) * downhill_recuperation * base_usage_per_km / 1000

        if total_weight_kg > weight_threshold_kg:
            excess_weight = total_weight_kg - weight_threshold_kg
            weight_factor = 1 + (excess_weight * weight_adjustment_factor)
        else:
            underweight = weight_threshold_kg - total_weight_kg
            weight_factor = 1 - (underweight * weight_adjustment_factor)

        battery_usage *= weight_factor
        battery_usage += duration_min * duration_usage_per_minute
        battery_usage = min(battery_usage, BATTERY_CAPACITY_KWH)

        return battery_usage
    except Exception as e:
        logger.error(f"Error calculating battery usage: {e}")
        return 0