import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parameters
origin_lat, origin_long = -33.938299405675814, 18.858465193281816
start_time = datetime.strptime("2024-12-09 08:00", "%Y-%m-%d %H:%M")
end_time = datetime.strptime("2024-12-15 20:00", "%Y-%m-%d %H:%M")
days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Time blocks adjusted to singular hours
time_blocks = [f"{hour}:00-{hour+1}:00" for hour in range(8, 20)]  # From 08:00 to 20:00

# Suburb data
suburbs = [
    "Cloetesville", "Dalsig", "De Zalze", "Die Boord", "Jamestown",
    "Karindal", "Krigeville", "La Colline", "Mostertsdrift",
    "Onder Papegaaiberg", "Paradyskloof", "Rozendal", "Simonswyk",
    "Stellenbosch Central", "Techno Park", "Universiteitsoord", "Kayamandi"
]
suburb_lat = [
    -33.90770887312897, -33.949945420294874, -33.97278131480808,
    -33.94766254112964, -33.97953144558116, -33.9373571518299,
    -33.943867555604506, -33.92397110391025, -33.9336235193442,
    -33.935958728806014, -33.9635231915123, -33.93291829463445,
    -33.92797933821835, -33.933972523994996, -33.96496303240772,
    -33.929204838131554, -33.91917089753285
]
suburb_long = [
    18.854509038674426, 18.859110733905027, 18.82777868894758,
    18.85020279875129, 18.847110177289505, 18.88788559157545,
    18.859306876852244, 18.85980807601389, 18.88051287057267,
    18.832615835636247, 18.855701859006757, 18.891094583347332,
    18.878561304233088, 18.860590245791727, 18.835828841640236,
    18.872059774106695, 18.847033131029836
]
suburb_radius = [1, 0.4, 0.8, 0.5, 0.7, 0.3, 0.3, 0.4, 0.4, 0.5, 0.4, 0.25, 0.3, 0.8, 0.2, 0.25, 0.5]
suburb_probs = [0.06, 0.04, 0.06, 0.08, 0.09, 0.05, 0.05, 0.03, 0.07, 0.07, 0.07, 0.04, 0.06, 0.10, 0.02, 0.05, 0.02]

# Normalize probabilities to sum to 1
suburb_probs = np.array(suburb_probs)
suburb_probs /= suburb_probs.sum()

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic trip data.")
    parser.add_argument("--start_date", type=str, required=True, help="Start date for the data generation (format: YYYY-MM-DD HH:MM)")
    parser.add_argument("--end_date", type=str, required=True, help="End date for the data generation (format: YYYY-MM-DD HH:MM)")
    parser.add_argument("--trips", type=str, required=True, help="Expected number of trips per day")
    return parser.parse_args()

# Helper functions
def generate_random_coordinates_within_radius(lat, long, radius_km):
    """Generate random coordinates within a given radius from a central point."""
    radius_deg = radius_km / 111  # 1 degree is approximately 111 km
    u = np.random.uniform()
    v = np.random.uniform()
    w = radius_deg * np.sqrt(u)
    t = 2 * np.pi * v
    x = w * np.cos(t)
    y = w * np.sin(t)
    new_lat = lat + x
    new_long = long + y
    return new_lat, new_long

def generate_trip_orders(trips_per_day, distribution):
    """Generate synthetic trip orders."""
    current_time = start_time
    data = []

    while current_time <= end_time:
        day_of_week = current_time.strftime("%A")
        hour = current_time.hour
        
        # Determine the time block (hourly)
        time_block = f"{hour}:00-{hour+1}:00"
        if time_block not in distribution[day_of_week]:
            current_time += timedelta(hours=1)
            continue

        # Determine the number of trips for this hour
        num_trips_this_hour = int(distribution[day_of_week][time_block])
        trip_prob_per_minute = num_trips_this_hour / 60  # Each block is 1 hour = 60 minutes

        for minute in range(60):
            if np.random.rand() < trip_prob_per_minute:
                destination_index = np.random.choice(range(len(suburbs)), p=suburb_probs)
                destination_lat, destination_long = generate_random_coordinates_within_radius(
                    suburb_lat[destination_index], suburb_long[destination_index], suburb_radius[destination_index]
                )

                trip_data = {
                    "datetime": current_time,
                    "destination": (destination_lat, destination_long)
                }
                data.append(trip_data)

            current_time += timedelta(minutes=1)

    df = pd.DataFrame(data)
    df.to_csv("../input_data/trip_orders.csv", index=False)
    logger.info("Synthetic trip orders generated and saved to trip_orders.csv")

# Example usage
if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Convert parsed date strings to datetime objects
    start_time = datetime.strptime(args.start_date, "%Y-%m-%d %H:%M")
    end_time = datetime.strptime(args.end_date, "%Y-%m-%d %H:%M")
    # Get the expected number of trips per day from the user
    trips_per_day = int(args.trips)
    
    # Create a distribution for the days of the week and time blocks based on the given charts
    distribution = {
        "Monday": {"8:00-9:00": 0.04, "9:00-10:00": 0.05, "10:00-11:00": 0.08, "11:00-12:00": 0.12, "12:00-13:00": 0.13, 
                   "13:00-14:00": 0.08, "14:00-15:00": 0.05, "15:00-16:00": 0.05, "16:00-17:00": 0.08, "17:00-18:00": 0.10, "18:00-19:00": 0.09, "19:00-20:00": 0.06},
        "Tuesday": {"8:00-9:00": 0.04, "9:00-10:00": 0.05, "10:00-11:00": 0.08, "11:00-12:00": 0.12, "12:00-13:00": 0.13, 
                    "13:00-14:00": 0.08, "14:00-15:00": 0.05, "15:00-16:00": 0.05, "16:00-17:00": 0.08, "17:00-18:00": 0.10, "18:00-19:00": 0.09, "19:00-20:00": 0.06},
        "Wednesday": {"8:00-9:00": 0.04, "9:00-10:00": 0.05, "10:00-11:00": 0.08, "11:00-12:00": 0.12, "12:00-13:00": 0.13, 
                      "13:00-14:00": 0.08, "14:00-15:00": 0.05, "15:00-16:00": 0.05, "16:00-17:00": 0.08, "17:00-18:00": 0.10, "18:00-19:00": 0.09, "19:00-20:00": 0.06},
        "Thursday": {"8:00-9:00": 0.04, "9:00-10:00": 0.05, "10:00-11:00": 0.08, "11:00-12:00": 0.12, "12:00-13:00": 0.13, 
                     "13:00-14:00": 0.08, "14:00-15:00": 0.05, "15:00-16:00": 0.05, "16:00-17:00": 0.08, "17:00-18:00": 0.10, "18:00-19:00": 0.09, "19:00-20:00": 0.06},
        "Friday": {"8:00-9:00": 0.04, "9:00-10:00": 0.05, "10:00-11:00": 0.08, "11:00-12:00": 0.12, "12:00-13:00": 0.13, 
                   "13:00-14:00": 0.08, "14:00-15:00": 0.05, "15:00-16:00": 0.05, "16:00-17:00": 0.08, "17:00-18:00": 0.10, "18:00-19:00": 0.09, "19:00-20:00": 0.06},
        "Saturday": {"8:00-9:00": 0.04, "9:00-10:00": 0.06, "10:00-11:00": 0.1, "11:00-12:00": 0.12, "12:00-13:00": 0.08, 
                     "13:00-14:00": 0.06, "14:00-15:00": 0.05, "15:00-16:00": 0.06, "16:00-17:00": 0.1, "17:00-18:00": 0.1, "18:00-19:00": 0.09},
        "Sunday": {"9:00-10:00": 0.08, "10:00-11:00": 0.1, "11:00-12:00": 0.12, "12:00-13:00": 0.1, "13:00-14:00": 0.08, 
                   "14:00-15:00": 0.06, "15:00-16:00": 0.04, "16:00-17:00": 0.04}
    }
    
    # Adjust distribution based on the number of trips per day
    for day, time_blocks in distribution.items():
        total_prob = sum(time_blocks.values())
        for time_block in time_blocks:
            distribution[day][time_block] = (distribution[day][time_block] / total_prob) * trips_per_day

    generate_trip_orders(trips_per_day, distribution)