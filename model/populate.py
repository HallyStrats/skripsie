import pandas as pd
import googlemaps
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Maps API setup
with open("api.txt", "r") as file:
    API_KEY = file.read().strip()
gmaps = googlemaps.Client(key=API_KEY)

def parse_coordinates(coord_str):
    """Parse coordinates from the CSV format."""
    coord_str = coord_str.strip("()")
    lat, long = map(float, coord_str.split(", "))
    return lat, long

def get_trip_details(origin, destination, departure_time):
    """Get trip distance, duration, and elevation difference from Google Maps API."""
    try:
        # Ensure departure_time is in datetime format
        if isinstance(departure_time, str):
            departure_time = datetime.strptime(departure_time, "%Y-%m-%d %H:%M:%S")

        # Get directions
        directions_result = gmaps.directions(
            origin, destination, mode="driving", departure_time=departure_time
        )
        
        if directions_result:
            leg = directions_result[0]['legs'][0]
            trip_distance = leg['distance']['value'] / 1000  # Convert meters to kilometers
            trip_duration = leg['duration']['value'] / 60  # Convert seconds to minutes

            # Get elevation details
            elev_origin = get_elevation(origin[0], origin[1])
            elev_destination = get_elevation(destination[0], destination[1])

            if elev_origin is None or elev_destination is None:
                return None, None, None

            elev_difference = elev_destination - elev_origin

            return trip_distance, trip_duration, elev_difference
        else:
            logger.warning(f"No directions found for {origin} to {destination} at {departure_time}. Full response: {directions_result}")
            return None, None, None
    except googlemaps.exceptions.ApiError as e:
        logger.error(f"Google Maps API error: {e}")
        return None, None, None
    except googlemaps.exceptions.HTTPError as e:
        logger.error(f"HTTP Error: {e}")
        return None, None, None
    except googlemaps.exceptions.Timeout as e:
        logger.error(f"Timeout error: {e}")
        return None, None, None
    except googlemaps.exceptions.TransportError as e:
        logger.error(f"Transport error: {e}")
        return None, None, None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None, None, None

def get_elevation(lat, long):
    """Get the elevation for a given latitude and longitude from Google Maps Elevation API."""
    try:
        elevation_result = gmaps.elevation((lat, long))
        
        if elevation_result:
            elevation = elevation_result[0]['elevation']
            return elevation
        else:
            logger.warning(f"No elevation data found for {lat}, {long}. Full response: {elevation_result}")
            return None
    except googlemaps.exceptions.ApiError as e:
        logger.error(f"Google Maps API error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

def process_trip_data(input_file, output_file):
    """Process trip data from input CSV and save calculated trip details to output CSV."""
    origin_lat, origin_long = -33.938299405675814, 18.858465193281816  # Origin coordinates
    origin = (origin_lat, origin_long)
    
    # Read input CSV file
    trip_orders = pd.read_csv(input_file)
    trip_orders['destination'] = trip_orders['destination'].apply(parse_coordinates)
    
    # Prepare list to store API results
    api_data_log = []

    for index, trip in trip_orders.iterrows():
        datetime_value = trip['datetime']
        destination = trip['destination']

        # Calculate trip details using Google Maps API
        trip_distance, trip_duration, elev_diff = get_trip_details(origin, destination, datetime_value)

        if trip_distance is None or trip_duration is None or elev_diff is None:
            logger.warning(f"Skipping trip due to missing data for {destination} at {datetime_value}.")
            continue

        # Log API data
        api_data_log.append({
            "datetime": datetime_value,
            "destination_coordinates": destination,
            "distance_km": trip_distance,
            "duration_min": trip_duration,
            "elev_diff_m": elev_diff,
        })
    
    # Convert the list of dictionaries to a DataFrame
    api_data_df = pd.DataFrame(api_data_log)

    # Save the results to the output CSV file
    api_data_df.to_csv(output_file, index=False)
    logger.info(f"API data has been successfully saved to {output_file}")

# Run the script
input_csv = 'trip_orders.csv'
output_csv = 'api_database.csv'
process_trip_data(input_csv, output_csv)