#!/bin/bash

# Array of dates to iterate through
dates=("2024-11-04" "2024-11-05" "2024-11-06" "2024-11-07" "2024-11-08" "2024-11-09" "2024-11-10")
# dates=("2024-11-04")

# Loop through each day
for date in "${dates[@]}"
do
    # Set the start and end times for the day
    start_date="${date} 08:00"
    end_date="${date} 20:00"

    echo "Running simulations for date range: $start_date to $end_date"

    # Loop through trips incrementing
    trips=200
    while [ $trips -le 200 ]
    do
        # Calculate fleet_size, pool_size, and chargers based on trips
        fleet_size=1
        pool_size=1
        chargers=1

        echo "Running simulation for trips=$trips, fleet_size=$fleet_size, pool_size=$pool_size, chargers=$chargers"

        # Call synthetic_data_generator.py with the current date and number of trips
        python3 synthetic_data_generator.py --start_date "$start_date" --end_date "$end_date" --trips $trips

        # Call static_simulator.py with fleet_size, pool_size, and chargers
        python3 static_simulator.py --fleet_size $fleet_size --pool_size $pool_size --chargers $chargers

        # Call static_plots.py
        python3 static_plots.py

        # Increment trips by 25
        trips=$((trips + 15))
    done
done

# After the loop completes, call preprocess.py
echo "Calling post_processing.py after all iterations are complete."
python3 post_processing.py
python3 post_plots.py