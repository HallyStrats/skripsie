
# Electric Micromobility Simulation: Planning for electric motorbikes with swappable batteries in last-mile delivery applications

### Overview

This codebase simulates electric bike operations to optimize fleet management, battery usage, and charging infrastructure. 
It generates synthetic trip data, allocates resources, and trains models to predict trip details. The project includes visualisation tools and supports analysis through detailed simulations.

### Features

- **Synthetic Data Generation**: Generates trip data with randomized geographic coordinates and timestamps.
- **Resource Allocation Simulation**: Simulates bike, battery, and charger usage.
- **Battery Management**: Tracks battery states, charging cycles, and swap events.
- **Data Preprocessing and Model Training**: Processes trip data and trains a model to predict trip-related metrics.
- **Plotting and Visualisation**: Creates pre- and post-simulation plots to assess fleet and battery efficiency.

### File Descriptions

1. **Data and Model Files**:
   - **`populate.py`**  
     Uses the Google Maps API to fetch trip data (distance, duration, and elevation difference) based on synthetic trip orders.
   - **`preprocess.py`**  
     Combines data files, preprocesses trip data, and splits it into training and testing sets.
   - **`model.py`**  
     Trains a model (Random Forest) to predict trip metrics (distance, duration, elevation) using preprocessed data.

2. **Simulation and Plotting Files**:
   - **`synthetic_data_generator.py`**  
     Generates synthetic trip data and saves it to `trip_orders.csv`.
   - **`static_simulator.py`**  
     Core simulation engine, managing trip allocations, battery usage, and charger utilization.
   - **`post_processing.py`**  
     Processes simulation results and prepares data for visual analysis.
   - **`static_plots.py`**  
     Creates pre-simulation plots to provide insights on resource requirements.
   - **`post_plots.py`**  
     Generates post-simulation visualizations to assess efficiency and resource usage.
   - **`run_static_simulation.sh`**  
     Automates simulation runs, iterating over different fleet configurations and logging the results.

3. **Utility File**:
   - **`helper_functions.py`**  
     Provides utility functions for data calculations and transformations.

### Installation

**Requirements**:
- Python 3.8+
- Packages: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `joblib`, `googlemaps`

Install dependencies using:
```bash
pip install -r requirements.txt
```

### Usage

1. **Generate Synthetic Data**:
   ```bash
   python synthetic_data_generator.py --start_date "YYYY-MM-DD HH:MM" --end_date "YYYY-MM-DD HH:MM" --trips N
   ```
   Replace `N` with the desired number of daily trips.

2. **Populate Trip Data**:
   ```bash
   python populate.py
   ```
   This fetches trip metrics from Google Maps API based on generated synthetic trip orders.

3. **Preprocess Data**:
   ```bash
   python preprocess.py
   ```
   Combines, preprocesses, and splits the data for model training.

4. **Train the Model**:
   ```bash
   python model.py
   ```
   Trains a model to predict trip metrics and evaluates it on the test set.

5. **Run Simulation**:
   Execute the simulation using:
   ```bash
   ./run_static_simulation.sh
   ```

6. **View Results**:
   - Pre-simulation plots: `static_plots.py`
   - Post-simulation plots: `post_processing.py` and `post_plots.py`

### Outputs

The simulation generates:
- **CSV Files**: Trip logs, battery states, charger cycles, training and testing data.
- **Plots**: Visualizations of resource usage, trip distributions, and operational efficiency.

### Notes

To explore different fleet sizes, battery pools, and chargers, adjust parameters in `run_static_simulation.sh`. Ensure `api.txt` contains a valid Google Maps API key for `populate.py`.
