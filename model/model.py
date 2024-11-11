import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib  # For saving the trained model and scaler
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print the current working directory
logger.info(f"Current working directory: {os.getcwd()}")

# Define absolute paths for files
base_dir = os.getcwd()  # Get the current working directory
file1 = os.path.join(base_dir, 'api_database_1.csv')
file2 = os.path.join(base_dir, 'api_database_2.csv')
combined_output_file = os.path.join(base_dir, 'api_data.csv')
train_output_file = os.path.join(base_dir, 'api_data_train.csv')
test_output_file = os.path.join(base_dir, 'api_data_test.csv')
model_file = os.path.join(base_dir, 'trained_model.pkl')
scaler_file = os.path.join(base_dir, 'output_scaler.pkl')

# Step 1: Combine the Two Databases
def combine_databases(file1, file2, output_file):
    """Combine two CSV files into one."""
    # Check if input files exist
    if not os.path.exists(file1):
        logger.error(f"File {file1} does not exist.")
        raise FileNotFoundError(f"File {file1} does not exist.")
    if not os.path.exists(file2):
        logger.error(f"File {file2} does not exist.")
        raise FileNotFoundError(f"File {file2} does not exist.")

    try:
        df1 = pd.read_csv(file1)
        logger.info(f"File {file1} read successfully. First 5 rows:\n{df1.head()}")  # Debugging output

        df2 = pd.read_csv(file2)
        logger.info(f"File {file2} read successfully. First 5 rows:\n{df2.head()}")  # Debugging output
        
        # Concatenate the DataFrames
        combined_df = pd.concat([df1, df2], ignore_index=True)
        logger.info(f"DataFrames combined successfully. Combined DataFrame shape: {combined_df.shape}")

        # Save combined data to a new CSV file
        combined_df.to_csv(output_file, index=False)
        logger.info(f"Combined data saved to {output_file}")

        return combined_df

    except Exception as e:
        logger.error(f"An error occurred while combining databases: {e}")
        raise

# Step 2: Preprocessing
def preprocess_data(df):
    """Preprocess the data by handling missing values, normalizing/standardizing, etc."""
    # Handling missing values (if any)
    df = df.dropna()  # Dropping rows with missing values, can be replaced with other strategies (mean, median, etc.)
    
    # Extract numerical features to be scaled
    features_to_scale = ['distance_km', 'duration_min', 'elev_diff_m']
    
    # Standardize numerical features (distance, duration, elevation difference)
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    # Save the scaler for inverse transforming predictions later
    try:
        joblib.dump(scaler, scaler_file)
        logger.info(f"Data has been standardized and scaler saved to {scaler_file}.")
    except Exception as e:
        logger.error(f"Error saving scaler to {scaler_file}: {e}")
        raise

    # Confirm that the scaler is saved correctly
    if not os.path.exists(scaler_file):
        logger.error(f"Scaler file {scaler_file} could not be found after saving.")
        raise FileNotFoundError(f"Scaler file {scaler_file} could not be found after saving.")
    
    logger.info(f"Scaler saved successfully at {scaler_file}")  # Print absolute path for confirmation
    
    return df

# Step 3: Split the Data
def split_data(df, train_output_file, test_output_file, test_size=0.2, random_state=42):
    """Split the data into training and testing sets and save them to CSV files."""
    # Splitting the data
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Save the train and test datasets to CSV
    train_df.to_csv(train_output_file, index=False)
    test_df.to_csv(test_output_file, index=False)
    
    logger.info(f"Training data saved to {train_output_file}")
    logger.info(f"Testing data saved to {test_output_file}")

# Step 4: Load Data
def load_data(train_file, test_file):
    """Load the training and testing data from CSV files."""
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    return train_df, test_df

# Step 5: Feature Engineering
def extract_datetime_features(df):
    """Extract features from the datetime column for modeling."""
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour  # Hour of the day
    df['day_of_week'] = df['datetime'].dt.weekday  # Day of the week (Monday=0, Sunday=6)
    return df

def split_coordinates(df):
    """Split the destination_coordinates column into latitude and longitude columns."""
    coords = df['destination_coordinates'].str.strip('()').str.split(', ', expand=True)
    df['destination_lat'] = coords[0].astype(float)
    df['destination_lon'] = coords[1].astype(float)
    return df

# Step 6: Prepare Data for Training
def prepare_data(train_df, test_df):
    """Prepare data for training and testing by separating features and target variables."""
    # Extract datetime features
    train_df = extract_datetime_features(train_df)
    test_df = extract_datetime_features(test_df)
    
    # Split destination coordinates into separate columns
    train_df = split_coordinates(train_df)
    test_df = split_coordinates(test_df)
    
    # Define the features (X) and the target variables (y)
    feature_columns = ['hour', 'day_of_week', 'destination_lat', 'destination_lon']
    target_columns = ['distance_km', 'duration_min', 'elev_diff_m']
    
    # Separate input features and targets
    X_train = train_df[feature_columns]
    print(X_train)
    y_train = train_df[target_columns]
    X_test = test_df[feature_columns]
    y_test = test_df[target_columns]
    
    return X_train, y_train, X_test, y_test

# Step 7: Train Model
def train_random_forest(X_train, y_train):
    """Train a Random Forest Regressor model for multi-output regression."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logger.info("Model training completed.")
    return model

# Step 8: Evaluate Model
def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance on the test set for multiple outputs."""
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Load the scaler to inverse transform the predictions
    if not os.path.exists(scaler_file):
        logger.error(f"Scaler file {scaler_file} not found for loading.")
        raise FileNotFoundError(f"Scaler file {scaler_file} not found for loading.")
    
    scaler = joblib.load(scaler_file)
    y_pred_original = scaler.inverse_transform(y_pred)
    y_test_original = scaler.inverse_transform(y_test)

    # Calculate evaluation metrics for each target
    metrics = {}
    for i, target in enumerate(['distance_km', 'duration_min', 'elev_diff_m']):
        mae = mean_absolute_error(y_test_original[:, i], y_pred_original[:, i])
        mse = mean_squared_error(y_test_original[:, i], y_pred_original[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_original[:, i], y_pred_original[:, i])
        
        metrics[target] = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }
        
        # Print the evaluation metrics for each target
        logger.info(f"\nModel Evaluation Metrics for {target}:")
        logger.info(f"Mean Absolute Error (MAE): {mae:.4f}")
        logger.info(f"Mean Squared Error (MSE): {mse:.4f}")
        logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        logger.info(f"R-squared (R2): {r2:.4f}")

    return metrics

# Step 9: Save Model
def save_model(model, model_file='trained_model.pkl'):
    """Save the trained model to a file."""
    joblib.dump(model, model_file)
    logger.info(f"Model saved to {model_file}.")

# Main Function to Execute All Steps
def main():
    # Check if train and test files exist; if not, preprocess and split the data
    if not os.path.exists(train_output_file) or not os.path.exists(test_output_file):
        # Combine the two databases
        combined_df = combine_databases(file1, file2, combined_output_file)
        
        # Preprocess the combined data
        preprocessed_df = preprocess_data(combined_df)
        
        # Split the data into training and testing sets
        split_data(preprocessed_df, train_output_file, test_output_file)
    
    # Load the training and testing data
    train_df, test_df = load_data(train_output_file, test_output_file)
    
    # Prepare the data for training and testing
    X_train, y_train, X_test, y_test = prepare_data(train_df, test_df)
    
    # Train the Random Forest Regressor model
    model = train_random_forest(X_train, y_train)
    
    # Evaluate the model performance
    evaluate_model(model, X_test, y_test)
    
    # Save the trained model
    save_model(model)

if __name__ == "__main__":
    main()