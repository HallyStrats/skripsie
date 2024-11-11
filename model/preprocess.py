import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Step 1: Combine the Two Databases
def combine_databases(file1, file2, output_file):
    """Combine two CSV files into one."""
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Concatenate the DataFrames
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Save combined data to a new CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")
    return combined_df

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
    
    print("Data has been standardized.")
    
    return df

# Step 3: Split the Data
def split_data(df, train_output_file, test_output_file, test_size=0.2, random_state=42):
    """Split the data into training and testing sets and save them to CSV files."""
    # Splitting the data
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Save the train and test datasets to CSV
    train_df.to_csv(train_output_file, index=False)
    test_df.to_csv(test_output_file, index=False)
    
    print(f"Training data saved to {train_output_file}")
    print(f"Testing data saved to {test_output_file}")

# Main Function
def main():
    # File names
    file1 = 'api_database_1.csv'
    file2 = 'api_database_2.csv'
    combined_output_file = 'api_data.csv'
    train_output_file = 'api_data_train.csv'
    test_output_file = 'api_data_test.csv'
    
    # Combine the two databases
    combined_df = combine_databases(file1, file2, combined_output_file)
    
    # Preprocess the combined data
    preprocessed_df = preprocess_data(combined_df)
    
    # Split the data into training and testing sets
    split_data(preprocessed_df, train_output_file, test_output_file)

# Execute the main function
if __name__ == "__main__":
    main()