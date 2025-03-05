import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import os

# Load player data
player_data_path = "raw_data/players_2024_2025_cleaned.csv"  # Changed data file
df = pd.read_csv(player_data_path)

# Define the target positions and their corresponding position values in 'Pos' column
position_mapping = {
    "Forward": ['FW', 'FW,MF', 'MF,FW'],
    "Midfielder": ['MF', 'FW,MF', 'MF,FW', 'MF,DF', 'DF,MF'],
    "Defender": ['DF', 'MF,DF', 'DF,MF'],
    "Goalkeeper": ['GK']
}

# Define relevant features per position (ADJUST THESE BASED ON YOUR DATA)
position_features = {
    "Forward": ["Gls", "Ast", "Sh", "SoT", "xG"],  # Sample features. You MUST customize these.
    "Midfielder": ["Ast", "Cmp%", "PrgDist", "KP"],  # Sample features. You MUST customize these.
    "Defender": ["Tkl", "Int", "Clr"],  # Sample features. You MUST customize these.
    "Goalkeeper": ["SoT", "GA", "Saves", "Save%"],  # Sample features. You MUST customize these.
}

# Create directories if they don't exist
os.makedirs("scalers", exist_ok=True)
os.makedirs("scaled_data", exist_ok=True)

for position_name, position_values in position_mapping.items():
    print(f"Processing position: {position_name}")

    # Filter the DataFrame for players who have any of the position values in their 'Pos' column
    df_filtered = df[df['Pos'].isin(position_values)].copy()

    # Extract the relevant features for the current position
    features = position_features.get(position_name, [])
    if not features:
        print(f"No features defined for position '{position_name}'. Skipping.")
        continue

    X = df_filtered[features].copy()  # Create a copy to avoid SettingWithCopyWarning

    # Handle missing values by imputing with the mean
    X = X.fillna(X.mean())

    # Initialize and fit the StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler to a pickle file
    scaler_path = f"scalers/{position_name}_scaler.pkl"
    with open(scaler_path, 'wb') as file:
        pickle.dump(scaler, file)
    print(f"Scaler saved to: {scaler_path}")

    # Save the scaled data to a pickle file
    X_scaled_path = f"scaled_data/{position_name}_X_scaled.pkl"
    with open(X_scaled_path, 'wb') as file:
        pickle.dump(X_scaled, file)
    print(f"Scaled data saved to: {X_scaled_path}")

print("Finished processing all positions.")
