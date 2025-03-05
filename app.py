import streamlit as st
import pandas as pd
import joblib
import pickle
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

# Custom CSS for styling
st.markdown(
    """
    <style>
        .main {
            background-color: #d9dce1;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #2E8B57;
            text-align: center;
        }
        .description {
            font-size: 18px;
            text-align: center;
            color: #333333;
        }
        .player-box {
            background-color: #ffffff;
            padding: 10px;
            margin: 5px 0;
            border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            font-weight: bold;
            text-align: center;
            color: #000000;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title of the application
st.markdown("<div class='title'>⚽ Awesome Soccer Project That You Should Rename ⚽</div>", unsafe_allow_html=True)

# Project description
st.markdown("<div class='description'>This awesome project finds the 10 closest players given a subject and calculates the market value. YOU SHOULD ALSO CHANGE THIS</div>", unsafe_allow_html=True)

st.markdown("---")

# Select position to analyze
st.subheader("Select the Position You Want to Analyze")
positions = ["Forward", "Midfielder", "Defender", "Goalkeeper"]
selected_position = st.radio("Choose Position:", positions, horizontal=True)

st.success(f"You selected: {selected_position}")

st.markdown("---")

# Define relevant features per position
position_features = {
    "Forward": ["Gls", "Ast", "Sh", "SoT", "xG"],
    "Midfielder": ["Ast", "Cmp%", "PrgDist", "KP"],
    "Defender": ["Tkl", "Int", "Clr"],
    "Goalkeeper": ["SoT", "GA", "Saves", "Save%"]
}

# Load player data
player_data_path = "raw_data/players_2024_2025_cleaned.csv"
df = pd.read_csv(player_data_path)

# Position mapping
position_mapping = {
    "Forward": ['FW', 'FW,MF', 'MF,FW'],
    "Midfielder": ['MF', 'FW,MF', 'MF,FW', 'MF,DF', 'DF,MF'],
    "Defender": ['DF', 'MF,DF', 'DF,MF'],
    "Goalkeeper": ['GK']
}

# Add a 'Position' column to the DataFrame based on the mapping
def assign_position(pos_value):
    for position_name, position_values in position_mapping.items():
        if pos_value in position_values:
            return position_name
    return None  # Or 'Other' if you want to handle unmatched positions

df['Position'] = df['Pos'].apply(assign_position)

# Filter players by position
df_filtered = df[df["Position"] == selected_position].copy() #Change the filter

# Divide page into two sections
col1, col2 = st.columns(2)

# Load scaler and scaled data based on the selected position
scaler_path = f"scalers/{selected_position}_scaler.pkl"
X_scaled_path = f"scaled_data/{selected_position}_X_scaled.pkl"

scaler = None
X_scaled = None

try:
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    with open(X_scaled_path, 'rb') as file:
        X_scaled = pickle.load(file)
    st.success(f"Scaler and Scaled Data loaded for {selected_position} from pickle files!")
except FileNotFoundError:
    st.warning(f"Scaler or scaled data pickle files not found for {selected_position}. Please ensure these files exist.")
    scaler = None  # Set scaler to None to avoid further errors
    X_scaled = None

# Left column: User input for new player
with col1:
    st.subheader("Input New Player Features")
    input_data = {}
    for feature in position_features[selected_position]:
        input_data[feature] = st.number_input(f"{feature.replace('_', ' ').capitalize()}", min_value=0.0, step=0.1)

# Right column: Display results
with col2:
    st.subheader("Results & Similar Players")

    if scaler is not None and X_scaled is not None:  # Check if scaler and X_scaled are loaded

        # Convert input data to a DataFrame and scale it
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)

        # Calculate distances to all players in the dataset
        distances = pairwise_distances(X_scaled, input_scaled, metric='euclidean').flatten()

        # Get the indices of the 10 most similar players
        similar_player_indices = np.argsort(distances)[:10]

        # Get the names of the most similar players
        similar_players = df_filtered.iloc[similar_player_indices]["Player"].tolist() #Change "player_name" with "Player"

        # Calculate average market value of similar players (using random values for demonstration)
        market_value = round(random.uniform(1.0, 100.0), 2)  # Random market value in millions

        st.write(f"**Predicted Market Value:** €{market_value}m")
        st.write("**Similar Players:**")

        # Display players in styled boxes
        for player in similar_players:
            st.markdown(
                f"""
                <div class='player-box'>
                    {player}
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.write("Please check that the scaler and scaled data files exist.")

# Display attributes comparison using a radial plot
st.markdown("---")
st.subheader("Player Attributes Comparison - Radial Plot")

# Prepare data for the radial plot
comparison_columns = position_features[selected_position]
comparison_df = df_filtered[df_filtered["Player"].isin(similar_players)][["Player"] + comparison_columns] if scaler is not None and X_scaled is not None else pd.DataFrame()  # Ensure dataframe is not empty

if not comparison_df.empty:
    input_player_df = pd.DataFrame([{**{"Player": "Input Player"}, **input_data}])
    comparison_df = pd.concat([input_player_df, comparison_df], ignore_index=True)

    # Create radial plot
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})
    angles = np.linspace(0, 2 * np.pi, len(comparison_columns), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    for i, row in comparison_df.iterrows():
        values = row[comparison_columns].tolist()
        values += values[:1]  # Close the circle
        ax.plot(angles, values, label=row["Player"], linewidth=2)
        ax.fill(angles, values, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(comparison_columns)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    st.pyplot(fig)

else:
    st.write("No players to compare. Please check that the data and models are loaded correctly.")
