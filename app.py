import streamlit as st
import pandas as pd
import joblib
import pickle
import os
import random
import matplotlib.pyplot as plt
import numpy as np

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
    "Forward": ["goals", "assists", "total_attempts", "attempts_on_target", "dribbles", "offsides"],
    "Midfielder": ["passes_attempted", "passes_completed", "assists", "crosses_attempted", "crossing_accuracy(%)", "distance_covered(km/h)"],
    "Defender": ["tackles", "tackles_won", "clearance_attempted", "balls_recovered", "fouls_committed", "yellow_cards"],
    "Goalkeeper": ["saves", "goals_conceded", "clean_sheets", "saves_on_penalty", "punches_made"]
}

# Divide page into two sections
col1, col2 = st.columns(2)

# Left column: User input for new player
with col1:
    st.subheader("Input New Player Features")
    input_data = {}
    for feature in position_features[selected_position]:
        input_data[feature] = st.number_input(f"{feature.replace('_', ' ').capitalize()}", min_value=0.0, step=0.1)

# Load models dynamically
classification_model_path = f"models/{selected_position}_classification.pkl"
regression_model_path = f"models/{selected_position}_regression.pkl"

classification_model = None
regression_model = None

if os.path.exists(classification_model_path):
    classification_model = joblib.load(classification_model_path)
if os.path.exists(regression_model_path):
    regression_model = joblib.load(regression_model_path)

# Load player data
player_data_path = "raw_data/all_players_with_valuations.csv"
df = pd.read_csv(player_data_path)

# Filter players by position
df_filtered = df[df["field_position"] == selected_position].copy()

# Generate predictions or use random values
if classification_model and regression_model:
    input_df = pd.DataFrame([input_data])
    similar_players = classification_model.predict(input_df)[0]
    market_value = regression_model.predict(input_df)[0]
else:
    similar_players = random.sample(df_filtered["player_name"].tolist(), min(10, len(df_filtered)))
    market_value = round(random.uniform(1.0, 100.0), 2)  # Random market value in millions

# Right column: Display results
with col2:
    st.subheader("Results & Similar Players")
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

# Display attributes comparison using a radial plot
st.markdown("---")
st.subheader("Player Attributes Comparison - Radial Plot")

# Prepare data for the radial plot
comparison_columns = position_features[selected_position]
comparison_df = df_filtered[df_filtered["player_name"].isin(similar_players)][["player_name"] + comparison_columns]
input_player_df = pd.DataFrame([{**{"player_name": "Input Player"}, **input_data}])
comparison_df = pd.concat([input_player_df, comparison_df], ignore_index=True)

# Create radial plot
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})
angles = np.linspace(0, 2 * np.pi, len(comparison_columns), endpoint=False).tolist()
angles += angles[:1]  # Close the circle

for i, row in comparison_df.iterrows():
    values = row[comparison_columns].tolist()
    values += values[:1]  # Close the circle
    ax.plot(angles, values, label=row["player_name"], linewidth=2)
    ax.fill(angles, values, alpha=0.2)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(comparison_columns)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
st.pyplot(fig)
