import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import joblib
import os
import random

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
    </style>
    """,
    unsafe_allow_html=True,
)

# Title of the application
### CHANGE HERE
st.markdown("<div class='title'>⚽ Awesome Soccer Project That You Should Rename ⚽</div>", unsafe_allow_html=True)

# Project description
### CHANGE HERE
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
df_filtered = df[df["field_position"] == selected_position]

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
    for player in similar_players:
        st.write(f"- {player}")
