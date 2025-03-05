import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Custom CSS for styling
st.markdown(
    """
    <style>
        /* Overall Page Background - Sleek Modern Color Palette */
        .main {
            background-color: ##192a56; 
            font-family: 'Arial', sans-serif; /* Modern font */
        }

        /* Title Styling */
        .title {
            font-size: 42px; /* Slightly larger title */
            font-weight: 800; /* Black font weight */
            color: #F5F1E3; /* Off-white text for contrast */
            text-align: center;
            margin-bottom: 20px; /* Spacing below title */
            text-transform: uppercase; /* Capitalize all letters */
            letter-spacing: 3px; /* Spacing between each letter */
            text-shadow: 2px 2px 2px rgba(0, 0, 0, 0.35); /* Added soft shadow */
        }

        /* Subheaders and Text Headings */ 
        h1, h2, h3, h4, h5 {
            color: #F5F1E3; /* Same color as title */
            font-family: 'Arial', sans-serif; /* Same font as title */
            font-size: 26px; /* Match other headers' size */
        }

        /* Links */
        a {
            color: #F5F1E3 !important; /* Match links to off-white color */
            text-decoration: none; /* Remove underline for links */
        }

        /* Text Shadow for Headers and Text */
        h1, h2, h3, h4, h5, p, div {
            text-shadow: 2px 2px 2px rgba(0, 0, 0, 0.35); /* Soft shadow */
        }

        /* Custom Message Styling (For Success Box) */
        .custom-success {
            background-color: #e1b12c !important;  /* Match the 'You selected' box background color */
            color: white !important;  /* Text color */
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
            font-size: 16px;
            font-weight: bold;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2); /* Subtle shadow for better aesthetic */
        }

        /* Styled Player Cards */
        .player-box {
            background-color: #e1b12c; /* Updated player box background color */
            color: white; /* Player names in white */
            padding: 15px;
            margin: 10px 0; /* More separation between cards */
            border-radius: 15px; /* Rounded corners */
            box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.2); /* Professional box shadow */
            font-weight: bold;
            text-align: center;
            font-size: 18px;
            transition: transform 0.3s, box-shadow 0.3s; /* Animation for hover */
        }

        /* Hover Effect on Player Cards */
        .player-box:hover {
            transform: translateY(-5px); /* Slight hover lift */
            box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.4); /* Enhanced shadow on hover */
        }

        /* Add Custom Button Styles */
        .streamlit-button {
            background-color: #F5F1E3 !important;
            color: #1B9AAA !important;
            font-size: 16px !important;
            font-weight: bold !important;
            text-transform: uppercase;
            border-radius: 30px !important; /* Rounded buttons */
            padding: 10px 20px !important; /* Modern spacing */
            transition: 0.4s ease-in-out all; /* Smooth animation */
            border: 2px solid #F5F1E3 !important;
        }
        
        /* On hover - reverse color scheme for buttons */
        .streamlit-button:hover {
            background-color: #1B9AAA !important;
            color: #F5F1E3 !important;
            border: 2px solid #F5F1E3 !important;
        }

        /* Footer Styling - Minimal Information */
        .footer {
            position: fixed; /* Sticks the footer at the bottom */
            bottom: 0;
            width: 100%; /* Full width of page */
            background-color: #113F67; /* Dark blue for contrast */
            color: white;
            text-align: center;
            padding: 10px 0;
            font-size: 13px;
            font-weight: normal;
        }
        
            /* Highlight Box Styling with Desired Background Color */
        .highlight-box {
            background-color: #e1b12c; /* Your Desired Color */
            color: white; /* Text on Highlight Box */
            padding: 15px;
            margin: 10px 0; /* Adds spacing between sections */
            border-radius: 8px; /* Rounded corners */
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2); /* Subtle shadow for better aesthetic */
            font-weight: bold;
            font-size: 18px;
        }

        /* Hover Effect for Highlight Boxes */
        .highlight-box:hover {
            filter: brightness(1.05);  /* Slight increase in brightness */
            transition: all 0.3s ease-in-out;
        }
    </style>
    """,
     unsafe_allow_html=True,
)

# Title of the application
st.markdown("<div class='title'>⚽ PlayerLens ⚽</div>", unsafe_allow_html=True)

# Project description
st.markdown(
    "<div style='text-align: center; font-size: 18px; color: #F5F1E3; line-height: 1.6;'>"
    "Discover Similar Players Based on Performance, Position and Market Value."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# Load player data
@st.cache_data
def load_data():
    df = pd.read_csv("raw_data/players_2024_2025_cleaned.csv")
    # Convert market value to millions and create log target
    df['log_target'] = np.log1p(df['Market Value Euros'])
    return df

df = load_data()

# Define important features for regression model
important_features = [
    'Age', 'Min', 'Starts', '90s', 'MP',
    'CPA', 'Rec', 'G/SoT', 'GCA', 'Carries',
    'SoT', 'SCA', 'Touches', 'PPA'
]

# Ensure all important features exist, fill missing with 0
for feature in important_features:
    if feature not in df.columns:
        df[feature] = 0
    else:
        df[feature] = df[feature].fillna(0)

# Create a simple model for prediction instead of loading failed model
@st.cache_resource
def create_model():
    try:
        X = df[important_features]
        y = df['log_target']
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    except Exception as e:
        st.error(f"Error creating model: {str(e)}")
        # Even simpler fallback
        return None

# Create the model
regression_model = create_model()

# Simplify position categories
def simplify_position(pos):
    if pd.isna(pos):
        return "Unknown"
    elif "GK" in pos:
        return "Goalkeeper"
    elif "DF" in pos and "MF" not in pos and "FW" not in pos:
        return "Defender"
    elif "MF" in pos and "FW" not in pos and "DF" not in pos:
        return "Midfielder"
    elif "FW" in pos and "MF" not in pos and "DF" not in pos:
        return "Forward"
    elif "FW" in pos:
        return "Forward"
    elif "MF" in pos:
        return "Midfielder"
    else:
        return "Defender"

# Add simplified position column
df["SimplePosition"] = df["Pos"].apply(simplify_position)

# Function to format market value based on amount
def format_market_value(value_in_millions):
    if value_in_millions >= 1:
        # For values over 1 million, show as €X.XXM
        return f"€{value_in_millions:.2f}M"
    else:
        # For values under 1 million, show as €XXXK
        value_in_thousands = int(value_in_millions * 1000)
        return f"€{value_in_thousands}K"

# Select position to analyze
st.subheader("Select the Position You Want to Analyse:")
positions = ["Forward", "Midfielder", "Defender", "Goalkeeper"]
selected_position = st.radio("Choose Position:", positions, horizontal=True)

st.markdown("---")

# Filter players by position
filtered_df = df[df["SimplePosition"] == selected_position].copy()

# Define relevant features per position
position_features = {
    "Forward": ["Gls", "Ast", "Sh", "SoT", "G/SoT", "xG", "xAG", "PrgC", "PrgP", "PrgR"],
    "Midfielder": ["Gls", "Ast", "Cmp", "Att", "Cmp%", "PrgP", "KP", "SCA", "PrgC", "Tkl+Int"],
    "Defender": ["Tkl", "TklW", "Int", "Clr", "Blocks_stats_defense", "Cmp%", "Ast", "Gls", "PrgP", "Tkl+Int"],
    "Goalkeeper": ["GA90", "Save%", "CS", "CS%", "PSxG", "/90", "Cmp%_stats_keeper_adv", "Launch%"]
}

# Define human-readable feature labels
feature_labels = {
    "Gls": "Goals",
    "Ast": "Assists",
    "Sh": "Shots",
    "SoT": "Shots on Target",
    "G/SoT": "Goals per Shot on Target",
    "xG": "Expected Goals",
    "xAG": "Expected Assisted Goals",
    "PrgC": "Progressive Carries",
    "PrgP": "Progressive Passes",
    "PrgR": "Progressive Passes Received",
    "Cmp": "Passes Completed",
    "Att": "Passes Attempted",
    "Cmp%": "Pass Completion %",
    "KP": "Key Passes",
    "SCA": "Shot-Creating Actions",
    "GCA": "Goal-Creating Actions",
    "Tkl": "Tackles",
    "TklW": "Tackles Won",
    "Int": "Interceptions",
    "Clr": "Clearances",
    "Blocks_stats_defense": "Blocks",
    "Tkl+Int": "Tackles + Interceptions",
    "GA90": "Goals Against per 90",
    "Save%": "Save Percentage",
    "CS": "Clean Sheets",
    "CS%": "Clean Sheet Percentage",
    "PSxG": "Post-Shot Expected Goals",
    "/90": "Post-Shot Expected Goals per 90",
    "Cmp%_stats_keeper_adv": "Goalkeeper Pass Completion %",
    "Launch%": "Long Pass Percentage",
    "Carries": "Carries",
    "Touches": "Touches",
    "CPA": "Carries into Penalty Area",
    "Rec": "Passes Received",
    "PPA": "Passes into Penalty Area"
}

# Left column: Player selection
st.subheader("Player Analysis")

# Get list of players for the selected position
players_list = filtered_df["Player"].unique().tolist()
players_list.sort()

selected_player = st.selectbox("Select Player:", players_list)
player_data = filtered_df[filtered_df["Player"] == selected_player].iloc[0]

# Capitalize player name for display
display_player_name = selected_player.title()

# Player info container with better styling
st.markdown(f"""
<div style="background-color: #e1b12c; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #2E8B57;">
    <div style="font-size: 22px; font-weight: bold; margin-bottom: 10px;">{display_player_name}</div>
    <div style="display: flex; flex-wrap: wrap;">
        <div style="margin-right: 30px;">
            <span style="font-weight: bold;">Team:</span> {player_data['Squad']}
        </div>
        <div style="margin-right: 30px;">
            <span style="font-weight: bold;">League:</span> {player_data['Comp']}
        </div>
        <div>
            <span style="font-weight: bold;">Age:</span> {player_data['Age']}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Display player metrics in a more organized grid with better styling
st.markdown("<div style='margin-top: 20px; margin-bottom: 15px; font-size: 26px;'><strong>Performance Metrics</strong></div>", unsafe_allow_html=True)

# Determine the number of columns (2 for small screens, more for larger ones)
num_cols = 2
metrics_cols = st.columns(num_cols)

# Get the features for the selected position
position_feats = position_features[selected_position]

for i, feature in enumerate(position_feats):
    if feature in player_data and not pd.isna(player_data[feature]):
        with metrics_cols[i % num_cols]:
            label = feature_labels.get(feature, feature)
            value = player_data[feature]

            # Format value based on type
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)

            # Use a custom styled metric
            st.markdown(f"""
            <div style="background-color: #e1b12c; padding: 10px 15px; margin-bottom: 10px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="color: white; font-size: 20px;">{label}</div>
                <div style="font-size: 22px; font-weight: bold; color: white;">{formatted_value}</div>
            </div>
            """, unsafe_allow_html=True)

# Create input data for similarity calculation
input_data = {}
for feature in position_features[selected_position]:
    if feature in player_data:
        input_data[feature] = player_data[feature]
    else:
        input_data[feature] = 0

# Add additional player information for regression model
input_data['Age'] = player_data['Age']
input_data['Min'] = player_data['Min']
input_data['Starts'] = player_data['Starts']
input_data['90s'] = player_data['90s']
input_data['MP'] = player_data['MP']

# Function to find similar players
def find_similar_players(input_data, df, position, num_players=10):
    # Select relevant features and players with sufficient minutes
    min_minutes = 270  # At least 3 full games
    position_df = df[(df["SimplePosition"] == position) & (df["Min"] >= min_minutes)].copy()

    # Ensure all required columns exist
    features = position_features[position]
    for feature in features:
        if feature not in position_df.columns:
            position_df[feature] = 0

    # Handle NaN values
    position_df = position_df.fillna(0)

    # Extract feature matrix
    X = position_df[features].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create input vector
    input_vector = np.array([[input_data[feature] for feature in features]])
    input_scaled = scaler.transform(input_vector)

    # Find nearest neighbors
    knn = NearestNeighbors(n_neighbors=min(num_players + 1, len(position_df)))
    knn.fit(X_scaled)
    distances, indices = knn.kneighbors(input_scaled)

    # Get similar players
    similar_players = []
    for i in range(min(num_players, len(indices[0]))):
        idx = indices[0][i]
        player = position_df.iloc[idx]
        similar_players.append({
            "name": player["Player"],
            "team": player["Squad"],
            "league": player["Comp"],
            "distance": distances[0][i],
            "market_value": np.expm1(player["log_target"]) / 1000000 if not pd.isna(player["log_target"]) else 0,
            "data": {feature: player[feature] for feature in features if feature in player}
        })

    return similar_players

# Function to predict market value using the regression model
def predict_market_value(player_features, model):
    # Create a DataFrame with required features
    prediction_df = pd.DataFrame([player_features])

    # Make sure all required features are present
    for feature in important_features:
        if feature not in prediction_df.columns:
            prediction_df[feature] = 0

    # Select only the important features in the correct order
    input_df = prediction_df[important_features]

    # Replace any NaN values with 0
    input_df = input_df.fillna(0)

    # Make prediction
    try:
        if model is not None:
            log_predicted_value = model.predict(input_df)[0]
            # Convert from log scale back to millions
            return np.expm1(log_predicted_value) / 1000000
        else:
            # If no model, use the average value of similar players
            similar_players = find_similar_players(player_features, df, selected_position)
            values = [p["market_value"] for p in similar_players if p["market_value"] > 0]
            return sum(values) / len(values) if values else 0
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return 0

# Market Value Prediction Section Improvements

# 1. Modified UI for Market Value display
st.markdown("---")
st.subheader("Market Value Prediction & Similar Players")

# Prepare data for model prediction
player_data_for_prediction = {}

# Extract all important features for prediction
for feature in important_features:
    if feature in input_data:
        player_data_for_prediction[feature] = input_data[feature]
    else:
        # Get positional average if not specified
        position_avg = df[df["SimplePosition"] == selected_position][feature].mean()
        player_data_for_prediction[feature] = position_avg if not pd.isna(position_avg) else 0

# Predict market value using the model
model_predicted_value = predict_market_value(player_data_for_prediction, regression_model)

# IMPROVED PLAYER SIMILARITY CRITERIA
# Function to find similar players with better criteria
def find_similar_players_improved(input_data, df, position, num_players=10):
    # Select relevant features and players with sufficient minutes
    # Increase minimum minutes for more reliable comparisons
    min_minutes = 540  # At least 6 full games

    position_df = df[(df["SimplePosition"] == position) & (df["Min"] >= min_minutes)].copy()

    # Add age range filter (players within 3 years of selected player's age)
    if 'Age' in input_data and not pd.isna(input_data['Age']):
        position_df = position_df[(position_df['Age'] >= input_data['Age'] - 3) &
                                 (position_df['Age'] <= input_data['Age'] + 3)]

    # Ensure all required columns exist
    features = position_features[position]
    for feature in features:
        if feature not in position_df.columns:
            position_df[feature] = 0

    # Handle NaN values
    position_df = position_df.fillna(0)

    # Add feature weights to prioritize important metrics
    feature_weights = {}
    if position == "Forward":
        feature_weights = {"Gls": 2.0, "xG": 1.8, "G/SoT": 1.5, "SoT": 1.3, "Sh": 1.2}
    elif position == "Midfielder":
        feature_weights = {"Ast": 1.8, "PrgP": 1.5, "KP": 1.5, "SCA": 1.3, "PrgC": 1.2}
    elif position == "Defender":
        feature_weights = {"Tkl": 1.8, "Int": 1.8, "Clr": 1.5, "Blocks_stats_defense": 1.3}
    elif position == "Goalkeeper":
        feature_weights = {"Save%": 2.0, "CS%": 1.8, "GA90": 1.5}

    # Default weight for unspecified features
    default_weight = 1.0

    # Extract feature matrix with weights
    X = np.zeros((len(position_df), len(features)))
    for i, feature in enumerate(features):
        weight = feature_weights.get(feature, default_weight)
        X[:, i] = position_df[feature].values * weight

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create weighted input vector
    input_vector = np.zeros((1, len(features)))
    for i, feature in enumerate(features):
        weight = feature_weights.get(feature, default_weight)
        input_vector[0, i] = input_data.get(feature, 0) * weight

    input_scaled = scaler.transform(input_vector)

    # Find nearest neighbors
    knn = NearestNeighbors(n_neighbors=min(num_players + 1, len(position_df)))
    knn.fit(X_scaled)
    distances, indices = knn.kneighbors(input_scaled)

    # Get similar players
    similar_players = []
    for i in range(min(num_players, len(indices[0]))):
        idx = indices[0][i]
        player = position_df.iloc[idx]
        similar_players.append({
            "name": player["Player"],
            "team": player["Squad"],
            "league": player["Comp"],
            "distance": distances[0][i],
            "market_value": np.expm1(player["log_target"]) / 1000000 if not pd.isna(player["log_target"]) else 0,
            "data": {feature: player[feature] for feature in features if feature in player},
            "age": player["Age"]
        })

    return similar_players

# Find similar players with improved criteria
similar_players = find_similar_players_improved(input_data, df, selected_position)

# Calculate average market value from similar players
valid_values = [p["market_value"] for p in similar_players if p["market_value"] > 0]
avg_market_value = sum(valid_values) / len(valid_values) if valid_values else 0

# IMPROVED UI - Emphasize Model Prediction
# Format the market values
formatted_model_value = f"€{model_predicted_value:.2f}M" if model_predicted_value >= 1 else f"€{int(model_predicted_value * 1000)}K"
formatted_avg_value = f"€{avg_market_value:.2f}M" if avg_market_value >= 1 else f"€{int(avg_market_value * 1000)}K"
combined_value = (model_predicted_value + avg_market_value) / 2 if avg_market_value > 0 else model_predicted_value
formatted_combined_value = f"€{combined_value:.2f}M" if combined_value >= 1 else f"€{int(combined_value * 1000)}K"

# Create an improved market value section
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div style="background-color: #e1b12c; border-radius: 10px; padding: 15px; height: 100%;">
        <div style="text-align: center; margin-bottom: 15px;">
            <span style="font-size: 18px; font-weight: bold; color: white;">
                Model Prediction
            </span>
        </div>
        <div style="text-align: center; background-color: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 32px; font-weight: bold; color: black;">
                {formatted_model_value}
            </div>
            <div style="font-size: 14px; color: #666; margin-top: 10px;">
                Based on player metrics & regression analysis
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="background-color: #e1b12c; border-radius: 10px; padding: 15px; height: 100%;">
        <div style="text-align: center; margin-bottom: 15px;">
            <span style="font-size: 18px; font-weight: bold; color: white;">
                Similar Players Average
            </span>
        </div>
        <div style="text-align: center; background-color: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 32px; font-weight: bold; color: black;">
                {formatted_avg_value}
            </div>
            <div style="font-size: 14px; color: #666; margin-top: 10px;">
                Based on {len(valid_values)} similar players
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Add combined value in smaller text (secondary importance)
st.markdown(f"""
<div style="text-align: center; margin-top: 10px; font-size: 14px; color: #666;">
    Combined estimate: {formatted_combined_value}
</div>
""", unsafe_allow_html=True)

# Display similar player criteria
with st.expander("Player similarity criteria"):
    st.markdown("""
    **Player similarity criteria:**
    - Minimum playing time: 540 minutes (6 full games)
    - Age range: Within ±3 years of selected player
    - Position-specific weighted metrics:
      - Forwards: Goals, xG, and shooting efficiency weighted higher
      - Midfielders: Assists, progressive passes, and key passes weighted higher
      - Defenders: Tackles, interceptions, and clearances weighted higher
      - Goalkeepers: Save percentage and clean sheets weighted higher
    """)

# Display similar players with improved styling - add Age information
st.markdown("<h3 style='margin-top: 30px;'>Similar Players</h3>", unsafe_allow_html=True)

# Create a grid layout for similar players
cols_per_row = 2
similar_players_filtered = [p for p in similar_players if p["name"] != selected_player]
rows = [similar_players_filtered[i:i+cols_per_row] for i in range(0, len(similar_players_filtered), cols_per_row)]

for row in rows:
    cols = st.columns(cols_per_row)
    for i, player in enumerate(row):
        # Capitalize player name
        player_name_title = player["name"].title()

        # Format market value consistently
        market_value = player["market_value"]
        formatted_value = f"€{market_value:.2f}M" if market_value >= 1 else f"€{int(market_value * 1000)}K"

        with cols[i]:
            st.markdown(
                f"""
                <div style="background-color: #e1b12c; padding: 15px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); border-top: 3px solid #4682B4;">
                    <div style="font-size: 20px; font-weight: bold; margin-bottom: 5px;">{player_name_title}</div>
                    <div style="color: white; font-size: 18px; margin-bottom: 5px;">{player["team"]} - {player["league"]}</div>
                    <div style="color: white; font-size: 18px; margin-bottom: 10px;">Age: {player["age"]}</div>
                    <div style="font-weight: bold; color: white; font-size: 20px;">{formatted_value}</div>
                </div>
                """,
                unsafe_allow_html=True)

# Display attributes comparison using a radar chart
st.markdown("---")
st.subheader("Player Attributes Comparison - Radar Chart")

# Let user select up to 5 players to compare with
num_players_to_compare = st.slider("Number of similar players to compare:", 1, 5, 3)
similar_player_names = [p["name"] for p in similar_players[:num_players_to_compare] if p["name"] != selected_player]
selected_players = [selected_player] + similar_player_names

# Prepare data for the radar chart
features = position_features[selected_position]
feature_names = [feature_labels.get(f, f) for f in features]

# Get data for the selected players
comparison_data = []
for player_name in selected_players:
    if player_name == selected_player:
        # Get data from the selected player
        player_data = filtered_df[filtered_df["Player"] == player_name].iloc[0]
        player_values = [player_data[f] if f in player_data and not pd.isna(player_data[f]) else 0 for f in features]
        comparison_data.append((player_name, player_values))
    elif player_name in similar_player_names:
        # Get data from similar players
        idx = similar_player_names.index(player_name)
        player_data = similar_players[idx]["data"]
        player_values = [player_data.get(f, 0) for f in features]
        comparison_data.append((player_name, player_values))

# Create radar chart
if comparison_data:
    # Set figure size
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)

    # Compute angle for each feature
    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
    # Make the plot circular by appending the first angle
    angles += angles[:1]

    # Standardize values for better comparison
    # Get min and max values for each feature from the filtered dataframe
    feature_max = {}
    for feature in features:
        if feature in filtered_df.columns:
            max_val = filtered_df[feature].quantile(0.95)  # Using 95th percentile to avoid outliers
            feature_max[feature] = max_val if max_val > 0 else 1.0
        else:
            feature_max[feature] = 1.0

    # Plot each player
    colors = ['b', 'r', 'g', 'purple', 'orange']

    for i, (player_name, values) in enumerate(comparison_data):
        # Normalize values
        normalized_values = []
        for j, val in enumerate(values):
            feature = features[j]
            max_val = feature_max[feature]
            normalized_values.append(min(val / max_val, 1.0))  # Cap at 1.0

        # Complete the loop for the plot
        normalized_values += normalized_values[:1]

        # Plot values
        ax.plot(angles, normalized_values, linewidth=2, linestyle='solid', label=player_name, color=colors[i % len(colors)])
        ax.fill(angles, normalized_values, alpha=0.1, color=colors[i % len(colors)])

    # Set labels and title
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names)
    plt.title(f'Player Comparison - {selected_position}s', size=15)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    st.pyplot(fig)
