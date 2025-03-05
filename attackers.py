import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

# 1. Loading data
df_all = pd.read_csv('raw_data/players_2024_2025_cleaned.csv')
attacker_pos = ['FW', 'FW,MF', 'MF,FW']
attackers_df = df_all[df_all['Pos'].isin(attacker_pos)]

# 2. Feature selection
attackers_features = ['Gls', 'G-PK', 'PKatt', 'Ast', 'xG', 'npxG', 'Sh/90', 'SoT/90', 'G/SoT', 'Off',]
attackers_features_renamed = ['Goals', 'Penalty Kicks Attempted', 'Assists', 'Expected Goals','Non-Penalty Expected Goals','Shots Per 90','Shots on Target Per 90','Goals Per Shots on Target','Offsides','Offsides']

# Mapping for renamed features to original features
feature_mapping = {
    'Goals': 'Gls',
    'Penalty Kicks Attempted': 'PKatt',
    'Assists': 'Ast',
    'Expected Goals': 'xG',
    'Non-Penalty Expected Goals': 'npxG',
    'Shots Per 90': 'Sh/90',
    'Shots on Target Per 90': 'SoT/90',
    'Goals Per Shots on Target':'G/SoT',
    'Offsides':'Off',
    'off':'Off'
}

# DataFrame with attackers features
X = attackers_df[attackers_features].copy()

# Data scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dumping with pickle the X_scaled and scaler for later use
filename = 'X_scaled.pkl'
pickle.dump(X_scaled, open(filename, 'wb'))
filename = 'scaler.pkl'
pickle.dump(scaler, open(filename, 'wb'))

#loading with pickle, test if that worked
pickle_in = open('X_scaled.pkl', 'rb')
X_scaled_with_pickle = pickle.load(pickle_in)
pickle_in = open('scaler.pkl', 'rb')
scaler_with_pickle = pickle.load(pickle_in)


# Ask the user how they want to search for similar players
search_method = input("Choose search method: (1) Enter player name, or (2) Enter stats manually: ")

if search_method == '1':
    player_name = input("Enter the name of the player to find similar players: ").lower() #convert the player name to be lower to be compatible

    try:
        # Find the player in the dataframe and extract their stats
        # Check for full name match first
        player_stats = attackers_df[attackers_df['Player'] == player_name][attackers_features].iloc[0].to_dict()
    except (IndexError, KeyError):
        # If full name not found, try searching by surname
        try:
            player_stats = attackers_df[attackers_df['Player'].str.contains(player_name)][attackers_features].iloc[0].to_dict() #search if the playe contains the given name, or surname
        except (IndexError, KeyError):
            print(f"Player '{player_name}' not found in the dataset or missing required stats.")
            exit()

elif search_method == '2':
    # Finding similar players by entering features
    player_stats = {}
    for i, feature_renamed in enumerate(attackers_features_renamed):
        while True:  # Input validation loop
            try:
                player_stats[attackers_features[i]] = float(input(f"Enter {feature_renamed} for the player: "))
                break  # Exit loop if input is valid
            except ValueError:
                print("Invalid input. Please enter a numerical value.")

else:
    print("Invalid search method. Please enter '1' or '2'.")
    exit()

# Convert player stats to numpy array and scale it
player_array = np.array([player_stats[feature] for feature in attackers_features]).reshape(1, -1)

#player_scaled = scaler.transform(player_array) # Scale with the *fitted* scaler
player_scaled = scaler_with_pickle.transform(player_array) # Scale with the loaded scaler

# Calculate distances from the input player
#distances = pairwise_distances(X_scaled, player_scaled, metric='euclidean')
distances = pairwise_distances(X_scaled_with_pickle, player_scaled, metric='euclidean')
distances = distances.flatten()

# Find common index by merging attacker_df and X, while checking that it exist,
attackers_df_with_index = attackers_df.merge(X, left_index=True, right_index=True)

# Get indices of the n smallest distances (the first one is the most similar)
similar_player_indices = np.argsort(distances)[:10]  # Get top 10

# Create a DataFrame of similar players and their similarity scores
try:
        similar_players_df = pd.DataFrame({
        'Player': attackers_df_with_index.iloc[similar_player_indices]['Player'].values,  # Get names from all df
        'similarity_score': 1 / (1 + distances[similar_player_indices])  # Convert distance to similarity
    })
except KeyError as e:
    print(f"The column {e} was not found in the DataFrame.")
    similar_players_df = pd.DataFrame()


print(f"Top 10 players similar to the input stats:")
print(similar_players_df)
