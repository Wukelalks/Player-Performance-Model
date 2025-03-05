import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

def find_similar_players(player_name=None, player_stats=None):
    """
    Finds the top 10 players similar to a given player based on their stats,
    either by name or by providing the stats manually.
    """

    try:
        # 1. Load Data and Model
        df = pd.read_csv('../raw_data/players_2024_2025_cleaned.csv')
        with open('../models/forward_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('../models/training_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('../models/attackers_df.pkl', 'rb') as f:
            attackers_df = pickle.load(f)

        # Check if the player colmn can be defined, if not, exit
        try:
            df['Player'] = df['Player'].str.lower()
        except KeyError as e:
            print(f"the column {e} was not found in the dataFrame.")
            exit()

        # 2. Feature Selection and Scaling
        attackers_features = ['Gls', 'G-PK', 'PKatt', 'Ast', 'xG', 'npxG', 'Sh/90', 'SoT/90', 'G/SoT', 'Off']
        #attackers_features_renamed = ['Goals', 'Penalty Kicks Attempted', 'Assists', 'Expected Goals', 'Non-Penalty Expected Goals', 'Shots Per 90', 'Shots on Target Per 90', 'Goals Per Shots on Target', 'Offsides','Offsides']  # Removed

        X = df[attackers_features].copy()
        X_scaled = scaler.transform(X) #Scale the x features to be the scaled
        attackers_df_with_index = attackers_df.merge(X, left_index=True, right_index=True)

        # 3. Handle Player Input

        if player_name: #Search by player name, use lowercase
            try:
                # Find the player in the dataframe and extract their stats, and change their values
                player_stats_search = attackers_df[attackers_df['Player'] == player_name][attackers_features].iloc[0].to_dict()
            except (IndexError, KeyError):
                try:
                    #If the first search did not work, try the surname
                    player_stats_search = attackers_df[attackers_df['Player'].str.contains(player_name)][attackers_features].iloc[0].to_dict()  # search if the playe contains the given name, or surname
                except (IndexError, KeyError):
                    print(f"Player '{player_name}' not found in the dataset or missing required stats.")
                    exit()

            #4, Check if the model is valid
            # Check the data for every value in player stats and mark all people to be similar.
            player_stat_array = np.array([player_stats_search[feature] for feature in attackers_features]).reshape(1,-1)
            player_stat_scaled = scaler.transform(player_stat_array) #Transform the data from the training scaler so there can be correct
            similar = model.predict(player_stat_scaled)[0]

            if (similar == 0):
                print('player not valid')
                exit()

            else:

                distances = pairwise_distances(X_scaled, player_stat_scaled, metric='euclidean')
                distances = distances.flatten()

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

        #5
        elif player_stats:  # stats manual
            # Convert player stats to numpy array and scale it
            player_stat_array = np.array([player_stats[feature] for feature in attackers_features]).reshape(1, -1)
            player_scaled = scaler.transform(player_stat_array) # Scale with the *fitted* scaler

            similar = model.predict(player_scaled)[0]

            if (similar == 0):
                print('player not valid')
                exit()

            else:

                distances = pairwise_distances(X_scaled, player_scaled, metric='euclidean')
                distances = distances.flatten()

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

        else:
            print("Must provide either player_name or player_stats.")
            return None

    except FileNotFoundError as e:
        print("File not found")
        exit()

    return similar_players_df
