import streamlit as st
import pandas as pd
import joblib #may not be used
import pickle #The way to load and unload into the files
import os #May not be used
import random #May not be used
import matplotlib.pyplot as plt
import numpy as np
from .similarity import find_similar_players
#All from the web that are needed to display correctly for the information

X_Scaled_Route = 'X_scaled.pkl'
loadedScaler = pickle.load(open('scaler.pkl','rb'))

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
#All code for the styling

# Title of the application
st.markdown("<div class='title'>⚽ Awesome Soccer Project That You Should Rename ⚽</div>", unsafe_allow_html=True)

# Project description
st.markdown("<div class='description'>This awesome project finds the 10 closest players given a subject and calculates the market value. YOU SHOULD ALSO CHANGE THIS</div>", unsafe_allow_html=True)
# The text that is the design part of the application

st.markdown("---")
#The little divider

# Select position to analyze
st.subheader("Select the Position You Want to Analyze")
positions = ["Forward"] #The only position is what we selected to filter it with
selected_position = st.radio("Choose Position:", positions, horizontal=True) #Select the position

st.success(f"You selected: {selected_position}")
#Success when you choose

st.markdown("---") #the little divider

st.header("Find Similar Players")
#Header to say this does all
#The new way to find the player using the packages

#Finding what type of name you wanted
search_method = st.radio("Choose search method:",
                            ('Enter player name', 'Enter stats manually')) #Only player right now

#Load that with a copy call

if search_method == 'Enter player name':
        # Find the player
        player_name = st.text_input("Enter the name of the player:").lower()

        if len(player_name) != 0:

            # The loading
            if player_name != 0:

                attackers_df = " "
                scaler = " "

                try: #Make sure there is data here so that a failure is not presented.
                    similar_players_df = find_similar_players(player_name)

                except FileNotFoundError as e:

                    print("FileNotFoundError")
                    st.write("Can not find the models dataset")
                    exit()

                # The loading also will require some testing on the data and the model.

                if similar_players_df is not None:
                    similar_players = []

                    # Create similar player array using pickle if the find similar worked from the function.
                    with open('X_scaled.pkl', 'rb') as f:

                        similar_players = pickle.load(f)

                    st.subheader("Top 10 Similar Players:")
                    for players in similar_players_df['Player']:
                        st.write(players)

elif search_method == 'Enter stats manually':
    # The correct player is
    st.write("Manual mode not supported at this time")
    exit()
