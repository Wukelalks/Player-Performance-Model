# Player Performance Model

## Overview

The **Player Performance Model** is a machine learning-powered application designed to analyze soccer players' performance, find similar players based on attributes, and predict their estimated transfer market value.

## Features

- **Find Similar Players**: Given a player, the model finds other players with similar attributes.
- **Market Value Prediction**: Predicts the estimated transfer fee based on player statistics.
- **Interactive Frontend**: Uses Streamlit for an intuitive user experience.

## Folder Structure

```
Player-Performance-Model/
├── app.py             # Main Streamlit application
├── images/            # Frontend images and assets
├── models/            # Machine learning models (linear & classification)
├── raw_data/          # Raw datasets for different player attributes
├── utils/             # Auxiliary functions for preprocessing & inference
├── README.md          # Project documentation
├── requirements.txt   # Dependencies list
```

## Installation

### Prerequisites

Ensure you have Python installed (>=3.8). Then, create a virtual environment and install dependencies:

```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Application

```sh
streamlit run app.py
```

This will launch the Streamlit web interface in your browser.

## Usage

1. **Select a Player**: Choose or input a player.
2. **Find Similar Players**: The model will retrieve players with similar attributes.
3. **Market Value Prediction**: Get an estimated transfer value for the selected player.

## Model Details

- **Linear Regression Model**: Predicts transfer market value.
- **Classification Model**: Groups players based on similar attributes.

## Future Enhancements

- Improve model accuracy with more data features.
- Add player comparison visualizations.
- Deploy as a cloud-hosted web app.

## Contributors

- 



