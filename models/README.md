# Models Directory

This folder contains pre-trained machine learning models used for player classification and market value prediction in the Streamlit application.

## Model Files Structure
Each position has two types of models:
- **Classification Model:** Predicts similar players based on input features.
- **Regression Model:** Estimates the market value of the given player.

### **Model Naming Convention**
Each model file follows this naming pattern:
```
models/{position}_classification.pkl  # Classification model
models/{position}_regression.pkl      # Regression model
```
Where `{position}` can be one of:
- `Forward`
- `Midfielder`
- `Defender`
- `Goalkeeper`

### **Example Model Paths**
```
models/Forward_classification.pkl
models/Forward_regression.pkl
models/Midfielder_classification.pkl
models/Midfielder_regression.pkl
models/Defender_classification.pkl
models/Defender_regression.pkl
models/Goalkeeper_classification.pkl
models/Goalkeeper_regression.pkl
```

## Model Usage in Streamlit
The Streamlit app automatically loads the corresponding models based on the selected position. If the models are missing, the app generates random values for demonstration purposes.

## Adding or Updating Models
To replace or update a model, simply add a new `.pkl` file with the appropriate naming convention and ensure it follows the expected format for classification and regression tasks.


