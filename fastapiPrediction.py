from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import envirVariables as datasetPath

# Initialize FastAPI app
app = FastAPI()

# Allow CORS (Cross-Origin Resource Sharing) so that frontend can access the backend
origins = [
    "http://localhost",  # If your frontend is hosted on localhost
    "http://localhost:3000",  # If your frontend is on port 3000 (for example, React app)
    "*",  # Allow all origins (be careful with this in production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained LSTM model (assuming it's already saved as multi_asset_lstm_model.h5)
model = load_model('multi_asset_lstm_model.h5')

# Define the scaler (recreate with the same scaling used during training)
scaler = MinMaxScaler(feature_range=(0, 1))

# Assume the original training dataset is needed for consistent scaling
df = pd.read_csv(datasetPath.giveDatasetPath())  # Make sure your dataset is in the same folder
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Scale the data used for model training
scaled_data = scaler.fit_transform(df)

# Define input schema for prediction request
class PredictionRequest(BaseModel):
    data: list  # A list of last N days' prices, where N = sequence length

# Sequence length used during training
SEQUENCE_LENGTH = 50

@app.get("/")
def read_root():
    return {"message": "Welcome to the Multi-Asset Price Prediction API!"}

@app.post("/predict/")
def predict(request: PredictionRequest):
    """
    Predict the next day's prices for all assets based on the provided sequence.
    """
    # Validate input size
    if len(request.data) != SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Input data must contain exactly {SEQUENCE_LENGTH} timesteps.",
        )
    
    # Convert the input data into a numpy array and validate dimensions
    input_array = np.array(request.data)
    
    # Ensure input is 2D (for example, [50 timesteps, number of assets])
    if input_array.ndim != 2 or input_array.shape[0] != SEQUENCE_LENGTH or input_array.shape[1] != df.shape[1]:
        raise HTTPException(
            status_code=400,
            detail=f"Each timestep must contain {df.shape[1]} asset prices, and the sequence length must be {SEQUENCE_LENGTH}.",
        )
    
    # Scale the input data
    scaled_input = scaler.transform(input_array)
    
    # Reshape the input to fit the LSTM input format (1 sample, SEQUENCE_LENGTH, features)
    reshaped_input = np.reshape(scaled_input, (1, SEQUENCE_LENGTH, df.shape[1]))
    
    # Make prediction
    scaled_prediction = model.predict(reshaped_input)
    
    # Inverse scale the prediction to get the original prices
    prediction = scaler.inverse_transform(scaled_prediction)[0]
    
    # Convert prediction values to native Python floats for JSON serialization
    prediction = [float(price) for price in prediction]
    
    # Return the result as a dictionary of predicted prices for each asset
    response = {asset: price for asset, price in zip(df.columns, prediction)}
    
    return {"predicted_prices": response}

# If running as a script, start the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
