from __future__ import annotations
from typing import List
import logging
import json
from pathlib import Path
from datetime import date
import pandas as pd
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from model.model import RealEstateChilePriceModel, model_folder # Import our defined model
from api.key_manager import KeyManager

# Create a log path, ensure it's available to write on
log_path = Path(f"logs/log_{str(date.today())}.txt").resolve()
log_path.parent.mkdir(mode=0o754, parents=True, exist_ok=True)
log_path.touch(mode=0o754, exist_ok=True)

# New log file every day, very basic
# WON'T generate new file for next day, unless the server restarts
# Eventually logging should be done in a more cloud-friendly way, sending the log message to a trusted separate server for safe-keeping.
logging.basicConfig(
    format="PID:%(process)d (TID:%(thread)d) %(levelname)s:\t[%(asctime)s] %(message)s",
    filename=log_path, 
    level=logging.DEBUG
)

# Model path
MODEL_PATH = model_folder / "model.pkl"
# Load model
try:
    model = RealEstateChilePriceModel().load_pretrained_model(MODEL_PATH) # Could include model path here, using default
    logging.info(f"Model loaded successfuly from path {MODEL_PATH}, beginning operation...")
except FileNotFoundError:
    logging.error("Failed to load archived model! Models must be trained and made available before running API server!")
    raise Exception("Pre-trained model cound not be found! Please train your model before running the API, or update api/app/api.py with the correct filepath!")

# Api key management stuff
key_manager = KeyManager() # Manager object, used to validate keys

# Basic API authentication found and adapted from https://medium.com/@valerio.uberti23/a-beginners-guide-to-using-api-keys-in-fastapi-and-python-256fe284818d
api_key_header = APIKeyHeader(name="X-API-Key") # Header key used to deliver API Key
def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    if key_manager.validateKey(api_key_header):
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )

app = FastAPI()
logging.info("Server started, listening for calls")

class PredictionInput(BaseModel):
    """
    Basic JSON formatting for sent items
    """
    type           : str
    sector         : str
    net_usable_area: float
    net_area       : float
    n_rooms        : float
    n_bathroom     : float
    latitude       : float
    longitude      : float
    price          : float # This is wrong, must change

class BatchInput(BaseModel):
    """
    A Different model, meant to allow for batch prediction of multiple data points at once.
    """
    data           : List[PredictionInput]

@app.get("/new_key")
async def new_key(keyLifespan: float = None):
    """
    Route for requesting new API keys.
    In a real project, this would of course be a very different process,
    possibly requiring users to either submit real forms specifying their use case,
    or maybe just some internal key-vault distribution system, that automatically refreshes
    service endpoints' keys when needed.

    Query Parameters:
    keyLifespan: float, when specified, indicates the desired lifespan of the key in hours. Defaults to 24h.
    """
    # No need to validate types, FastAPI and pydantic handle that for us.
    # Maybe in the future a criterion for determining valid lifespans, 
    # such as a maximum and minimum valid lifespan could be implemented here.
    return key_manager.generateNewKey(keyLifespan)

@app.post("/predict")
async def predict(model_input: PredictionInput, api_key: str = Security(get_api_key)):
    """
    Prediction route, used to submit values for the model to analyze.
    Requires a valid API Key, filled in the X-API-Key field in the header of the request.

    Parameters:
    model_input: Data for prediction, folowing format of PredictionInput BaseModel.
    api_key: Is provided through the headers of the request. A str containing the key generated by the /new_key route.
    """
    # Pydantic's Base Model can be turned into a dictionary easily
    input_dict = model_input.model_dump()

    # Log arrival of request
    logging.info(f"[MODEL INPUT] API Call using key {api_key} made for prediction using data: {input_dict}")
    
    # Make into pd.DataFrame
    df = pd.DataFrame(input_dict, index=[0])
    prediction = model.predict(df)[0]
    
    # Log prediction result
    logging.info(f"[MODEL OUTPUT] API call is returning predicted value, which is `{prediction}`")
    return prediction

@app.post("/predict_batch")
async def predict(model_input_batch: BatchInput, api_key: str = Security(get_api_key)):
    """
    Prediction route, used to submit values for the model to analyze.
    Requires a valid API Key, filled in the X-API-Key field in the header of the request.

    Parameters:
    model_input_batch: Data for prediction, folowing format of BatchInput BaseModel.
    api_key: Is provided through the headers of the request. A str containing the key generated by the /new_key route.
    """
    # Pydantic's Base Model can be turned into a dictionary easily
    input_dict = model_input_batch.model_dump()["data"]

    # Log arrival of request
    logging.info(f"[MODEL INPUT] Batch API Call using key {api_key} made for predictions using data (length {len(model_input_batch)}): {input_dict}")
    
    # Make into pd.DataFrame
    df = pd.DataFrame(input_dict)
    prediction = model.predict(df).flatten().tolist()
    
    # Log prediction result
    logging.info(f"[MODEL OUTPUT] Batch API call is returning predicted values, which is `{prediction}`")
    return json.dumps(prediction)