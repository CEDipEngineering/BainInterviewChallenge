import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from model.model import RealEstateChilePriceModel, model_folder # Import our defined model

MODEL_PATH = model_folder / "model.pkl"

app = FastAPI()
try:
    model = RealEstateChilePriceModel().load_pretrained_model(MODEL_PATH) # Could include model path here, using default
except FileNotFoundError:
    raise Exception("Pre-trained model cound not be found! Please train your model before running the API, or update api/app/api.py with the correct filepath!")


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

@app.post("/predict")
async def predict(model_input: PredictionInput):
    # Pydantic's Base Model can be turned into a dictionary easily
    input_dict = model_input.model_dump()
    # Make into pd.DataFrame
    df = pd.DataFrame(input_dict, index=[0])
    prediction = model.predict(df)[0]
    return prediction

