from __future__ import annotations # Needed to use better TypeHints
import joblib # Pickle-like pyhton module used to save/load models
from pathlib import Path # Resolve paths
from model.data import load_data # For testing

import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error
)

from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV # Not being used, should be used or removed

model_folder = Path(__file__).resolve().parent
class RealEstateChilePriceModel():
    def __init__(self) -> None:
        """
        This class is meant to abstract for the use of different models in the future.
        Currently, it is responsible for training the predictive model and evaluating new inputs.
        """
        
        self.model = None # So we can check if fit was called at least once
        pass

    def fit(self, train: np.ndarray | pd.DataFrame, test: np.ndarray | pd.DataFrame, save_when_done: bool = True) -> RealEstateChilePriceModel:
        """
        Method used to fit pre-programmed model to input data.
        Basically the same as Scikit-Learn's fit function, except we build the whole model here.
        We need information from the training data (such as column names and shapes) for this, 
        which is why the model construction is not done elsewhere. 
        """

        # X Columns
        train_cols = [
            col for col in train.columns if col not in ['id', 'target']
        ]

        # Categorical variables
        categorical_cols = ["type", "sector"]
        target           = "price"
        categorical_transformer = TargetEncoder()

        # Preprocessing steps of pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('categorical',
                categorical_transformer,
                categorical_cols)
            ]
        )

        # Pipeline construction (this is essentially the real model)
        steps = [
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(**{
                "learning_rate":0.01,
                "n_estimators":300,
                "max_depth":5,
                "loss":"absolute_error"
            }))
        ]
        pipeline = Pipeline(steps)
        
        # Calling pipeline.fit, essentially, training happens here
        self.model = pipeline.fit(train[train_cols], train[target])

        # Saving model for later use
        if save_when_done: self.save_current_model()

        # In case user wants/needs to assign trained model to different variable, return self
        return self

    def predict(self, input_data: np.ndarray | pd.DataFrame) -> float:
        """
        Method used to predict regressed values for input data.
        Basically the same as Scikit-Learn's predict function.
        """
        # Make prediction
        prediction = self.model.predict(input_data)
        return prediction
    
    def load_pretrained_model(self, model_path: str = model_folder / "model.pkl") -> RealEstateChilePriceModel:
        self.model = joblib.load(model_path) 
        return self

    def save_current_model(self, model_path: str = model_folder / "model.pkl"):
        joblib.dump(self.model, model_path)

if __name__ == "__main__":
    # Load data
    train, test = load_data()
    
    # Instantiate model
    model = RealEstateChilePriceModel()
    # Fit model
    model.fit(train, test)
    
    # Make prediction on test set
    pred = model.predict(test)
    test_target = test["price"].values
    def print_metrics(predictions, target):
        print("RMSE: ", np.sqrt(mean_squared_error(predictions, target)))
        print("MAPE: ", mean_absolute_percentage_error(predictions, target))
        print("MAE : ", mean_absolute_error(predictions, target))
    print_metrics(pred, test_target)
    # RMSE:  10254.155686652393
    # MAPE:  0.40042979298798137
    # MAE :  5859.374796053153