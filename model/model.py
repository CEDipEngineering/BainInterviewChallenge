from __future__ import annotations # Needed to use better TypeHints
import joblib # Pickle-like pyhton module used to save/load models
from pathlib import Path # Resolve paths

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

    def fit(self, train: np.ndarray | pd.DataFrame, test: np.ndarray | pd.DataFrame, save_when_done: bool = True, model_path: str = model_folder / 'model.pkl') -> RealEstateChilePriceModel:
        """
        Method used to fit pre-programmed model to input data.
        Basically the same as Scikit-Learn's fit function, except we build the whole model here.
        We need information from the training data (such as column names and shapes) for this, 
        which is why the model construction is not done elsewhere. 

        Parameters:
        train: np.ndarray or pd.DataFrame, for this particular case, the model was built using column names on the transformers, and so pd.DataFrame is mandatory.
        train: np.ndarray or pd.DataFrame, for this particular case, the model was built using column names on the transformers, and so pd.DataFrame is mandatory.
        save_when_done: True or False, indicates whether or not the model should be saved to disk after fit. Defaults to True (model is saved).
        model_path: A str or path-like object, denoting where the model should be saved to on disk. Ignored if save_when_done is False.
        """

        # X Columns
        train_cols = [
            col for col in train.columns if col not in ['id', 'target'] # The target column is called 'price' and is included here
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
        
        try:
            # Calling pipeline.fit, essentially, training happens here
            self.model = pipeline.fit(train[train_cols], train[target])
        except Exception as e:
            raise Exception(f"Error when trying to fit model, make sure you are passing a pd.DataFrame for both train and test\n{e}")
        
        # Saving model for later use
        if save_when_done: self.save_current_model(model_path)

        # In case user wants/needs to assign trained model to different variable, return self
        return self

    def predict(self, input_data: np.ndarray | pd.DataFrame) -> float:
        """
        Method used to predict regressed values for input data.
        Basically the same as Scikit-Learn's predict function.

        Parameters:
        input_data: np.ndarray or pd.DataFrame. Ideally the same type as was used to fit the model. Must have the same shape as training data (except for number of samples).
        """
        try:
            # Make prediction
            prediction = self.model.predict(input_data)
        except Exception as e:
            raise Exception(f"Error when trying to predict given data, make sure data is the same format as training data\n{e}")
        return prediction
    
    def load_pretrained_model(self, model_path: str = model_folder / "model.pkl") -> RealEstateChilePriceModel:
        """
        Method used for loading a pre-trained model.
        
        Parameters: 
        model_path: A str or path-like object, denoting where the model is on disk. 
        """
        try:
            self.model = joblib.load(model_path) 
        except FileNotFoundError:
            raise Exception("The path you are attempting to load a model from currently does not exist. Please run model/model.py to create and save a model.")
        return self

    def save_current_model(self, model_path: str = model_folder / "model.pkl"):
        """
        Method used for saving current model.
        
        Parameters: 
        model_path: A str or path-like object, denoting where the model should be saved to on disk. 
        """
        if model is None: raise Exception("Cannot save empty model! Please use fit before calling save_current_model.")
        joblib.dump(self.model, model_path)

if __name__ == "__main__":
    from data import load_data # For testing
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