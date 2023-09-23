import os,sys

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model,save_object

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initaite_model_trainig(self,train_arr,test_arr):
        try:
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                "LinearRegression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor()
            }

            model_report=evaluate_model(X_train,X_test,y_train,y_test,models)

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(models.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            print(f"best model found: {best_model_name}, r2_score:{best_model_score}")
            print("="*50)
            logging.info(f"best model found,model name: {best_model_name},r2_score:{best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info("error occured in initiate model training")
            raise CustomException(e,sys)