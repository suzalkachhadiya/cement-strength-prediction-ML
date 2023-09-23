import os,sys

import pandas as pd

from src.exception import CustomException
from src.logger import logging

from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            scaled_data=preprocessor.transform(features)

            pred=model.predict(scaled_data)

            return pred

        except Exception as e:
            logging.info("error occured in predict")

class CustomData:
    def __init__(self,
                 Cement,
                 Blast_furnace_slag,
                 Fly_ash,
                 Water,
                 Superplasticizer,
                 Coarse_aggregate,
                 Fine_aggregate,
                 Age                               
                 ):
        
        self.Cement=Cement
        self.Blast_furnace_slag=Blast_furnace_slag
        self.Fly_ash=Fly_ash
        self.Water=Water
        self.Superplasticizer=Superplasticizer
        self.Coarse_aggregate=Coarse_aggregate
        self.Fine_aggregate=Fine_aggregate
        self.Age=Age

        def get_data_as_dataframe(self):
            try:
                custom_data_input_dict={
                "Cement":[self.Cement],
                "Blast_furnace_slag":[self.Blast_furnace_slag],
                "Fly_ash":[self.Fly_ash],
                "Water":[self.Water],
                "Superplasticizer":[self.Superplasticizer],
                "Coarse_aggregate":[self.Coarse_aggregate],
                "Fine_aggregate":[self.Fine_aggregate],
                "Age":[self.Age],
                }

                df=pd.DataFrame(custom_data_input_dict)
                logging.info("Dataframe gathered")

                return df

            except Exception as e:
                logging.info("error occureD in get Data as Dataframe")
                raise CustomException(e,sys)



        




