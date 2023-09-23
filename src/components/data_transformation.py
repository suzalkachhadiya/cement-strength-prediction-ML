import os,sys

import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformation()

    def get_data_transformation_obj(self):
        logging.info("Data transformationinitiateD")

        try:
            num_columns=[
                "Cement (component 1)(kg in a m^3 mixture)",
                "Blast Furnace Slag (component 2)(kg in a m^3 mixture)",
                "Fly Ash (component 3)(kg in a m^3 mixture)",
                "Water  (component 4)(kg in a m^3 mixture)",
                "Superplasticizer (component 5)(kg in a m^3 mixture)",
                "Coarse Aggregate  (component 6)(kg in a m^3 mixture)",
                "Fine Aggregate (component 7)(kg in a m^3 mixture)",
                "Age (day)"
            ]

            logging.info('Pipeline Initiated')

            pipe=Pipeline(
                steps=[
                    ("imputer",KNNImputer()),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
                ("num_pipeline",pipe,num_columns)
            ])

            logging.info("pipeline completed")

            return preprocessor
        
        except Exception as e:
            logging.info("error in Data transformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info("obtainig preprocessor object")

            preprocessor_obj=self.get_data_transformation_obj()

            target_column_name="Concrete compressive strength(MPa, megapascals)"
            drop_cols=[target_column_name]

            # features into independent and dependent features
            input_feature_train_df = train_df.drop(columns=drop_cols,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_cols,axis=1)
            target_feature_test_df=test_df[target_column_name]

            # transform the Data
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            logging.info("applying transformation on train anD test Datasets")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info("preprocessor.pkl created")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info("error occured in initiate data transformation")
            raise CustomException(e,sys)
