from model import *
import pandas as pd
import sys

from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder
sys.path.append('../')

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import os
import time
import optuna
import joblib
import warnings
import wandb
from wandb.lightgbm import wandb_callback, log_summary

from category_encoders import MEstimateEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from src.utils import cross_val_model
from run.prepare_data import *
from src.tune import *
from omegaconf import DictConfig
import hydra
class cfg:
    my_path = os.path.abspath(os.path.dirname(__file__))

    file_path = os.path.join(my_path, 'data')
    RANDOM_SEED = 42
    target_mapping = {
                  'Insufficient_Weight':0,
                  'Normal_Weight':1,
                  'Overweight_Level_I':2,
                  'Overweight_Level_II':3, 
                  'Obesity_Type_I':4,
                  'Obesity_Type_II':5 ,
                  'Obesity_Type_III':6
                  }
    exp_name = 'exp001'

AgeRounder = FunctionTransformer(age_rounder)
HeightRounder = FunctionTransformer(height_rounder)
ExtractFeatures = FunctionTransformer(extract_features)
ColumnRounder = FunctionTransformer(col_rounder)

train = pd.read_csv(os.path.join(cfg.file_path, "train.csv"))
test = pd.read_csv(os.path.join(cfg.file_path, "test.csv"))
train_org = pd.read_csv(os.path.join(cfg.file_path, "ObesityDataSet.csv"))

train.drop(['id'],axis=1, inplace=True)
test_ids = test['id']
test.drop(['id'],axis=1, inplace=True)

train = pd.concat([train, train_org], axis=0)
train = train.drop_duplicates()
train.reset_index(drop=True, inplace=True)

numerical_columns = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = train.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('NObeyesdad')

# empty dataframe to store score, & train / test predictions.
score_list, oof_list, predict_list = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def main():
    pass


if __name__ == "__main__":
    main()