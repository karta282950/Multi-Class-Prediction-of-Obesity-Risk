import sys
sys.path.append('../')


from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import os
import optuna
import warnings
import wandb
from category_encoders import MEstimateEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from src.utils import cross_val_model
from run.prepare_data import *

class cfg:
    file_path = './data/'
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

# empty dataframe to store score, & train / test predictions.
score_list, oof_list, predict_list = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Define Random Forest Model Pipeline
RFC = make_pipeline(
        ExtractFeatures,
        MEstimateEncoder(cols=['Gender','family_history_with_overweight','FAVC','CAEC',
                            'SMOKE','SCC','CALC','MTRANS']),
        RandomForestClassifier(random_state=cfg.RANDOM_SEED)
        )


# Define Optuna Function To Tune LGBM Model

def lgbm_objective(trial):
    params = {
        'learning_rate' : trial.suggest_float('learning_rate', .001, .1, log = True),
        'max_depth' : trial.suggest_int('max_depth', 2, 20),
        'subsample' : trial.suggest_float('subsample', .5, 1),
        'min_child_weight' : trial.suggest_float('min_child_weight', .1, 15, log = True),
        'reg_lambda' : trial.suggest_float('reg_lambda', .1, 20, log = True),
        'reg_alpha' : trial.suggest_float('reg_alpha', .1, 10, log = True),
        'n_estimators' : 1000,
        'random_state' : cfg.RANDOM_SEED,
        'device_type' : "cpu",
        'num_leaves': trial.suggest_int('num_leaves', 10, 1000),
        'objective': 'multiclass_ova',
        #'boosting_type' : 'dart',
    }
    
    optuna_model = make_pipeline(
                                ExtractFeatures,
                                MEstimateEncoder(cols=[
                                    'Gender','family_history_with_overweight','FAVC','CAEC', 'SMOKE','SCC','CALC','MTRANS']),
                                LGBMClassifier(**params,verbose=-1)
                                )
    val_scores, _, _ = cross_val_model(train, test, optuna_model, verbose=False)
    return np.array(val_scores).mean()

def cal_cv():
    val_scores, val_predictions, test_predictions = cross_val_model(train, test, RFC)

    # Save train/test predictions in dataframes
    for k,v in cfg.target_mapping.items():
        oof_list[f"rfc_{k}"] = val_predictions[:,v]

    for k,v in cfg.target_mapping.items():
        predict_list[f"rfc_{k}"] = test_predictions[:,v]

def op(p='lgbm'):
    warnings.filterwarnings("ignore")
    if p=='lgbm':
        study = optuna.create_study(direction = 'maximize',study_name="LGBM")
        study.optimize(lgbm_objective, 50)

if __name__=='__main__':
    op()