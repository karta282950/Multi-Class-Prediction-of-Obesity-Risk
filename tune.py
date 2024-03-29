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

# Define Random Forest Model Pipeline
RFC = make_pipeline(
        ExtractFeatures,
        MEstimateEncoder(cols=['Gender','family_history_with_overweight','FAVC','CAEC',
                            'SMOKE','SCC','CALC','MTRANS']),
        RandomForestClassifier(random_state=cfg.RANDOM_SEED)
        )

def cal_cv(model=RFC):
    _, val_scores, val_predictions, test_predictions = cross_val_model(train, test, model)

    # Save train/test predictions in dataframes
    for k,v in cfg.target_mapping.items():
        oof_list[f"rfc_{k}"] = val_predictions[:,v]

    for k,v in cfg.target_mapping.items():
        predict_list[f"rfc_{k}"] = test_predictions[:,v]

#@hydra.main(config_path="run/conf", config_name="tune", version_base="1.1")
def op(p='lgbm', num_trail=50):
    warnings.filterwarnings("ignore")
    if p=='lgbm':
        study = optuna.create_study(direction='maximize', study_name="LGBM")
        study.optimize(lgbm_objective, num_trail)
        joblib.dump(study, os.path.join(cfg.my_path, f'output/train/{p}_optuna_5fold_{num_trail}trail.pkl'))
    #if p=='rfc':
    #    study = optuna.create_study(direction='maximize', study_name="RFC")
    #    study.optimize(lgbm_objective, num_trail)
    #    joblib.dump(study, os.path.join(cfg.my_path, f'output/train/{p}_optuna_5fold_{num_trail}trail.pkl'))
    if p=='xgb':
        study = optuna.create_study(direction='maximize', study_name="XGB")
        study.optimize(xgb_objective, num_trail)
        joblib.dump(study, os.path.join(cfg.my_path, f'output/train/{p}_optuna_5fold_{num_trail}trail.pkl'))
    if p=='cat':
        study = optuna.create_study(direction='maximize', study_name="CAT")
        study.optimize(cat_objective, num_trail)
        joblib.dump(study, os.path.join(cfg.my_path, f'output/train/{p}_optuna_5fold_{num_trail}trail.pkl'))

def train_and_predict(p='lgb'):
    if p=='lgb':
        model = make_pipeline(
                        ColumnTransformer(
                        transformers=[('num', StandardScaler(), numerical_columns),
                                  ('cat', OneHotEncoder(handle_unknown="ignore"), categorical_columns)]),
                        LGBMClassifier(**jl.best_params, verbose=-1)
                    )
    if p=='rfc':
        model = make_pipeline(
                        ColumnTransformer(
                        transformers=[('num', StandardScaler(), numerical_columns),
                                  ('cat', OneHotEncoder(handle_unknown="ignore"), categorical_columns)]),
                        LGBMClassifier(**jl.best_params, verbose=-1)
                    )
    if p=='cat':
        model = make_pipeline(
                        ColumnTransformer(
                        transformers=[('num', StandardScaler(), numerical_columns),
                                  ('cat', OneHotEncoder(handle_unknown="ignore"), categorical_columns)]),
                        LGBMClassifier(**jl.best_params, verbose=-1)
                    )
    if p=='xgb':
        model = make_pipeline(
                        ColumnTransformer(
                        transformers=[('num', StandardScaler(), numerical_columns),
                                  ('cat', OneHotEncoder(handle_unknown="ignore"), categorical_columns)]),
                        LGBMClassifier(**jl.best_params, verbose=-1)
                    )
    return model

if __name__=='__main__':
    warnings.filterwarnings("ignore")
    tune = True
    p = 'xgb'
    if tune:
        op(p)
    if not tune:
        jl = joblib.load(os.path.join(cfg.my_path, 'output/train/lgb_optuna_5fold_50trail.pkl'))
        print('Best Trial', jl.best_trial.params)
        wandb.init(project="multi-class-prediction-of-obesity-risk", config=jl.best_trial.params, name='lgb_fold50_op5')
        model = train_and_predict()
        train_scores, val_scores, _, _ = cross_val_model(train, test, model, verbose=True) # 有test predict
        wandb.log({"train_acc": np.array(train_scores).mean()})
        wandb.log({"val_acc": np.array(val_scores).mean()})
        #log_summary(lgbm, save_model_checkpoint=True)# 上傳model與feature importance