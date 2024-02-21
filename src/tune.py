from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import numpy as np
from sklearn.pipeline import make_pipeline
from category_encoders import MEstimateEncoder
from src.utils import cross_val_model

class cfg:
    RANDOM_SEED=42

def lgbm_objective(train, test, ExtractFeatures, trial):
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

# Optuna study for XGB Model
def xgb_objective(train, test, trial):
    params = {
        'grow_policy': trial.suggest_categorical('grow_policy', ["depthwise", "lossguide"]),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
        'gamma' : trial.suggest_float('gamma', 1e-9, 1.0),
        'subsample': trial.suggest_float('subsample', 0.25, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.25, 1.0),
        'max_depth': trial.suggest_int('max_depth', 0, 24),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 10.0, log=True),
    }

    params['booster'] = 'gbtree'
    params['objective'] = 'multi:softmax'
    params["device"] = "cuda"
    params["verbosity"] = 0
    params['tree_method'] = "gpu_hist"
    
    
    optuna_model = make_pipeline(
#                     ExtractFeatures,
                    MEstimateEncoder(cols=['Gender','family_history_with_overweight','FAVC','CAEC',
                                           'SMOKE','SCC','CALC','MTRANS']),
                    XGBClassifier(**params,seed=cfg.RANDOM_SEED)
                   )
    
    val_scores, _, _ = cross_val_model(train, test, optuna_model, verbose=False)
    return np.array(val_scores).mean()


# Optuna Function For Catboost Model
def cat_objective(train, test, ExtractFeatures, trial):
    
    params = {
        
        'iterations': 1000,  # High number of estimators
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.01, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_seed': cfg.RANDOM_SEED,
        'verbose': False,
        'task_type':"GPU"
    }
    
    cat_features = ['Gender','family_history_with_overweight','FAVC','FCVC','NCP',
                'CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS']
    optuna_model = make_pipeline(
                        ExtractFeatures,
#                         AgeRounder,
#                         HeightRounder,
#                         MEstimateEncoder(cols = raw_cat_cols),
                        CatBoostClassifier(**params,cat_features=cat_features)
                        )
    val_scores,_,_ = cross_val_model(train, test, optuna_model,verbose = False)
    return np.array(val_scores).mean()
    