import wandb 
import joblib
from lightgbm import LGBMClassifier

def lgb_train_and_predict(p):

    lgbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=30,
        valid_sets=lgb_eval,
        valid_names=("validation"),
        callbacks=[wandb_callback()],
        early_stopping_rounds=5,
    )

def cat_train_and_predict(p):
    pass

def rfc_train_and_predict(p):
    pass

def xgb_train_and_predict(p):
    pass

def nn_train_and_predict(p):
    pass