# In cross_val_model we cross vaidate models using
# Stratified K-Fold.

# Encoding target values with int
import numpy as np
from sklearn import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd


target_mapping = {
                  'Insufficient_Weight':0,
                  'Normal_Weight':1,
                  'Overweight_Level_I':2,
                  'Overweight_Level_II':3, 
                  'Obesity_Type_I':4,
                  'Obesity_Type_II':5 ,
                  'Obesity_Type_III':6
                  }

class cfg:
    n_splits = 5
    seed = 42
    TARGET = 'NObeyesdad'

# Define a method for Cross validation here we are using StartifiedKFold
skf = StratifiedKFold(n_splits=cfg.n_splits)

def cross_val_model(train, test, estimators, cv=skf, verbose=True):
    '''
        estimators : pipeline consists preprocessing, encoder & model
        cv : Method for cross validation (default: StratifiedKfold)
        verbose : print train/valid score (yes/no)
    '''
    
    X = train.copy()
    y = X.pop(cfg.TARGET)

    y = y.map(target_mapping)
    test_predictions = np.zeros((len(test), 7))
    valid_predictions = np.zeros((len(X), 7))

    val_scores, train_scores = [],[]
    for fold, (train_ind, valid_ind) in enumerate(skf.split(X,y)):
        model = clone(estimators)
        #define train set
        X_train = X.iloc[train_ind]
        y_train = y.iloc[train_ind]
        #define valid set
        X_valid = X.iloc[valid_ind]
        y_valid = y.iloc[valid_ind]

        model.fit(X_train, y_train)
        if verbose:
            print("-" * 100)
            print(f"Fold: {fold}")
            print(f"Train Accuracy Score: {accuracy_score(y_true=y_train,y_pred=model.predict(X_train))}")
            print(f"Valid Accuracy Score: {accuracy_score(y_true=y_valid,y_pred=model.predict(X_valid))}")
            print("-" * 100)

        
        test_predictions += model.predict_proba(test)/cv.get_n_splits()
        valid_predictions[valid_ind] = model.predict_proba(X_valid)
        train_scores.append(accuracy_score(y_true=y_train,y_pred=model.predict(X_train)))
        val_scores.append(accuracy_score(y_true=y_valid,y_pred=model.predict(X_valid)))
    if verbose: 
        print(f"Average Mean Accuracy Score: {np.array(val_scores).mean()}")
    return train_scores, val_scores, valid_predictions, test_predictions