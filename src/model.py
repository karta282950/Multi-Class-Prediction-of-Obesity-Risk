
from catboost import CatBoostClassifier
from category_encoders import MEstimateEncoder
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

def select_model(
        best_params, p='lgbm', 
        numerical_columns=None, categorical_columns=None, 
        AgeRounder=None, HeightRounder=None, ExtractFeatures=None, ColumnRounder=None):
    if p=='lgbm': 
        model = make_pipeline(    
                                ColumnTransformer(
                                transformers=[('num', StandardScaler(), numerical_columns),
                                        ('cat', OneHotEncoder(handle_unknown="ignore"), categorical_columns)]),
                                LGBMClassifier(**best_params, verbose=-1)
                            )
        return model
    if p=='xgb':
        model = make_pipeline(
#                     ExtractFeatures,
                    MEstimateEncoder(cols=['Gender','family_history_with_overweight','FAVC','CAEC',
                                           'SMOKE','SCC','CALC','MTRANS']),
                    XGBClassifier(**best_params)
                   )
        return model
    if p=='cat':
        model = make_pipeline(
                        ExtractFeatures,
#                         AgeRounder,
#                         HeightRounder,
#                         MEstimateEncoder(cols = raw_cat_cols),
                        CatBoostClassifier(**best_params, cat_features=categorical_columns)
                        )
        return model