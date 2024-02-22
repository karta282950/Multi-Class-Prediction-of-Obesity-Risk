import os
import pandas as pd
from sklearn.pipeline import FunctionTransformer

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

#model predict
