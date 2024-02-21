from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix 


# skf = StratifiedKFold(n_splits=5)
weights = {"rfc_":0,
           "lgbm_":3,
           "xgb_":1,
           "cat_":0}
tmp = oof_list.copy()
for k,v in target_mapping.items():
    tmp[f"{k}"] = (weights['rfc_']*tmp[f"rfc_{k}"] +
              weights['lgbm_']*tmp[f"lgbm_{k}"]+
              weights['xgb_']*tmp[f"xgb_{k}"]+
              weights['cat_']*tmp[f"cat_{k}"])    
tmp['pred'] = tmp[target_mapping.keys()].idxmax(axis = 1)
tmp['label'] = train[TARGET]
print(f"Ensemble Accuracy Scoe: {accuracy_score(train[TARGET],tmp['pred'])}")
    
cm = confusion_matrix(y_true = tmp['label'].map(target_mapping),
                      y_pred = tmp['pred'].map(target_mapping),
                     normalize='true')

cm = cm.round(2)
plt.figure(figsize=(8,8))
disp = ConfusionMatrixDisplay(confusion_matrix = cm,
                              display_labels = target_mapping.keys())
disp.plot(xticks_rotation=50)
plt.tight_layout()
plt.show()

"""   BEST     """

# Best LB [0,1,0,0]
# Average Train Score:0.9142044335854003
# Average Valid Score:0.91420543252078

# Best CV [1,3, 1,1]
# Average Train Score:0.9168308163711971
# Average Valid Score:0.9168308163711971
# adding orignal data improves score


for k,v in target_mapping.items():
    predict_list[f"{k}"] = (weights['rfc_']*predict_list[f"rfc_{k}"]+
                            weights['lgbm_']*predict_list[f"lgbm_{k}"]+
                            weights['xgb_']*predict_list[f"xgb_{k}"]+
                            weights['cat_']*predict_list[f"cat_{k}"])

final_pred = predict_list[target_mapping.keys()].idxmax(axis = 1)

sample_sub[TARGET] = final_pred
sample_sub.to_csv("submission.csv",index=False)
sample_sub
