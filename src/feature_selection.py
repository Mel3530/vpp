import pandas as pd
import xgboost as xgb

def xgboost_feature_importance(X, y, top_n=20):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    importance = model.feature_importances_
    return pd.Series(importance, index=X.columns).nlargest(top_n).index.tolist()

def select_features(X, y):
    return xgboost_feature_importance(X, y, top_n=15)
