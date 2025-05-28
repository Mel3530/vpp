# src/models.py
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

model_configs = {
    "RandomForest": {
        "model": RandomForestClassifier,
        "params": {
            "class_weight": ["balanced"],
            "criterion": ["gini"],
            "max_depth": (1, 10),
            "min_samples_leaf": (0.01, 0.47),
            "n_estimators": (10, 1000)
        }
    },
    "ExtraTrees": {
        "model": ExtraTreesClassifier,
        "params": {
            "class_weight": ["balanced"],
            "criterion": ["gini"],
            "max_depth": (1, 10),
            "min_samples_leaf": (0.01, 0.461),
            "n_estimators": (10, 1000)
        }
    },
    "AdaBoost": {
        "model": AdaBoostClassifier,
        "params": {
            "learning_rate": (0.0001, 1.0),
            "n_estimators": (10, 1000)
        }
    },
    "XGBoost": {
        "model": xgb.XGBClassifier,
        "params": {
            "alpha": (0.001, 1.0),
            "eta": (0.0001, 1.0),
            "eval_metric": ["auc"],
            "lambda": (0.001, 1.0),
            "max_depth": (1, 7),
            "n_estimators": (10, 1000),
            "objective": ["binary:logistic"],
            "subsample": (0.01, 1.0)
        }
    },
    "MLP": {
        "model": MLPClassifier,
        "params": {
            "activation": ["relu"],
            "alpha": (0.0001, 1.0),
            "batch_size": ["auto"],
            "early_stopping": [True],
            "hidden_layer_sizes": [(12, 8, 4)],
            "learning_rate": ["adaptive", "invscaling"],
            "max_iter": [100000],
            "solver": ["adam"]
        }
    },
    "SGD": {
        "model": SGDClassifier,
        "params": {
            "alpha": (0.0001, 60.0),
            "class_weight": ["balanced"],
            "early_stopping": [True],
            "eta0": (0.0001, 6.0),
            "learning_rate": ["adaptive", "constant"],
            "loss": ["modified_huber"],
            "max_iter": [10000],
            "penalty": ["elasticnet", "l2"],
            "l1_ratio": (0.001, 1.0)
        }
    },
    "KNN": {
        "model": KNeighborsClassifier,
        "params": {
            "algorithm": ["auto", "uniform"],
            "n_neighbors": (1, 95),
            "p": (1, 2),
            "weights": ["distance", "uniform"]
        }
    }
}
