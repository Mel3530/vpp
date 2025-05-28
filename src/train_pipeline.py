import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.utils import evaluate_all_metrics, heuristic_score
from sklearn.base import clone
from deap import base, creator, tools, algorithms
import random

def run_full_pipeline(model_name, config, X_train, y_train, X_test, y_test, X_val, y_val):
    base_model = config["model"]
    param_space = config["params"]

    def create_model(individual):
        params = {}
        for (key, bounds), value in zip(param_space.items(), individual):
            if isinstance(bounds, tuple):
                params[key] = type(bounds[0])(value)
            else:
                params[key] = value
        return base_model(**params)

    def objective(individual):
        model = create_model(individual)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        train_aucs, test_aucs = [], []

        for train_idx, test_idx in skf.split(X_train, y_train):
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            train_proba = model.predict_proba(X_train.iloc[train_idx])[:, 1]
            test_proba = model.predict_proba(X_train.iloc[test_idx])[:, 1]
            train_aucs.append(roc_auc_score(y_train.iloc[train_idx], train_proba))
            test_aucs.append(roc_auc_score(y_train.iloc[test_idx], test_proba))

        return heuristic_score(np.mean(train_aucs), np.mean(test_aucs)),

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    attributes = []
    for key, bounds in param_space.items():
        if isinstance(bounds, tuple):
            attributes.append(lambda: random.uniform(*bounds))
        else:
            attributes.append(lambda: random.choice(bounds))
    toolbox.register("individual", tools.initCycle, creator.Individual, attributes, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", objective)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=10)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, halloffame=hof, verbose=False)

    best_model = create_model(hof[0])
    best_model.fit(X_train, y_train)

    results = {"Model": model_name}

    for name, X, y in [("Test", X_test, y_test), ("Validation", X_val, y_val)]:
        preds = best_model.predict(X)
        probas = best_model.predict_proba(X)[:, 1]
        metrics = evaluate_all_metrics(y, preds, probas)
        results.update({f"{name}_{k}": v for k, v in metrics.items()})

    return results
