from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

def create_model(config: dict):
    name = config["type"]
    args = config.get("args", {})

    if name == "logistic_regression":
        return LogisticRegression(**args)
    elif name == "ridge":
        return Ridge(**args)
    elif name == "linear_regression":
        return LinearRegression(**args)
    elif name == "random_forest_clf":
        return RandomForestClassifier(**args)
    elif name == "random_forest_reg":
        return RandomForestRegressor(**args)
    elif name == "gb_clf":
        return GradientBoostingClassifier(**args)
    elif name == "gb_reg":
        return GradientBoostingRegressor(**args)
    elif name == "mlp_clf":
        return MLPClassifier(**args)
    elif name == "mlp_reg":
        return MLPRegressor(**args)
    else:
        raise ValueError(f"Unknown model type: {name}")
