from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor

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
    elif name == "xgboost":
        # Auto-detect task from somewhere or assume classifier?
        # Usually config specifies. But 'xgboost' is generic.
        # Let's assume XGBClassifier for now since task is classification.
        # Ideally, we should have 'xgboost_clf' and 'xgboost_reg'.
        # For compatibility with my config:
        # Enable categorical support for XGBoost if data has strings
        defaults = {"enable_categorical": True}
        defaults.update(args)
        return XGBClassifier(**defaults)
    elif name == "xgboost_clf":
        return XGBClassifier(**args)
    elif name == "xgboost_reg":
        return XGBRegressor(**args)
    else:
        raise ValueError(f"Unknown model type: {name}")
