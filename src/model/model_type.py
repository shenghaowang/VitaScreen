from enum import Enum


class ModelType(Enum):
    CatBoost = "catboost"
    IGTD = "igtd"
    NCTD = "nctd"
    MLP = "mlp"
