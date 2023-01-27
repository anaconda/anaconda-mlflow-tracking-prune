""" Pruneable Definition """

from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion

from anaconda.enterprise.server.contracts import BaseModel


# pylint: disable=too-few-public-methods
class Pruneable(BaseModel):
    """
    Prunable DTO

    Attributes
    ----------
    runs: list[Run]
        A list of pruneable runs
    models: list[ModelVersion]
        A list of pruneable models
    """

    runs: list[Run] = []
    models: list[ModelVersion] = []
