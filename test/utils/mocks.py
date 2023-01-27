from datetime import datetime

from mlflow.entities import Experiment, Run
from mlflow.entities.model_registry import ModelVersion, RegisteredModel


class MockFactory:
    counter: int = 0

    def generate_mock_experiment(self) -> Experiment:
        experiment = Experiment(
            artifact_location="",
            creation_time=datetime.now(),
            experiment_id=str(self.counter),
            last_update_time=datetime.now(),
            lifecycle_stage="",
            name=str(self.counter),
            tags={},
        )
        self.counter += 1
        return experiment

    def generate_mock_run(self) -> Run:
        run = Run(run_info={}, run_data={})
        self.counter += 1
        return run

    def generate_mock_model_version(self) -> ModelVersion:
        model_version = ModelVersion(
            name=f"mock-model-{str(self.counter)}",
            version=str(self.counter),
            creation_timestamp=datetime.now(),
        )
        self.counter += 1
        return model_version

    def generate_mock_registered_model(self) -> RegisteredModel:
        registered_model = RegisteredModel(name=str(self.counter))
        self.counter += 1
        return registered_model
