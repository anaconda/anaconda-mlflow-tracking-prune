import unittest
from datetime import datetime, timedelta
from test.utils.mocks import MockFactory
from typing import Any, Optional
from unittest.mock import MagicMock, patch

from mlflow.entities import Experiment, Run
from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.store.entities import PagedList

from anaconda.mlflow.tracking.sdk import build_mlflow_client
from src.anaconda.mlflow.tracking.prune.dto.pruneable import Pruneable
from src.anaconda.mlflow.tracking.prune.service.client import PruneClient


class TestClient(unittest.TestCase):
    client: Optional[PruneClient]
    factory: MockFactory = MockFactory()

    def setUp(self) -> None:
        self.client = PruneClient(client=build_mlflow_client())
        self.client.client = MagicMock()

    # is_model_version_pruneable tests

    def test_is_model_version_pruneable_not_pruneable_due_to_stage(self):
        version: ModelVersion = self.factory.generate_mock_model_version()
        version._current_stage = "Staging"

        result: bool = self.client.is_model_version_pruneable(version=version)

        self.assertEqual(result, False)

    def test_is_model_version_pruneable_not_pruneable_due_to_age(self):
        version: ModelVersion = self.factory.generate_mock_model_version()
        version._last_updated_timestamp = (datetime.utcnow() + timedelta(days=1000)).timestamp() * 1000

        result: bool = self.client.is_model_version_pruneable(version=version)

        self.assertEqual(result, False)

    def test_is_model_version_pruneable_pruneable_stale(self):
        version: ModelVersion = self.factory.generate_mock_model_version()
        version._current_stage = "None"
        version._last_updated_timestamp = (datetime.utcnow() - timedelta(days=1000)).timestamp() * 1000

        result: bool = self.client.is_model_version_pruneable(version=version)

        self.assertEqual(result, True)

    # get_pruneable_model_versions tests

    def test_get_pruneable_model_versions(self):
        # Scenario 1: Empty List
        result: list[ModelVersion] = self.client.get_pruneable_model_versions(versions=[])
        self.assertEqual(len(result), 0)

        # Scenario 2: Very Stale
        mock_model_version: ModelVersion = self.factory.generate_mock_model_version()
        mock_model_version._last_updated_timestamp = 0
        mock_model_version._current_stage = "None"

        result: list[ModelVersion] = self.client.get_pruneable_model_versions(versions=[mock_model_version])
        self.assertEqual(len(result), 1)

    # get_stale_runs tests

    def test_get_stale_runs_empty(self):
        # Scenario 1: Empty list
        self.client.client.search_runs.return_value = []
        runs: list[Run] = self.client.get_stale_runs(experiment_ids=[])
        self.assertEqual(runs, [])

    def test_get_stale_runs_empty_results(self):
        # Scenario 2: Review Calls
        mock_run: Run = self.factory.generate_mock_run()
        self.client.client.search_runs.return_value = [mock_run]
        runs: list[Run] = self.client.get_stale_runs(experiment_ids=["mock"])

        self.client.client.search_runs.mock_calls[0].assert_called_with(
            experiment_ids=["mock"],
            filter_string=f"attributes.end_time < {self.client.oldest_allowed_timestamp} AND attributes.status = 'FAILED'",
            run_view_type=1,
        )
        self.client.client.search_runs.mock_calls[1].assert_called_with(
            experiment_ids=["mock"],
            filter_string=f"attributes.end_time < {self.client.oldest_allowed_timestamp} AND attributes.status = 'FINISHED'",
            run_view_type=1,
        )

    # filter_runs tests

    def test_filter_runs_empty(self):
        results: list[Run] = self.client.filter_runs(runs=[], model_versions=[])
        self.assertEqual(results, [])

    def test_filter_runs_no_exclusions(self):
        mock_run: Run = self.factory.generate_mock_run()
        mock_run._info = MagicMock()
        mock_run._info.run_id = "mock_run_id"

        results: list[Run] = self.client.filter_runs(runs=[mock_run], model_versions=[])
        self.assertEqual(results[0].info.run_id, "mock_run_id")

    def test_filter_runs_with_exclusions(self):
        mock_run_one: Run = self.factory.generate_mock_run()
        mock_run_one._info = MagicMock()
        mock_run_one._info.run_id = "mock_run_id_1"

        mock_run_two: Run = self.factory.generate_mock_run()
        mock_run_two._info = MagicMock()
        mock_run_two._info.run_id = "mock_run_id_2"

        mock_model_version: ModelVersion = self.factory.generate_mock_model_version()
        mock_model_version._run_id = "mock_run_id_1"

        results: list[Run] = self.client.filter_runs(
            runs=[mock_run_one, mock_run_two], model_versions=[mock_model_version]
        )
        self.assertEqual(results[0].info.run_id, "mock_run_id_2")

    # get_pruneable_runs tests

    def test_get_pruneable_runs_empty(self):
        # self.client.client.search_experiments.return_result = PagedList[Experiment](items=[], token=None)

        with patch("anaconda.mlflow.tracking.sdk.AnacondaMlFlowClient.get_experiments") as patched_get_experiments:
            with patch(
                "src.anaconda.mlflow.tracking.prune.service.client.PruneClient.get_stale_runs"
            ) as patched_get_stale_runs:
                with patch(
                    "src.anaconda.mlflow.tracking.prune.service.client.PruneClient.filter_runs"
                ) as patched_filter_runs:
                    patched_get_experiments.return_result = []
                    patched_get_stale_runs.return_value = []
                    patched_filter_runs.return_value = []

                    runs: list[Run] = self.client.get_pruneable_runs(model_versions=[])
                    self.assertEqual(runs, [])

    def test_get_pruneable_runs(self):
        mock_experiments: list[Experiment] = [self.factory.generate_mock_experiment()]

        mock_run: Run = self.factory.generate_mock_run()
        mock_run._info = MagicMock()
        mock_run._info.run_id = "mock_run_id_1"
        mock_stale_runs: list[Run] = [mock_run]

        mock_filtered_runs: list[Run] = [mock_run]

        def mock_get_experiments(self: Any) -> list[Experiment]:
            return mock_experiments

        def mock_get_stale_runs(self: Any, experiment_ids: list[str]) -> list[Run]:
            return mock_stale_runs

        def mock_filter_runs(runs: list[Run], model_versions: list[ModelVersion]) -> list[Run]:
            return mock_filtered_runs

        with patch("anaconda.mlflow.tracking.sdk.AnacondaMlFlowClient.get_experiments", mock_get_experiments):
            with patch(
                "src.anaconda.mlflow.tracking.prune.service.client.PruneClient.get_stale_runs", mock_get_stale_runs
            ):
                with patch(
                    "src.anaconda.mlflow.tracking.prune.service.client.PruneClient.filter_runs", mock_filter_runs
                ):

                    mock_model_version: ModelVersion = self.factory.generate_mock_model_version()

                    # Perform test
                    runs: list[Run] = self.client.get_pruneable_runs(model_versions=[mock_model_version])

                    # Review results
                    self.assertEqual(runs, [mock_run])

    # get_pruneables tests
    def test_get_pruneables_empty(self):
        def mock_get_registered_models(self: Any, filter_string: Optional[str] = None) -> list[RegisteredModel]:
            return []

        def mock_get_model_versions(self: Any, model_name: str) -> PagedList[ModelVersion]:
            return PagedList[ModelVersion](items=[], token=None)

        def mock_get_pruneable_model_versions(self: Any, versions: list[ModelVersion]) -> list[ModelVersion]:
            return []

        def mock_get_pruneable_runs(self: Any, model_versions: list[ModelVersion]) -> list[Run]:
            return []

        with patch(
            "anaconda.mlflow.tracking.sdk.AnacondaMlFlowClient.get_registered_models", mock_get_registered_models
        ):
            with patch("anaconda.mlflow.tracking.sdk.AnacondaMlFlowClient.get_model_versions", mock_get_model_versions):
                with patch(
                    "src.anaconda.mlflow.tracking.prune.service.client.PruneClient.get_pruneable_model_versions",
                    mock_get_pruneable_model_versions,
                ):
                    with patch(
                        "src.anaconda.mlflow.tracking.prune.service.client.PruneClient.get_pruneable_runs",
                        mock_get_pruneable_runs,
                    ):

                        # perform test
                        pruneable: Pruneable = self.client.get_pruneables()

                        # Review results
                        self.assertEqual(pruneable.runs, [])
                        self.assertEqual(pruneable.models, [])

    def test_get_pruneables(self):

        mock_registered_models: list[RegisteredModel] = [self.factory.generate_mock_registered_model()]

        mock_model_version_one: ModelVersion = self.factory.generate_mock_model_version()
        mock_model_version_one._run_id = mock_registered_models[0].name
        mock_paged_model_versions: PagedList[ModelVersion] = PagedList[ModelVersion](
            items=[mock_model_version_one], token=None
        )

        mock_model_version_two: ModelVersion = self.factory.generate_mock_model_version()
        mock_model_versions: list[ModelVersion] = [mock_model_version_two]

        mock_runs: list[Run] = [self.factory.generate_mock_run()]

        def mock_get_registered_models(arg: Any, filter_string: Optional[str] = None) -> list[RegisteredModel]:
            return mock_registered_models

        def mock_get_model_versions(self: Any, model_name: str) -> PagedList[ModelVersion]:
            return mock_paged_model_versions

        def mock_get_pruneable_model_versions(self: Any, versions: list[ModelVersion]) -> list[ModelVersion]:
            return mock_model_versions

        def mock_get_pruneable_runs(self: Any, model_versions: list[ModelVersion]) -> list[Run]:
            return mock_runs

        with patch(
            "anaconda.mlflow.tracking.sdk.AnacondaMlFlowClient.get_registered_models", mock_get_registered_models
        ):
            with patch("anaconda.mlflow.tracking.sdk.AnacondaMlFlowClient.get_model_versions", mock_get_model_versions):
                with patch(
                    "src.anaconda.mlflow.tracking.prune.service.client.PruneClient.get_pruneable_model_versions",
                    mock_get_pruneable_model_versions,
                ):
                    with patch(
                        "src.anaconda.mlflow.tracking.prune.service.client.PruneClient.get_pruneable_runs",
                        mock_get_pruneable_runs,
                    ):

                        # perform test
                        pruneable: Pruneable = self.client.get_pruneables()

                        # Review results
                        self.assertEqual(pruneable.runs, mock_runs)
                        self.assertEqual(pruneable.models, mock_model_versions)

    def test_prune_dry_run(self):
        # Set up test
        mock_run: Run = self.factory.generate_mock_run()
        mock_run._info = MagicMock()
        mock_run.info.run_id = "1"
        mock_run.info.end_time = "1"
        mock_run.info.experiment_id = "1"
        mock_model_version: ModelVersion = self.factory.generate_mock_model_version()

        # Perform test
        mock_pruneable: Pruneable = Pruneable(runs=[mock_run], models=[mock_model_version])
        self.client.prune(pruneables=mock_pruneable, dry_run=True)

        # Review results
        mock_client: MagicMock = self.client.client
        mock_client.delete_model_version.assert_not_called()
        mock_client.delete_run.assert_not_called()

    def test_prune(self):
        # Set up test
        mock_run: Run = self.factory.generate_mock_run()
        mock_run._info = MagicMock()
        mock_run.info.run_id = "1"
        mock_run.info.end_time = "1"
        mock_run.info.experiment_id = "1"
        mock_model_version: ModelVersion = self.factory.generate_mock_model_version()

        # Perform test
        mock_pruneable: Pruneable = Pruneable(runs=[mock_run], models=[mock_model_version])
        self.client.prune(pruneables=mock_pruneable, dry_run=False)

        # Review results
        mock_client: MagicMock = self.client.client
        mock_client.delete_model_version.assert_called_once_with(
            name=mock_model_version.name, version=mock_model_version.version
        )
        mock_client.delete_run.assert_called_once_with(run_id=mock_run.info.run_id)


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(TestClient())
