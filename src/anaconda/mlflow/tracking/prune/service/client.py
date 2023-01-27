""" Defines MLFlow Tracking Server Pruning Client """

from datetime import datetime, timedelta
from typing import Optional

from mlflow.entities import Experiment, Run, ViewType
from mlflow.entities.model_registry import ModelVersion, RegisteredModel

from anaconda.enterprise.server.common.sdk import demand_env_var_as_int
from anaconda.mlflow.tracking.sdk import AnacondaMlFlowClient

from ..dto.pruneable import Pruneable


class PruneClient(AnacondaMlFlowClient):
    """MLFlow Tracking Server Pruning Client"""

    oldest_allowed_timestamp: Optional[float]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oldest_allowed_timestamp = round(
            (datetime.utcnow() - timedelta(days=demand_env_var_as_int(name="MLFLOW_TRACKING_ENTITY_TTL"))).timestamp()
            * 1000
        )
        print(f"Stale cut-off: {self.oldest_allowed_timestamp}")

    def is_model_version_pruneable(self, version: ModelVersion) -> bool:
        """
        For a specified `ModelVersion`, returns `True` if the version is pruneable, `False` otherwise.

        Parameters
        ----------
        version: ModelVersion
            The `ModelVersion` to review.

        Returns
        -------
        pruneable: bool
            The pruneability of the `ModelVersion`.
        """

        pruneable_flag: bool = False
        # Do not prune models which have had a stage set (Staging, Production, Archived)
        # If not assigned a stage the default is the literal string "None", it is NOT `None`.

        message_header: str = f"{version.last_updated_timestamp}:{version.version}:{version.current_stage}"

        if version.current_stage == "None":
            if version.last_updated_timestamp < self.oldest_allowed_timestamp:
                pruneable_flag = True
                print(f"{message_header} can be pruned")
            else:
                print(f"{message_header} not stale, can not be pruned")
        else:
            print(f"{message_header} stage is not None, can not be pruned")
        return pruneable_flag

    def get_pruneable_model_versions(self, versions: list[ModelVersion]) -> list[ModelVersion]:
        """
        Returns a list of models found to be pruneable.

        Parameters
        ----------
        models: list[RegisteredModel]
            The list of models to review for pruneability.

        Returns
        -------
        pruneable_models: list[ModelVersion]
            A list of `ModelVersion` objects to pruneable.
        """

        prunable_versions: list[ModelVersion] = []
        for version in versions:
            if self.is_model_version_pruneable(version=version):
                prunable_versions.append(version)
        return prunable_versions

    def get_stale_runs(self, experiment_ids: list[str]) -> list[Run]:
        """
        Queries MLFlow Tracking Server for runs:
        1. With end times older than the allowed (defined) max age.
        2. With statuses of either FINISHED or FAILED.

        Parameters
        ----------
        experiment_ids: list[str]
            A list of experiment ids to review.

        Returns
        -------
        stale_runs: list[Run]
            A list of runs which are stale.
        """

        # Get Stale Runs
        runs: list[Run] = []

        # The query language does not support `IN` clauses with status. We have to perform this as two queries.
        status_types: list[str] = ["FINISHED", "FAILED"]
        for status in status_types:
            query: str = f"attributes.end_time < {self.oldest_allowed_timestamp} AND attributes.status = '{status}'"
            query_runs: list[Run] = list(
                self.client.search_runs(
                    experiment_ids=experiment_ids, filter_string=query, run_view_type=ViewType.ACTIVE_ONLY
                )
            )
            runs += query_runs
        return runs

    @staticmethod
    def filter_runs(runs: list[Run], model_versions: list[ModelVersion]) -> list[Run]:
        """
        Creates a list of `Run` objects which to not have registered model versions.

        Parameters
        ----------
        runs: list[Run]
            The list of runs to filter
        model_versions: list[ModelVersion]
            The model versions to check for relationships.

        Returns
        -------
        runs: list[Run]
            A list of `Run` objects which to not have registered model versions.
        """

        # Filter out runs which still have registered model versions
        stale_run_ids: list[str] = [run.info.run_id for run in runs]
        # Generate Run Exclusion List
        run_exclusion_list: list[str] = []
        for version in model_versions:
            if version.run_id in stale_run_ids:
                run_exclusion_list.append(version.run_id)

        # Filter Runs
        final_run_list: list[Run] = [run for run in runs if run.info.run_id not in run_exclusion_list]

        return final_run_list

    def get_pruneable_runs(self, model_versions: list[ModelVersion]) -> list[Run]:
        """
        Generates a list of `Run` objects suitable for pruning.

        Parameters
        ----------
        model_versions: list[ModelVersion]
            The list of model versions to cross-reference when determining prune-ability.

        Returns
        -------
        runs: list[Run]
            A list of `Run` objects suitable for pruning.
        """

        # Get Experiments
        experiments: list[Experiment] = self.get_experiments()
        experiment_ids: list[str] = [experiment.experiment_id for experiment in experiments]

        print(f"Reviewing experiments {experiment_ids} for stale runs")

        # Get Stale Runs
        runs: list[Run] = self.get_stale_runs(experiment_ids=experiment_ids)
        print(f"Found {len(runs)} stale runs")

        # Filter out runs which still have registered model versions
        final_run_list: list[Run] = PruneClient.filter_runs(runs=runs, model_versions=model_versions)
        print(f"{len(final_run_list)} of the stale runs are pruneable")

        # Return the final result
        return final_run_list

    def get_pruneables(self) -> Pruneable:
        """
        Returns a `Pruneable` DTO for suitable for processing.

        Returns
        -------
        pruneable: Pruneable
            A `Pruneable` object.
        """

        # Get registered models
        models: list[RegisteredModel] = self.get_registered_models()
        registered_model_names: list[str] = [model.name for model in models]

        # Get registered model versions
        model_versions: list[ModelVersion] = []
        for model_name in registered_model_names:
            model_versions += list(self.get_model_versions(model_name=model_name))
            print(f"Registered model name: {model_name}, Total number of model versions: {len(model_versions)}")

        # Get model versions to prune
        prunable_model_versions: list[ModelVersion] = self.get_pruneable_model_versions(versions=model_versions)
        print(f"Number of pruneable model versions: {len(prunable_model_versions)}")

        # Get experiment runs to prune
        pruneable_runs: list[Run] = self.get_pruneable_runs(model_versions=model_versions)
        print(f"Number of pruneable experiment runs: {len(pruneable_runs)}")

        return Pruneable(runs=pruneable_runs, models=prunable_model_versions)

    def prune(self, pruneables: Pruneable, dry_run: bool) -> None:
        """
        Performs the MLFlow Tracking Server Pruning Process.

        Parameters
        ----------
        pruneables: Pruneable
            A `Pruneable` defining the resources to process.
        """

        print("[START] Stale Model Pruning")
        for model in pruneables.models:
            message_dict: dict = {
                "name": model.name,
                "version": model.version,
                "last_updated_timestamp": model.last_updated_timestamp,
            }

            if dry_run:
                # Report only
                print(f"[DRY RUN] {message_dict}")
            else:
                # Perform removal
                print(f"[DELETE] {message_dict}")
                self.client.delete_model_version(name=model.name, version=model.version)
        print("[COMPLETE] Stale Model Pruning")
        print("[START] Stale Run Pruning")
        for run in pruneables.runs:
            message_dict: dict = {
                "id": run.info.run_id,
                "end_time": run.info.end_time,
                "experiment_id": run.info.experiment_id,
            }

            if dry_run:
                # Report only
                print(f"[DRY RUN] {message_dict}")
            else:
                # Perform the removal
                print(f"[DELETE] {message_dict}")
                self.client.delete_run(run_id=run.info.run_id)
        print("[COMPLETE] Stale Run Pruning")
