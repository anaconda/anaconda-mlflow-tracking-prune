""" Command For Pruning Process """
from ae5_tools import demand_env_var
from anaconda.enterprise.server.contracts import BaseModel

from .dto.pruneable import Pruneable
from .service.client import PruneClient


# pylint: disable=too-few-public-methods
class PruneCommand(BaseModel):
    """
    Prune Command
    Environment Independent MLFlow Tracking Server Resource Pruning Command.

    Attributes
    ----------
    pruner: PruneClient
        MLFlow Tracking Server Pruning Client
    """

    pruner: PruneClient

    def execute(self, dry_run: bool) -> None:
        """Default entry point for command. Executes the pruning process."""

        print(f"Pruning threshold set to: {int(demand_env_var(name='MLFLOW_TRACKING_ENTITY_TTL'))}")

        # Determine (by business logic) which runs and models we want to prune
        print("[START] Resource Pruneablilty Analysis")
        pruneables: Pruneable = self.pruner.get_pruneables()
        print("[COMPLETE] Resource Pruneablilty Analysis")

        # Call the MLFlow Tracking Server API to soft `delete` the artifacts.
        print("[START] Resource Pruning")
        self.pruner.prune(pruneables=pruneables, dry_run=dry_run)
        print("[COMPLETE] Resource Pruning")
