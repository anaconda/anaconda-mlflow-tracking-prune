import unittest
from typing import Any
from unittest.mock import MagicMock

from anaconda.mlflow.tracking.sdk import build_mlflow_client
from src.anaconda.mlflow.tracking.prune.command import PruneCommand
from src.anaconda.mlflow.tracking.prune.service.client import PruneClient


class TestCommand(unittest.TestCase):
    def test_(self):
        # setup
        pruning_client: PruneClient = PruneClient(client=build_mlflow_client())
        mock_prune_client: PruneClient = MagicMock()
        mock_prune_client.get_pruneables.return_value = "MOCK"
        command: PruneCommand = PruneCommand(pruner=pruning_client)
        command.pruner = mock_prune_client

        # Execute
        command.execute(dry_run=False)

        # Validate
        mock_prune_client.get_pruneables.assert_called_once()
        mock_prune_client.prune.assert_called_once_with(pruneables="MOCK", dry_run=False)


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(TestCommand())
