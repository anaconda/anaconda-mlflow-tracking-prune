""" AE5 Project Handler """
import sys
from argparse import ArgumentParser, Namespace

from ae5_tools import load_ae5_user_secrets
from anaconda.mlflow.tracking.sdk import build_mlflow_client

from .command import PruneCommand
from .service.client import PruneClient

if __name__ == "__main__":
    # This function is meant to provide a handler mechanism between the AE5 deployment arguments
    # and those required by the called process (or service).

    # arg parser for the standard anaconda-project options
    parser = ArgumentParser(
        prog="mlflow-tracking-server-prune-launch-wrapper", description="mlflow tracking server prune launch wrapper"
    )
    parser.add_argument("--anaconda-project-host", action="append", default=[], help="Hostname to allow in requests")
    parser.add_argument("--anaconda-project-port", action="store", default=8086, type=int, help="Port to listen on")
    parser.add_argument(
        "--anaconda-project-iframe-hosts",
        action="append",
        help="Space-separated hosts which can embed us in an iframe per our Content-Security-Policy",
    )
    parser.add_argument(
        "--anaconda-project-no-browser", action="store_true", default=False, help="Disable opening in a browser"
    )
    parser.add_argument(
        "--anaconda-project-use-xheaders", action="store_true", default=False, help="Trust X-headers from reverse proxy"
    )
    parser.add_argument("--anaconda-project-url-prefix", action="store", default="", help="Prefix in front of urls")
    parser.add_argument(
        "--anaconda-project-address",
        action="store",
        default="0.0.0.0",
        help="IP address the application should listen on",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Flag for controlling actual application of the system level change",
    )

    # Load command line arguments
    cli_args: Namespace = parser.parse_args(sys.argv[1:])
    print(cli_args)

    # load defined environmental variables
    load_ae5_user_secrets(silent=False)

    # Create our pruning client
    pruning_client: PruneClient = PruneClient(client=build_mlflow_client())

    # Execute the pruning
    PruneCommand(pruner=pruning_client).execute(dry_run=cli_args.dry_run)
