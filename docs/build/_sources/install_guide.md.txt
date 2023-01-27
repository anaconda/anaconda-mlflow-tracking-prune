#  Install Guide

Configuration
--------
The below variables control the configuration of the pruning process.  See (Deployment) below for information on when each is needed and created.

These should be defined as AE5 secrets within the service account running the tracking server.  Alternatively they can also be set within the `anaconda-project.yml` project files.

### Variables

1. `MLFLOW_TRACKING_URI`

    **Description**

    Remote Tracking Server URI 
   
    **Details**
      * This should be the static endpoint assigned to the private project deployment.


2. `MLFLOW_REGISTRY_URI`

    **Description**
    
    Model Registry URI
   
    **Details**
      * This should be the static endpoint assigned to the private project deployment.


3. `MLFLOW_TRACKING_TOKEN`

    **Description**
    
   AE5 Private Endpoint Access Token
   
    **Details**
      * Private deployment access token (bearer token)


4. `MLFLOW_TRACKING_ENTITY_TTL`

    **Description**
    
   	The age (measured in days) at which a resource within the MLFlow Tracking Server is considered stale.
   
    **Details**
      * Integer value.
      * Measured in days.

    **Default**

    * 30

Deployment
--------
1. **Use Dedicated Service Account**
     * The service account used to run the MLFlow Tracking Server **SHOULD** also run this service.

2. **Configure Python Environment**

    The deployment environment **MUST** occur within a conda environment with (at least):

        channels:
          - defaults
          - ae5-admin
        dependencies:
          - python=3
          - ae5-tools

3. **Configure AE5 Tools**

    Since the project needs to run under the user account created earlier we need to ensure we deploy to this user account.  This can be accomplished by either authenticating as the service account itself, or by an AE5 administrator who impersonates the service account at deployment time.  See [ae5-tools](https://github.com/Anaconda-Platform/ae5-tools) for additional details.
                    
4. **Download Latest Release**

     The latest releases can be found [here](https://github.com/Anaconda-Platform/anaconda-mlflow-tracking-prune/releases/latest).

5. **[Optional] Slip Stream Customizations**
 
    The default works for most scenarios.  However, the archive can be updated and repackaged for specific deployments if needed. This could be useful in scenarios where changes to dependency versions, client specific commands, or default variables must occur prior to deployment.

6. **Upload Project**

    This can be accomplished using ae5 tools.

      **Example**
      > ae5 project upload --name "anaconda.mlflow.tracking.prune" anaconda.mlflow.tracking.prune.x.y.z.tar.gz

12. **Create Prune Schedule**
    It is recommended to set up a schedule for the process so that it is occurring regularly.

    * These environment variables **MUST** be defined as ae5 secrets, within the anaconda-project.yml, or passed to the ae5 job create command are variables (see below).

    | Variable                   |
    |----------------------------|
    | MLFLOW_TRACKING_URI        |
    | MLFLOW_REGISTRY_URI        |
    | MLFLOW_TRACKING_TOKEN      |
    | MLFLOW_TRACKING_ENTITY_TTL |

    **Examples**

    > ae5 job create --command "Prune" --schedule "0 0 * * *" --name "scheduled anaconda.mlflow.tracking.prune" "anaconda.mlflow.tracking.prune"

    > ae5 job create --command "Prune" --schedule "0 0 * * *" --name "scheduled anaconda.mlflow.tracking.prune" "anaconda.mlflow.tracking.prune" -variable MLFLOW_TRACKING_ENTITY_TTL=10

Automated Deployments
--------

* See [automation notebook](https://github.com/Anaconda-Platform/anaconda-enterprise-mlops-orchestration/blob/main/notebooks/deployment/tracking_server_prune.ipynb) for an example.

Anaconda Project Runtime Commands
--------
These commands are used to start the server and perform the various administrative tasks.

| Command         | Environment | Description                        |
|-----------------|-------------|:-----------------------------------|
| Report          | Runtime     | Launches Report Only Prune Process |
| Prune           | Runtime     | Launches Prune Process             |

