# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""
Sample entry script for large scale training of Computer Vision models on Azure.

When running the script for the first time, set the BUILD_ENVIRONMENT variable
below to True to create an Azure custom environment. Then set it to False when
launching actual training runs.
"""

from azure.ai.ml import MLClient, Input, command, PyTorchDistribution
from azure.ai.ml.entities import Environment

from azure.identity import DefaultAzureCredential


CODE_DIRECTORY_NAME = "./"
COMMAND = (
    "python mae_trainer.py --dataset_directory_name ${{inputs.data}} --trainset ${{inputs.trainset}} --valset ${{inputs.valset}} "
    "--num_epochs ${{inputs.num_epochs}} --batch_size ${{inputs.batch_size}} --num_workers ${{inputs.num_workers}} "
    "--num_nodes ${{inputs.num_nodes}} --num_devices ${{inputs.num_devices}} --strategy ${{inputs.strategy}} --img_size ${{inputs.img_size}} "
    "--learning_rate ${{inputs.learning_rate}} --weight_decay ${{inputs.weight_decay}}"
)

# Enter the details of your Azure ML workspace.
SUBSCRIPTION_ID = "2630c4b9-c627-494d-8a87-a5002e5e7f8e"
RESOURCE_GROUP = "GODZILLA"
WORKSPACE_NAME = "GODZILLA"
CLUSTER_NAME = "godzilla-compute-1"
ENVIRONMENT_NAME = "acpt-pytorch-20-andre:1"
EXPERIMENT_NAME = "MAE_PTL_MOUNT"

NUM_NODES = 23
NUM_GPUS_PER_NODE = 8
SHARED_MEMORY_SIZE_STR = "1000G"
DATA_MODE = "mount_no_caching"  # alternative: "download"

NUM_EPOCHS = 100
BATCH_SIZE = 170
NUM_WORKERS = 12
IMG_SIZE = 224
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.05


if __name__ == "__main__":
    # Create the MLClient object.
    credential = DefaultAzureCredential(
        exclude_shared_token_cache_credential=True,
        exclude_visual_studio_code_credential=False,
    )
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )
    

    # Set job parameters.
    command_parameters = dict(
        code=CODE_DIRECTORY_NAME,
        command=COMMAND,
        inputs=dict(
            data=Input(
                type="uri_folder",
                path="azureml://subscriptions/2630c4b9-c627-494d-8a87-a5002e5e7f8e/resourcegroups/GODZILLA/workspaces/GODZILLA/datastores/datastorepremium/paths/flatzilla/",
                mode="ro_mount",
            ),
            trainset=Input(
                type="uri_file",
                path="azureml://subscriptions/2630c4b9-c627-494d-8a87-a5002e5e7f8e/resourcegroups/GODZILLA/workspaces/GODZILLA/datastores/datastorepremium/paths/MAE/train/trainset_mae_godzilla_aml.csv",
                mode="ro_mount",
            ),
            valset=Input(
                type="uri_file",
                path="azureml://subscriptions/2630c4b9-c627-494d-8a87-a5002e5e7f8e/resourcegroups/GODZILLA/workspaces/GODZILLA/datastores/datastorepremium/paths/MAE/val/valset_mae_godzilla_aml.csv",
                mode="ro_mount",
            ),
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            num_nodes=NUM_NODES,
            num_devices=NUM_GPUS_PER_NODE,
            strategy="ddp",
            img_size=IMG_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
            
        ),
        environment=ENVIRONMENT_NAME, #+ "@latest",
        shm_size=SHARED_MEMORY_SIZE_STR,
        compute=CLUSTER_NAME,
        instance_count=NUM_NODES,
        experiment_name=EXPERIMENT_NAME,
    )
    if DATA_MODE == "mount_no_caching":
        command_parameters.update(
            dict(
                environment_variables=dict(
                    DATASET_MOUNT_BLOCK_BASED_CACHE_ENABLED=True,  # enable block-based caching
                    DATASET_MOUNT_BLOCK_FILE_CACHE_ENABLED=False,  # disable caching on disk
                    DATASET_MOUNT_MEMORY_CACHE_SIZE=0,  # disabling in-memory caching
                )
            )
        )
    if NUM_NODES > 1:
        command_parameters.update(
            dict(
                distribution=PyTorchDistribution(
                    node_count=NUM_NODES,
                    process_count_per_instance=NUM_GPUS_PER_NODE,
                )
            )
        )

    # Submit the job.
    job = command(**command_parameters)
    ml_client.create_or_update(job)