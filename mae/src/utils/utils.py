import os
divider_str="-"*40

def get_env_display_text(var_name):
    var_value = os.environ.get(var_name, "")
    return f"{var_name} = {var_value}"

def display_environment(header='Environmental variables'):
    """
    Print a few environment variables of note
    """
    variable_names = [
        "PL_GLOBAL_SEED",
        "PL_SEED_WORKERS",
        "AZ_BATCH_MASTER_NODE",
        "AZ_BATCHAI_MPI_MASTER_NODE",
        "MASTER_ADDR",
        "MASTER_ADDRESS",
        "MASTER_PORT",
        "RANK",
        "NODE_RANK",
        "LOCAL_RANK",
        "GLOBAL_RANK",
        "WORLD_SIZE",
        "NCCL_SOCKET_IFNAME",
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "OMPI_COMM_WORLD_SIZE",
        "OMPI_COMM_WORLD_LOCAL_SIZE"
    ]

    var_text = "\n".join([get_env_display_text(var) for var in variable_names])
    print(f"\n{header}:\n{divider_str}\n{var_text}\n{divider_str}\n")
