from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.strategies import DeepSpeedStrategy
import os


class OpenMPIClusterEnvironment(ClusterEnvironment):
    def __init__(self, devices: int = 1) -> None:
        """ devices : devices per node"""
        super().__init__()
        self.devices = devices

    @property
    def creates_processes_externally(self) -> bool:
        """Return True if the cluster is managed (you don't launch processes yourself)"""
        return True

    def world_size(self) -> int:
        return int(os.environ.get("OMPI_COMM_WORLD_SIZE"))

    def set_world_size(self, size: int) -> None:
        pass

    def global_rank(self) -> int:
        return int(os.environ.get("OMPI_COMM_WORLD_RANK"))

    def set_global_rank(self, rank: int) -> None:
        pass

    def local_rank(self) -> int:
        return int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"))

    def node_rank(self) -> int:
        print(f'node_rank : {int(os.environ.get("OMPI_COMM_WORLD_RANK",0)) // int(self.devices)}') # debugging
        # this may not exist, defaulting to 0
        return int(os.environ.get("OMPI_COMM_WORLD_RANK",0)) // int(self.devices)

    @property
    def main_address(self) -> str:
        # AZ_BATCH_MASTER_NODE should be defined when num_nodes > 1
        if "AZ_BATCH_MASTER_NODE" in os.environ:
            print(f"main_address : {os.environ.get('AZ_BATCH_MASTER_NODE').split(':')[0]}") # debugging
            return os.environ.get("AZ_BATCH_MASTER_NODE").split(':')[0]
        elif "AZ_BATCHAI_MPI_MASTER_NODE" in os.environ:
            print(f"main_address : {os.environ.get('AZ_BATCHAI_MPI_MASTER_NODE')}") # debugging
            return os.environ.get("AZ_BATCHAI_MPI_MASTER_NODE")
        else:
            raise("main_address not found")

    @property
    def main_port(self) -> int:
        # AZ_BATCH_MASTER_NODE should be defined when num_nodes > 1
        if "AZ_BATCH_MASTER_NODE" in os.environ:
            print(f"main_port : {os.environ.get('AZ_BATCH_MASTER_NODE').split(':')[1]}") # debugging
            return int(os.environ.get("AZ_BATCH_MASTER_NODE").split(':')[1])
        else:
            return int(47586) # set port to arbitrary high number

    @staticmethod
    def detect() -> bool:
        return "OMPI_COMM_WORLD_SIZE" in os.environ
    
def set_strategy(args):
    if args.strategy == "ddp":
        return "ddp"
    elif args.strategy == "deepspeed":
        return DeepSpeedStrategy(
            stage = 3,
            cluster_environment = OpenMPIClusterEnvironment(devices=args.num_devices)
        )
    else:
        return None