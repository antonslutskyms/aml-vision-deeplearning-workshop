# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import torch, os, sys
#from deepspeed.ops.adam import FusedAdam
from azureml.core import Run
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichModelSummary
from os.path import join
from datasets.mae_datasets import make_dataloaders
import argparse
from utils.utils import display_environment
from utils.openmpi import set_strategy
from torch import optim
from models.models_mae import mae_vit_huge_patch14_dec512d8b
import timm
from torch import optim, nn, utils
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torch, os, sys
#from deepspeed.ops.adam import FusedAdam
from azureml.core import Run
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichModelSummary
from os.path import join
from datasets.mae_datasets import make_dataloaders
import argparse
from utils.utils import display_environment
from utils.openmpi import set_strategy
from torch import optim
from models.models_mae import mae_vit_huge_patch14_dec512d8b
import timm



class MAELightning(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args       
        self.model = mae_vit_huge_patch14_dec512d8b()

    def training_step(self, batch, _):
        # Get the images and labels.
        X = batch['image']

        # Compute the training loss.
        loss, _, _ = self.model(X)

        # Log the training loss.
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, _):
        # Get the images and labels.
        X = batch['image']
        val_loss, _, _ = self.model(X)
       
        self.log("val_loss", val_loss.item(), prog_bar=True, sync_dist=True) 
        
        return {"val_loss":val_loss.item()}

    def configure_optimizers(self):
        # Make the optimizer and learning rate scheduler.
        optimizer = optim.AdamW(
            self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay
        )
        return [optimizer]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_directory_name", type=str)
    parser.add_argument("--trainset", type=str)
    parser.add_argument("--valset", type=str)   
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", required=False, default=1, type=int)
    parser.add_argument("--num_nodes", required=False, default=1, type=int)
    parser.add_argument("--num_devices", required=False, default=1, type=int)
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--strategy", required=False, default="ddp")
    parser.add_argument("--experiment_name",type=str)
    parser.add_argument("--precision",type=int, default=32)
    parser.add_argument("--learning_rate",type=float, default=1e-4)
    parser.add_argument("--weight_decay",type=float, required=False, default=0)
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    args = parse_args()
    # If running on azure, get the active tracking uri and run id
    # otherwise, use the workspace to get a tracking uri
    active_run = Run.get_context() #  active run azureml object
    # offline = False
    try:
        print(active_run.experiment)
        tracking_uri=active_run.experiment.workspace.get_mlflow_tracking_uri()
        run_id = active_run.id
    except:
        print(f"WARNING ! Could not connect to the MLFlow tracking uri, please check !") #offline = True

    seed_everything(102938, workers = True)

    print("creating training and validation sets...")
    train_loader , val_loader = make_dataloaders(args)
    print("done")


    print("creating model...")
    model = MAELightning(args=args)
    print("done")

    logger = MLFlowLogger(
                experiment_name=args.experiment_name,
                tracking_uri=tracking_uri,
                run_id=run_id
    )

    display_environment("__main__")
    
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", 
                                          mode="min",
                                          save_top_k=999,
                                          verbose=True,
                                          dirpath="./outputs/checkpoints/",
                                          filename="{epoch}-{val_loss:.2f}",
                                          save_weights_only=True,
                                          auto_insert_metric_name=True)  
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    model_summary = RichModelSummary(max_depth=1)
    
    trainer = Trainer(
        num_nodes=args.num_nodes,
        accelerator='gpu',
        devices=args.num_devices,
        log_every_n_steps=1,
        logger=logger,
        num_sanity_val_steps=2,
        max_epochs=args.num_epochs,
        enable_model_summary=False,
        callbacks = [checkpoint_callback,lr_monitor,model_summary],
        #strategy=set_strategy(args),
        strategy=args.strategy,
        precision=args.precision
    )

    display_environment("__main__")

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f"""trainer.local_rank: {trainer.local_rank}
trainer.global_rank : {trainer.global_rank}
trainer.world_size : {trainer.world_size}
""")
