from torch.utils.data import Dataset, DataLoader,get_worker_info
from PIL import Image
import mltable
from transforms.mae_transforms import SquarePad
import torchvision.transforms as transforms
import numpy as np
import torch
from os.path import basename, join
from PIL import UnidentifiedImageError
import pandas as pd

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
   
class ImageDatasetMP(Dataset):
    def __init__(
        self, 
        source_dir,
        df, 
        transform=None

        ):
        super(Dataset).__init__()

        self.source_dir = source_dir
        self.transform = transform
        self.df = df
        self.missing_files = []
    
    def __len__(self):
       
        return len(self.df)
    
    def __getitem__(self, idx):
        
        #path = filezilla/filename.png ...
        #img_path = join(self.source_dir, basename(self.df.loc[idx, 'filename_fixed']))
        img_path = join(self.source_dir, self.df.loc[idx, 'filename_fixed'])
        #print(img_path)
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return {'image' : image}
        except:
            # Log the error or print a message indicating the corrupted image
            print(f"Error loading image {img_path}")
            self.missing_files.append(img_path)  # Store the path of the missing file
            with open('./outputs/missing.txt', 'a') as file:
                file.write(f"\n{img_path}")
            return None  # Placeholder for corrupted image
        


    # def save_missing_files(self):
    #     # Save the paths of missing files into a text file
    #     with open('./outputs/missing.txt', 'a') as file:
    #         file.write('\n'.join(self.missing_files))

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def make_dataloaders(args):
    
    print("Creating dataloaders.......... !!!!!!!!!!!!!!!!!!!!!")

    tfs_train = transforms.Compose([
            SquarePad(),
            #transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomResizedCrop(size=args.img_size, 
                                         scale=(0.7, 1.0), 
                                         ratio=(0.75, 1.3333333333333333)
                                         ),
            transforms.RandAugment(),
            transforms.RandomVerticalFlip(0.1),
            transforms.RandomHorizontalFlip(0.1),
            transforms.GaussianBlur(3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    tfs_val = transforms.Compose([
            SquarePad(),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomResizedCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    df_train = pd.read_csv(args.trainset)
    df_val = pd.read_csv(args.valset)

    print("**********************************************************")
    print("Trainset: ", df_train.head())
    print("Valset: ", df_val.head())

    print("!!!!! ********************************************************** !!!!!!!")

    train_dataset = ImageDatasetMP(source_dir=args.dataset_directory_name,
                                   df=df_train,
                                   transform=tfs_train)
    
    val_dataset = ImageDatasetMP(source_dir=args.dataset_directory_name,
                                 df=df_val,
                                 transform=tfs_val)
    
    print("????? ********************************************************** ??????")
    print("train_dataset[1]", train_dataset[1])

    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn, shuffle=False
    )
    
    return train_loader, val_loader