import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import config


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import config


class DepthDataset(Dataset):
    def __init__(self):
        self.image_path = config.config["image_path"]
        self.depth_path = config.config["depth_path"]
        
        # Ensure only existing images are loaded
        self.valid_indices = []
        for idx in range(len(os.listdir(self.image_path))):
            img_file = os.path.join(self.image_path, f"image_{idx:05d}.jpg")
            depth_file = os.path.join(self.depth_path, f"image_{idx:05d}.jpg")
            if os.path.exists(img_file) and os.path.exists(depth_file):
                self.valid_indices.append(idx)

    def __getitem__(self, idx):
        valid_idx = self.valid_indices[idx]
        img_path = os.path.join(self.image_path, f"image_{valid_idx:05d}.jpg")
        depth_path = os.path.join(self.depth_path, f"image_{valid_idx:05d}.jpg")
        
        img = load_image(img_path)  # 3 * H * W
        depth = load_image(depth_path, mode="L")  # 1 * H * W, convert to grayscale maps
        
        return img, depth
        
    def __len__(self):
        return len(self.valid_indices)
    

def load_image(path, mode="RGB"):
    '''
        Load image from path and convert to tensor
    '''
    img = Image.open(path).convert(mode)
    return T.ToTensor()(img)


def load_train_val_dataset():
    '''
        Load dataset from config file
    '''
    dataset = DepthDataset()
    ratio = config.config["train_val_split"]
    train_size = int(ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.config["batch_size"], 
        shuffle=True, 
        num_workers=config.config["num_workers"]
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config.config["batch_size"], 
        shuffle=False, 
        num_workers=config.config["num_workers"]
    )
    
    return train_dataloader, val_dataloader

# class DepthDataset(Dataset):
#     def __init__(self):
#         self.image_path = config.config["image_path"]
#         self.depth_path = config.config["depth_path"]
        
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.image_path, f"image_{idx:05d}.jpg")
#         depth_path = os.path.join(self.depth_path, f"image_{idx:05d}.jpg")
        
#         img = load_image(img_path) # 3 * H * W
#         depth = load_image(depth_path, mode="L") # 1 * H * W, convert to grayscale maps
        
#         return img, depth
        
#     def __len__(self):
#         return len(os.listdir(self.image_path))
    
    
# def load_image(path, mode = "RGB"):
#     '''
#         Load image from path and convert to tensor
#     '''
#     img = Image.open(path).convert(mode)
#     return T.ToTensor()(img)


# def load_train_val_dataset():
#     '''
#         Load dataset from config file
#     '''
#     dataset = DepthDataset()
#     ratio = config.config["train_val_split"]
#     train_dataset, val_dataset = torch.utils.data.random_split(dataset, [ratio, 1 - ratio])
#     train_dataloader = DataLoader(
#         train_dataset, 
#         batch_size=config.config["batch_size"], 
#         shuffle=True, 
#         num_workers=config.config["num_workers"]
#         )
#     val_dataloader = DataLoader(
#         val_dataset, 
#         batch_size=config.config["batch_size"], 
#         shuffle=False, 
#         num_workers=config.config["num_workers"]
#         )
#     return train_dataloader, val_dataloader
