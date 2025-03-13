import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import config
import random


class DepthDataset(Dataset):
    def __init__(self):
        self.image_path = config.config["image_path"]
        self.depth_path = config.config["depth_path"]
        self.image_mode = config.config["image_mode"]
        self.in_type_uint8 = config.config["input_type_uint8"]
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_path, f"image_{idx:05d}.jpg")
        
        if self.image_mode == "RGB" or self.image_mode == "L":
            img = load_image_tensor(img_path, mode=self.image_mode, use_uint8=self.in_type_uint8) # 3 * H * W (rgb) or 1 * H * W (grayscale)
        elif self.image_mode == "YUV":
            img = load_image_array(img_path) # H * W * 3 (rgb)
            img = rgb2yuv(img) # H * W * 3 (yuv)
            img = T.ToTensor()(img).to(torch.float32) # 3 * H * W (yuv)
        
        depth_matrix = np.load(os.path.join(self.depth_path, f"array_{idx:05d}.npy"))
        depth_vector = extract_center_from_depthmatrix(depth_matrix) # 1 * H
        # Convert to float tensor
        depth_vector = torch.tensor(depth_vector, dtype=torch.float32)
        
        return img, depth_vector
        
    def __len__(self):
        # There's a hidden file called ".DS_store" (some mac thing) which means this method counts 1 more image
        # if you don't do -1
        return len(os.listdir(self.image_path)) - 1
    
    
def load_image_array(path):
    img = Image.open(path).convert("RGB")
    return np.array(img) # H * W * 3 (rgb) or H * W (grayscale)

    
def load_image_tensor(path, mode = "RGB", use_uint8=False):
    '''
        Load image from path and convert to tensor
    '''
    img = Image.open(path).convert(mode)
    if use_uint8: return T.PILToTensor()(img)
    return T.ToTensor()(img)


def extract_center_from_depthmatrix(depth_matrix):
    H, W = depth_matrix.shape
    center_depth = depth_matrix[:, W//2] # 1 * H
    return center_depth


def extract_center_from_depthmap(batch_depth_map):
    '''
        Extract the center line depth from the depth map
        
        ATTENTION: 
        Since the depth map is always rotated by 90 degrees, the center line is actually the center column
    '''
    _, _, _, W = batch_depth_map.shape
    center_depth = batch_depth_map[:, :, :, W//2] # N * 1 * H
    # Interpolate the depth to H/8
    downsampled_depth = F.avg_pool1d(center_depth, kernel_size=8, stride=8) # N * 1 * H/8
    downsampled_depth = downsampled_depth.squeeze(1) # N * H/8x
    # print(f"Extracted center depth shape: {downsampled_depth.shape}")
    return downsampled_depth


def load_train_val_dataset():
    '''
        Load dataset from config file
    '''
    dataset = DepthDataset()
    ratio = config.config["train_val_split"]
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [ratio, 1 - ratio])
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


def load_eval_dataset(num_imgs):
    '''
        Load num_imgs number of images for evaluation
    '''
    dataset = DepthDataset()
    random_indices = random.sample(range(len(dataset)), num_imgs)
    eval_dataset = torch.utils.data.Subset(dataset, random_indices)

    eval_data_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.config["num_workers"]
        )
    
    return eval_data_loader


def yuv2rgb(im):
    """ 
        Convert YUV to RGB 
        ref: https://github.com/guidoAI/YUV_notebook/blob/master/YUV_slices.py
    """
    if(np.max(im[:]) <= 1.0):
        im *= 255
        
    Y = im[:,:,0]
    U = im[:,:,1]
    V = im[:,:,2]
    
    R  = Y + 1.402   * ( V - 128 )
    G  = Y - 0.34414 * ( U - 128 ) - 0.71414 * ( V - 128 )
    B  = Y + 1.772   * ( U - 128 )

    rgb = im
    rgb[:,:,0] = R / 255.0
    rgb[:,:,1] = G / 255.0
    rgb[:,:,2] = B / 255.0

    inds1 = np.where(rgb < 0.0)
    for i in range(len(inds1[0])):
        rgb[inds1[0][i], inds1[1][i], inds1[2][i]] = 0.0
        
    inds2 = np.where(rgb > 1.0)
    for i in range(len(inds2[0])):
        rgb[inds2[0][i], inds2[1][i], inds2[2][i]] = 1.0
    return rgb


def rgb2yuv(rgb):
    """
        Convert RGB to YUV
        input: H * W * 3 (rgb)
        output: H * W * 3 (yuv)
    """
    if np.max(rgb) <= 1.0:
        rgb *= 255
    R = rgb[:,:,0]
    G = rgb[:,:,1]
    B = rgb[:,:,2]
    
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.14713 * R - 0.28886 * G + 0.436 * B
    V = 0.615 * R - 0.51499 * G - 0.10001 * B
    
    yuv = np.zeros(rgb.shape)
    yuv[:,:,0] = Y
    yuv[:,:,1] = U
    yuv[:,:,2] = V
    return yuv
