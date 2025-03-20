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
import h5py


class DepthDataset(Dataset):
    def __init__(self):
        self.uyvy_path = config.config["uyvy_path"]
        self.yuv_path = config.config["yuv_path"]
        self.image_path = config.config["image_path"]
        self.depth_path = config.config["depth_path"]
        self.h5_path = config.config["h5_path"]
        self.image_mode = config.config["image_mode"]
        self.in_type_uint8 = config.config["input_type_uint8"]
        self.output_channels = config.config["output_channels"]
        self.image_height = config.config["image_height"]
        
    def __getitem__(self, idx):
        
        if self.image_mode == "RGB" or self.image_mode == "L":
            img_path = os.path.join(self.image_path, f"image_{idx:05d}.jpg")
            img = self.load_image_tensor(img_path, mode=self.image_mode, use_uint8=self.in_type_uint8) # 3 * H * W (rgb) or 1 * H * W (grayscale)
        elif self.image_mode == "UYVY":
            uyvy_img_path = os.path.join(self.uyvy_path, f"image_{idx:05d}.npy")
            img = self.load_uyvy_tensor(uyvy_img_path)
        elif self.image_mode == "YUV":
            yuv_img_path = os.path.join(self.yuv_path, f"image_{idx:05d}.npy")
            img = self.load_yuv_tensor(yuv_img_path)

            if img.shape[1] != self.image_height: img = img[:, 1:,:] # Some of Tim's images have height > 520 ...

        with h5py.File(self.h5_path, "r") as f:
            depth_matrix = f[list(f.keys())[idx]][:]
            depth_matrix = depth_matrix / 255.0
            depth_vector = extract_depth_vector(depth_matrix, self.output_channels)

        # Convert to float tensor
        depth_vector = torch.tensor(depth_vector, dtype=torch.float32)
        
        return img, depth_vector
        
    def __len__(self):
        # There's a hidden file called ".DS_store" (some mac thing) which means this method counts 1 more image
        # if you don't do -1
        return len(os.listdir(self.image_path)) - 1
   
   
    def load_uyvy_tensor(self, path):
        '''
            Load uyvy image from path and convert to tensor
        '''
        uyvy = np.load(path, allow_pickle=True)
        uyvy = torch.tensor(uyvy, dtype=torch.float32).unsqueeze(0) # 1 * H * W
        return uyvy
        
        
    def load_image_array(self, path):
        img = Image.open(path).convert("RGB")
        return np.array(img) # H * W * 3 (rgb)

        
    def load_image_tensor(self, path, mode = "RGB", use_uint8=False):
        '''
            Load image from path and convert to tensor
        '''
        img = Image.open(path).convert(mode)
        img = img.resize((img.width // 2, img.height // 2), Image.ANTIALIAS)  # downsampling.
        if use_uint8: return T.PILToTensor()(img)
        return T.ToTensor()(img)
    
    def load_yuv_tensor(self, path):
        '''
            Load yuv image from path and convert to tensor
        '''
        yuv = np.load(path, allow_pickle=False)
        if self.in_type_uint8: yuv = torch.tensor(yuv, dtype=torch.uint8)
        else: yuv = torch.tensor(yuv, dtype=torch.float32)
        return yuv


    def extract_center_from_depthmatrix(self, depth_matrix):
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

def extract_depth_vector(depth_image, output_size):
    """
    Extracts a depth vector from the center column of a depth image.
    Each value in the vector represents the minimum depth value in a section of the image.
    
    Parameters:
    depth_image (numpy.ndarray): A 2D array representing the depth image.

    Returns:
    numpy.ndarray: A 1D array of length 16 containing minimum depth values for each section.
    """
    height, width = depth_image.shape
    center_x = width // 2
    margin = int(0.15 * width)  # +/-15% margin of total width
    depth_vector = np.zeros(output_size)
    
    for i in range(output_size):
        y_start = (i * height) // output_size
        y_end = ((i + 1) * height) // output_size
        
        # Extract the region of interest (ROI) around the center column
        roi = depth_image[y_start:y_end, max(0, center_x - margin):min(width, center_x + margin)]
        
        # Find the minimum depth value in this section
        depth_vector[i] = np.max(roi)
    
    return depth_vector


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
    
    return eval_data_loader, random_indices


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
        input: 3 * H * W (rgb)

        output: 3 * H * W (yuv)
    """
    R = rgb[0,:,:]
    G = rgb[1,:,:]
    B = rgb[2,:,:]

    print(R)
    print(G)
    
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.14713 * R - 0.28886 * G + 0.436 * B
    V = 0.615 * R - 0.51499 * G - 0.10001 * B
    
    yuv = torch.stack([Y, U, V], dim=0).to(torch.float32)

    return yuv