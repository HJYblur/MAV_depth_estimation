import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
import pandas as pd


image_dir = "./data/depth_map/"
output_csv = "./data/center_columns.csv"

os.makedirs("output_directory", exist_ok=True)

num_images = len([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

print(f"Number of images in the directory: {num_images}")

transform = T.Compose([
    T.Resize((256, 256)),  # Resize 
    T.ToTensor()  #(C, H, W)
])


image_tensors = []
filenames = []  

for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.jpg', '.jpeg')): 
        img_path = os.path.join(image_dir, filename)
        image = Image.open(img_path)  
        image_tensor = transform(image)  
        image_tensors.append(image_tensor)
        filenames.append(filename)



if image_tensors:
    batch_images = torch.stack(image_tensors)  # Shape: (N, 1, H, W)



def extract_center_from_depthmap(batch_depth_map):
    '''
        re-used from dataset.py
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



center_column_output = extract_center_from_depthmap(batch_images).numpy()  

# Save to CSV file
df = pd.DataFrame(center_column_output, index=filenames)
df.to_csv(output_csv, header=False)
print(f"Center column vectors saved to {output_csv}")


print(f"Extracted center column shape: {center_column_output.shape}")