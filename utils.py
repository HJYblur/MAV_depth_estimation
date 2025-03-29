import os
import time
import logging
import argparse
import h5py
import random
from PIL import Image
from matplotlib import pyplot as plt
import config
import numpy as np
import cv2


def parse_args():
    '''
        Parse arguments from command line
    '''
    parser = argparse.ArgumentParser(description="Depth Estimation")
    parser.add_argument("--mode", type=str, default=None, help="Mode: data/train/eval")
    parser.add_argument("--h5file", type=str, default=None, help="H5 file name")
    parser.add_argument("--add_data", type=bool, default=True, help="Add new data or not")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--model_id", type=int, default=0, help="Model ID for evaluation")
    args = parser.parse_args()
    return args


def init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # File handler for file output
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    file_handler = logging.FileHandler(os.path.join(config.config["save_log_path"], f"log_{current_time}.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # log device info
    for i, item in enumerate(config.config.items()):
        if i == 8: break
        logger.info(item.__str__())
    
    return logger


def data_preprocess(h5_path, raw_image_path, append = False):
    '''
        Load depth matrix from h5 file and store as numpy, also store the original image
    '''
    depth_path = config.config["depth_path"]
    image_path = config.config["image_path"]
    os.makedirs(depth_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)
    
    # assert len(os.listdir(image_path)) == len(os.listdir(depth_path)), "Number of images and depth maps do not match"
    prev_index = len(os.listdir(depth_path)) if append else 0
    print(f"Previous index: {prev_index}")
    
    raw_image_list = os.listdir(raw_image_path)
    with h5py.File(h5_path, "r") as f:
        for i, key in enumerate(f.keys()):
            index = i + prev_index
            print(f"Processing {key}, index: {index}")
            depth_array = f[f[key].name][:]
            depth_array = depth_array / 255.0

            if key in raw_image_list:
                # Rename the image file
                os.rename(os.path.join(raw_image_path, key), os.path.join(image_path, f"image_{index:05d}.jpg"))
                # Save the depth matrix
                np.save(os.path.join(depth_path, f"array_{index:05d}.npy"), depth_array)
                print(f"Move {key} to image_{index:05d}.jpg, Save depth to array_{index:05d}.npy") 
            else:
                print(f"Image {key} not found")
                continue
    
    
def convert_images_to_uyvy(input_folder, output_folder):
    """
    Converts all RGB images in the input folder to UYVY format and saves them as .npy files.

    Args:
        input_folder (str): Path to the folder containing input RGBD images.
        output_folder (str): Path to save UYVY .npy files.
    """
    os.makedirs(output_folder, exist_ok=True)

    for fname in sorted(os.listdir(input_folder)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(input_folder, fname)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None or img.shape[2] < 3:
            print(f"Skipping {fname}: not a valid RGB image.")
            continue

        base_name = os.path.splitext(fname)[0]
        rgb = img[:, :, :3]
        h, w = rgb.shape[:2]

        # Ensure even width for UYVY packing
        if w % 2 != 0:
            rgb = rgb[:, :-1, :]
            w -= 1

        # Convert RGB (OpenCV BGR) to YUV
        yuv = cv2.cvtColor(rgb, cv2.COLOR_BGR2YUV)
        y = yuv[:, :, 0]
        u = yuv[:, :, 1]
        v = yuv[:, :, 2]

        # Prepare UYVY packed array (U Y0 V Y1 per 2 pixels)
        uyvy = np.zeros((h, w * 2), dtype=np.uint8)

        for row in range(h):
            for col in range(0, w, 2):
                idx = col * 2
                uyvy[row, idx + 0] = u[row, col]        # U
                uyvy[row, idx + 1] = y[row, col]        # Y0
                uyvy[row, idx + 2] = v[row, col]        # V
                uyvy[row, idx + 3] = y[row, col + 1]    # Y1

        # Save UYVY as .npy file
        np.save(os.path.join(output_folder, f"{base_name}.npy"), uyvy)

    print(f"All {len(os.listdir(input_folder))} images converted to UYVY format.")


def convert_images_to_yuv(input_folder, output_folder):
    """
    Converts all RGB images in the input folder to YUV format and saves them as .npy files.

    Args:
        input_folder (str): Path to the folder containing input RGBD images.
        output_folder (str): Path to save YUV .npy files.
    """
    os.makedirs(output_folder, exist_ok=True)

    for fname in sorted(os.listdir(input_folder)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(input_folder, fname)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None or img.shape[2] < 3:
            print(f"Skipping {fname}: not a valid RGB image.")
            continue

        base_name = os.path.splitext(fname)[0]
        rgb = img[:, :, :3]

        # Convert RGB (OpenCV BGR) to YUV
        yuv = cv2.cvtColor(rgb, cv2.COLOR_BGR2YUV)
        yuv = np.transpose(yuv, (2, 0, 1))
        yuv = yuv.astype(np.uint8)

        np.save(os.path.join(output_folder, f"{base_name}.npy"), yuv, allow_pickle=False)
        print(f"Saved {fname} as yuv")


def depth_checker():
    depth_path = config.config["depth_path"]
    random_index = random.randint(0, len(os.listdir(depth_path)))
    depth_array = np.load(os.path.join(depth_path, f"array_{random_index:05d}.npy"))
    print(f"Depth array shape: {depth_array.shape}")
    print(f"Depth array: {depth_array}")
    plt.imshow(depth_array, cmap="gray")
    plt.show()
    
    
def h5_checker(h5_path):
    with h5py.File(h5_path, "r") as f:
        print(f"Total {len(f.keys())} keys in the h5 file")
    

def convert_h5_to_array(h5_path): 
    '''
        Load depth matrix from h5 file and store as numpy 
    '''
    output_path = os.path.join("data", "depth_matrix")
    os.makedirs(output_path, exist_ok=True)
    
    with h5py.File(h5_path, "r") as f:
        for i, key in enumerate(f.keys()):
            # 520 * 240 array store the depth information
            array = f[f[key].name][:]
            array = array / 255.0
            # print(f"Array : {array}")
            # print(f"Array shape: {array.shape}")
            # break
            array_path = os.path.join(output_path, f"array_{i:05d}.npy")
            np.save(array_path, array)
            print(f"Save {f[key].name} array to {array_path}")


def convert_h5_to_image(h5_path): 
    '''
        Load image from h5 file and store in 
    '''
    output_path = os.path.join("data", "depth_map")
    os.makedirs(output_path, exist_ok=True)
    
    with h5py.File(h5_path, "r") as f:
        # print(f.keys())
        for i, key in enumerate(f.keys()):
            img_array = f[f[key].name][:]
            img = Image.fromarray(img_array)
            img_path = os.path.join(output_path, f"image_{i:05d}.jpg")
            img.save(img_path)
            print(f"Save {f[key].name} image to {img_path}")
    
    
def rename_files(path):
    '''
        Rename files in a folder
    '''
    files = sorted(os.listdir(path))
    output_path = os.path.join("data", "original_image")
    print(f"Found {len(files)} files")
    
    for i, file in enumerate(files):
        os.rename(os.path.join(path, file), os.path.join(output_path, f"image_{i:05d}.jpg"))
        print(f"Rename {file} to image_{i:05d}.jpg")
     

def load_comparison():
    '''
        Randomly load a pair of image and depth map for comparison
    '''
    img_data_folder = config.config["image_path"]
    depth_data_folder = config.config["depth_path"]
    
    idx = random.randint(0, len(os.listdir(img_data_folder)))
    img = Image.open(os.path.join(img_data_folder, f"image_{idx:05d}.jpg"))
    depth = Image.open(os.path.join(depth_data_folder, f"image_{idx:05d}.jpg"))
    
    # Rotate the image and depth map by 90 degrees
    img_rotated = img.rotate(90, expand=True)
    depth_rotated = depth.rotate(90, expand=True)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rotated)
    plt.title(f"original image {idx} (rotated)")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(depth_rotated, cmap="gray")
    plt.title(f"depth map {idx} (rotated)")
    plt.axis("off")
    plt.show()
    
    
def show_eval_images(depth_pred, img, depth_gt):
    '''
        Display the input image, ground truth and predicted depth maps
    '''
    if config.config["image_mode"] == "RGB": img = np.transpose(img, (1, 2, 0))
    img_rotated = np.rot90(img, k=1)
    depth_pred_rotated = np.rot90(depth_pred, k=1)
    depth_gt_rotated = np.rot90(depth_gt, k=1)

    plt.figure()
    plt.subplot(3, 1, 1)
    if config.config["image_mode"] == "RGB": plt.imshow(img_rotated, cmap="gray")
    else: plt.imshow(img_rotated)
    plt.title(f"Original image")
    plt.axis("off")
    
    plt.subplot(3, 1, 2)
    plt.imshow(depth_gt_rotated, cmap="gray")
    plt.title(f"Depth ground truth")
    plt.axis("off")

    plt.subplot(3, 1, 3)
    plt.imshow(depth_pred_rotated, cmap="gray")
    plt.title(f"Predicted depth map")
    plt.axis("off")
    plt.show()

def show_eval_vectors(depth_pred, depth_gt, img, depth_img):
    img_rotated = np.transpose(np.rot90(img[0], k=1, axes=[1,2]), [1,2,0]) / 255
    img_rotated = np.clip(cv2.cvtColor(img_rotated, cv2.COLOR_YUV2RGB), 0, 1) # Convert back to RGB for display

    depth_img_rotated = np.rot90(depth_img, k=1)

    cam_half_fov = config.config["cam_hor_FOV"] / 2
    min_angle = np.deg2rad(-cam_half_fov) * -1 + 0.5*np.pi
    max_angle = np.deg2rad(cam_half_fov) * -1 + 0.5*np.pi
    angles = np.linspace(min_angle, max_angle, len(depth_pred),  endpoint=False)

    plt.figure()
    plt.subplot(2, 2, 1)
    if config.config["image_mode"] == "L": plt.imshow(img_rotated, cmap="gray")
    else: plt.imshow(img_rotated)
    plt.title(f"Original image")
    plt.axis("off")
    
    plt.subplot(2, 2, 3)
    plt.imshow(depth_img_rotated, cmap="gray")
    plt.title(f"Depth image ground truth")
    plt.axis("off")

    plt.subplot(1, 2, 2, projection="polar")
    plt.plot(angles, depth_pred)
    plt.plot(angles, depth_gt)
    plt.title(f"Depth vector visualization")
    plt.xticks([max_angle, np.pi/2, min_angle], [f"{cam_half_fov}°", f"0°", f"{-cam_half_fov}°"])
    plt.legend(["Predicted depth", "Ground truth depth"], loc="lower center")

    plt.show()