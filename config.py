# Configurations & Hyperparameters for model training
import torch

config = {
    # Model training configurations
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 100,
    "batch_size": 4, # For test, use a smaller batch size
    "train_val_split": 0.8,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "num_workers": 1, # Number of CPU workers for data loading, small for test
    
    # Data 
    'image_path': 'data/original_image',
    'depth_path': 'data/depth_map',
    'image_width': 520,
    'image_height': 240,
    "cam_hor_FOV": 120, # degrees
    
    # Model configurations
    "image_mode": "L", # "RGB" or "L" (grayscale)
    "input_type_uint8": False,
    "input_channels": 1, # 3 for RGB, 1 for grayscale
    "output_channels": 65, # Output depth vector length

    # Model paths
    "logging_on": True,
    "save_model_path": "models",
    "save_log_path": "logs/logger",
    "save_event_path": "logs/events",
}