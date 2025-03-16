import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import config, utils, model, dataset
import time
from torch.quantization import quantize_dynamic
import numpy as np

def train():
    # Set up logging & device
    logging_on = config.config["logging_on"]
    if logging_on:
        os.makedirs(config.config["save_log_path"], exist_ok=True)
        logger = utils.init_logger()
        os.makedirs(config.config["save_event_path"], exist_ok=True)
        writer = SummaryWriter(config.config["save_event_path"])
    device = config.config["device"]
    
    # Load dataset
    train_loader, val_loader = dataset.load_train_val_dataset()
    
    # Load model
    depth_model = model.ShallowDepthModel()
    # depth_model = model.Mob3DepthModel()
    depth_model.to(device)
    if args.checkpoint:
        depth_model.load_state_dict(torch.load(args.checkpoint))
    
    # Loss function & optimizer
    loss = nn.MSELoss()
    
    optimizer = torch.optim.Adam(
        depth_model.parameters(), 
        lr=config.config["learning_rate"], 
        weight_decay=config.config["weight_decay"]
        )
    
    best_loss = float("inf")
    counter = 0
    for epoch in range(config.config["epochs"]):
        # Training loop
        for i, (img, depth_vec) in tqdm(enumerate(train_loader)):
            img, depth_vec = img.to(device), depth_vec.to(device)
            # print(f"img: {img.shape}, depth_vec: {depth_vec.shape}")
            
            optimizer.zero_grad()
            pred_depth = depth_model(img)
            loss_train = loss(pred_depth, depth_vec)
            # print(f"pred_depth: {pred_depth.shape}, depth_vector: {depth_vector.shape}")
            loss_train.backward()
            optimizer.step()
            
            if i % 10 == 0 and logging_on:
                logger.info(f"Epoch {epoch}, Iteration {i}, Loss: {loss_train.item()}")
                writer.add_scalar("Loss/train", loss_train.item(), epoch * len(train_loader) + i)
        
        torch.save(depth_model.state_dict(), os.path.join(config.config["save_model_path"], f"model_{epoch}.pth"))
        
        # Validation loop
        depth_model.eval()
        for i, (img, depth_vec) in enumerate(val_loader):
            img, depth_vec = img.to(device), depth_vec.to(device)
            
            pred_depth = depth_model(img)
            loss_val = loss(pred_depth, depth_vec)
            
            if i % 10 == 0 & logging_on:
                logger.info(f"Epoch {epoch}, Iteration {i}, Val Loss: {loss_val.item()}")
                writer.add_scalar("Loss/val", loss_val.item(), epoch * len(val_loader) + i)
            
            # Early stopping
            # if loss_val < best_loss:
            #     best_loss = loss_val
            #     counter = 0
            # else:
            #     counter += 1
            #     if counter > config.config["patience"]:
            #         if logging_on: logger.info(f"Early stopping at epoch {epoch}.")
            #         break
    if logging_on:
        logger.info("Training complete.")
        writer.close()

    print(f"Number of parameters: {depth_model.compute_parameters()}")


def eval(num_imgs, model_id=0):
    '''
    Pick some random input images and run depth estimation on it
    '''
    config.config["device"] = "cpu"
    depth_path = config.config["depth_path"]

    # Load model
    model_path = config.config["save_model_path"] + f"/model_{model_id}.pth"

    depth_model = model.ShallowDepthModel()
    # depth_model = model.Mob3DepthModel()
    depth_model.load_state_dict(torch.load(model_path, map_location=config.config["device"]))
    depth_model.eval()
    # depth_model = quantize_dynamic(depth_model, dtype=torch.qint8)
 
    print(f"Number of parameters: {depth_model.compute_parameters()}")

    # Load images
    eval_loader, random_indices = dataset.load_eval_dataset(num_imgs)

    # Run depth estimation
    with torch.no_grad():
        for i, (img, depth_gt) in enumerate(eval_loader):
            # Run model
            start_time = time.time()
            depth_pred = depth_model(img)
            print(f"Inference time: {time.time() - start_time:.2f} seconds")

            # print(f"Depth prediction: {depth_pred}")
            # max_depth, max_indices = torch.max(depth_pred, dim=1)
            # print(f"Predicted depth & position: {max_depth}, {max_indices / config.config['output_channels']}")

            depth_img = np.load(os.path.join(depth_path, f"array_{random_indices[i]:05d}.npy"))
            
            utils.show_eval_vectors(depth_pred[0], depth_gt[0], img, depth_img)

    
if __name__ == "__main__":
    args = utils.parse_args()
    if args.mode == "data":
        h5_path = os.path.join(config.config["h5_path"], args.h5file) # flight_5_depthmap.h5
        original_image_path = config.config["image_path"]
        utils.data_preprocess(h5_path, config.config["raw_path"], append=args.add_data)

    elif args.mode == "save_uyvy":
        uyuv_path = config.config["uyvy_path"]
        original_image_path = config.config["image_path"]
        utils.convert_images_to_uyvy(original_image_path, uyuv_path)

    elif args.mode == "save_yuv":
        yuv_path = config.config["yuv_path"]
        original_image_path = config.config["image_path"]
        utils.convert_images_to_yuv(original_image_path, yuv_path)

    elif args.mode == "train":
        train()

    elif args.mode == "eval":
        eval(num_imgs=3, model_id=args.model_id)

    else:
        h5_path = os.path.join(config.config["h5_path"], args.h5file) # flight_5_depthmap.h5
        utils.h5_checker(h5_path)
        # utils.depth_checker()
        # utils.load_comparison()
        # raise ValueError("Invalid mode. Please choose 'train' or 'eval'.")
