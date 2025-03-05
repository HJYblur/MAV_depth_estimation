import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import config, utils, model, dataset
import time

def train():
    # Set up logging & device
    logger = utils.init_logger()
    device = config.config["device"]
    
    # Load dataset
    train_loader, val_loader = dataset.load_train_val_dataset()
    
    # Load model
    depth_model = model.DepthModel()
    depth_model.to(device)
    if args.checkpoint:
        depth_model.load_state_dict(torch.load(args.checkpoint))
    
    # Loss function & optimizer
    weight = config.config["ssim_weight"]
    # loss = lambda x, y: weight * (1 - nn.SSIM()(x, y)) + (1 - weight) * nn.MSELoss()(x, y)
    loss = nn.MSELoss()
    
    optimizer = torch.optim.Adam(
        depth_model.parameters(), 
        lr=config.config["learning_rate"], 
        weight_decay=config.config["weight_decay"]
        )
    
    writer = SummaryWriter(config.config["save_event_path"])
    best_loss = float("inf")
    counter = 0
    for epoch in range(config.config["epochs"]):
        # Training loop
        for i, (img, depth) in tqdm(enumerate(train_loader)):
            img, depth = img.to(device), depth.to(device)
            rgbd = torch.cat([img, depth], dim=1)
            
            optimizer.zero_grad()
            
            # pred_depth = depth_model(rgbd)
            pred_depth = depth_model(img) # I don't understand why the depth is concatenated to the input, made it rgb for now
            loss_val = loss(pred_depth, depth)
            loss_val.backward()
            optimizer.step()
            
            if i % 10 == 0:
                logger.info(f"Epoch {epoch}, Iteration {i}, Loss: {loss_val.item()}")
                writer.add_scalar("Loss/train", loss_val.item(), epoch * len(train_loader) + i)
        
        torch.save(depth_model.state_dict(), os.path.join(config.config["save_model_path"], f"model_{epoch}.pth"))
        
        # Validation loop
        depth_model.eval()
        for i, (img, depth) in enumerate(val_loader):
            img, depth = img.to(device), depth.to(device)
            rgbd = torch.cat([img, depth], dim=1)
            
            pred_depth = depth_model(rgbd)
            loss_val = loss(pred_depth, depth)
            
            if i % 10 == 0:
                logger.info(f"Epoch {epoch}, Iteration {i}, Val Loss: {loss_val.item()}")
                writer.add_scalar("Loss/val", loss_val.item(), epoch * len(val_loader) + i)
            
            # Early stopping
            if loss_val < best_loss:
                best_loss = loss_val
                counter = 0
            else:
                counter += 1
                if counter > 5:
                    logger.info(f"Early stopping at epoch {epoch}.")
                    break
    logger.info("Training complete.")
    writer.close()

def eval(num_imgs, model_id=0):
    '''
    Pick some random input images and run depth estimation on it
    '''
    # Load model
    model_path = config.config["save_model_path"] + f"/model_{model_id}.pth"

    depth_model = model.DepthModel()
    depth_model.load_state_dict(torch.load(model_path))
    depth_model.eval()

    # Load images
    eval_loader = dataset.load_eval_dataset(num_imgs)

    # Run depth estimation
    with torch.no_grad():
        for i, (img, depth_gt) in enumerate(eval_loader):
            # Run model
            start_time = time.time()
            depth_pred = depth_model(img)[0]
            
            print(f"Inference time: {time.time() - start_time:.2f} seconds")

            # Go from tensors to numpy arrays
            depth_pred = depth_pred.squeeze().cpu().numpy()
            img = img.squeeze().cpu().numpy()
            depth_gt = depth_gt.squeeze().cpu().numpy()

            # Display images
            utils.show_eval_images(depth_pred, img, depth_gt, i)
    
if __name__ == "__main__":
    args = utils.parse_args()
    if args.mode == "train":
        train()
    elif args.mode == "eval":
        eval(num_imgs=1, model_id=0)
    else:
        utils.load_comparison()
        # raise ValueError("Invalid mode. Please choose 'train' or 'eval'.")
