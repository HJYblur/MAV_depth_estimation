import torch
import model
import config
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_input_tensor(filename):
    with open(filename, "r") as file:
        values = list(map(float, file.read().split()))  # Read all numbers in one go

    tensor = torch.tensor(values).reshape(1, 3, 520, 240)  # Reshape correctly
    return tensor

# Load the saved file
tensor = load_input_tensor("input_array.txt")
print(tensor.shape)  # Should print: torch.Size([1, 3, 520, 240])

config.config["device"] = "cpu"

# Load model
model_path = config.config["save_model_path"] + f"/model_99.pth"

depth_model = model.ShallowDepthModel()
# depth_model = model.Mob3DepthModel()
depth_model.load_state_dict(torch.load(model_path))
depth_model.eval()

depth_vector = depth_model(tensor)
print(depth_vector)

# [0.12 0.16 0.19 0.21 0.27 0.11 0.33 0.29 0.22 0.28 0.25 0.27 0.39 0.46 0.58 0.57 ]
# img_rotated = np.transpose(np.rot90(tensor[0], k=1, axes=[1,2]), [1,2,0]) / 255
img_rotated = np.transpose(tensor[0].numpy(), [1,2,0]) / 255
img_rotated = cv2.cvtColor(img_rotated, cv2.COLOR_YUV2RGB) # Convert back to RGB for display

plt.figure()
plt.imshow(img_rotated)
plt.axis("off")
plt.show()