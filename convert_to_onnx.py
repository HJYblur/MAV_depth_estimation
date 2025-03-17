import torch
import config
import model
from torch.quantization import quantize_dynamic
import onnx

def export_to_onnx(model_path, model_id, use_quantization=False):
    config.config["device"] = "cpu" # Always set device to cpu for export, so you don't get errors
    img_width = config.config["image_width"]
    img_height = config.config["image_height"]

    depth_model = model.ShallowDepthModel()
    depth_model.load_state_dict(torch.load(model_path))
    depth_model.eval()

    # Optimize for embedded CPU's by reducing precision (32 bit float to 8 bit int) to reduce 
    # model size and improve inference speed
    if use_quantization:
        depth_model = quantize_dynamic(depth_model, dtype=torch.qint8)

    # Export to ONNX
    if config.config["input_type_uint8"]:
        dummy_input = torch.randint(0, 256, (1, config.config["input_channels"], img_width, img_height * 2), dtype=torch.uint8)
    else:
        dummy_input = torch.randn(1, config.config["input_channels"], img_width, img_height * 2)

    onnx_model_path = config.config["save_model_path"] + f"/onnx_models/model_{model_id}.onnx"
    onnx_program = torch.onnx.export(depth_model, dummy_input, dynamo=True)
    onnx_program.optimize()
    onnx_program.save(onnx_model_path)

    print(f"Exported model_{model_id} to ONNX in {onnx_model_path}")

    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

if __name__ == "__main__":
    model_id = 99
    model_path = config.config["save_model_path"] + f"/model_{model_id}.pth"

    export_to_onnx(model_path, model_id, False)