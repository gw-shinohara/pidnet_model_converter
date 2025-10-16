#!/usr/bin/env python3
import torch
import argparse
import os
import importlib.util
from collections import OrderedDict
import sys

def load_module_from_file(module_name, file_path):
    """Dynamically loads a Python module from a given file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise FileNotFoundError(f"Could not find module file at {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def convert_generic_model_to_torchscript(
    checkpoint_path,
    model_file_path,
    model_class_name,
    output_path
):
    """
    Loads a generic PyTorch model from a file and checkpoint, and converts it to TorchScript
    using torch.jit.script(). This method can handle control flow (if/for).

    Args:
        checkpoint_path (str): Path to the PyTorch checkpoint file (.pth or .pt).
        model_file_path (str): Path to the Python file containing the model definition.
        model_class_name (str): The name of the model class to instantiate (e.g., 'ResNet50').
        output_path (str): Path to save the output TorchScript file (.ts).
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: The input checkpoint file '{checkpoint_path}' does not exist.")
        return
    if not os.path.exists(model_file_path):
        print(f"Error: The model definition file '{model_file_path}' does not exist.")
        return

    try:
        # 1. Dynamically load the model definition module
        print(f"INFO: Loading model definition from '{model_file_path}'...")
        model_module = load_module_from_file("custom_model_module", model_file_path)

        # 2. Get the model class and instantiate it
        model_class = getattr(model_module, model_class_name)

        try:
            # Attempt instantiation with no arguments
            model = model_class()
        except TypeError as e:
            print(f"Warning: Model instantiation failed without arguments. You may need to edit this script to pass required constructor arguments.")
            print(f"Original error: {e}")
            return

        # 3. Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        # Extract state_dict
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Remove 'module.' prefix if necessary
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        # 4. Load weights into the model
        print(f"INFO: Loading weights into model '{model_class_name}'...")
        model.load_state_dict(new_state_dict, strict=True)
        model.eval()

        # 5. Script the model using torch.jit.script()
        # This method analyzes the model code and is not dependent on a specific input size.
        print(f"INFO: Converting model to TorchScript using torch.jit.script()...")
        scripted_model = torch.jit.script(model)

        # 6. Save the TorchScript model
        scripted_model.save(output_path)

        print(f"\nSUCCESS: Converted '{checkpoint_path}' to TorchScript and saved as '{output_path}'.")
        print(f"Model Class: {model_class_name}")
        print(f"Conversion Method: torch.jit.script()")

    except Exception as e:
        print(f"\nAn error occurred during generic conversion:")
        print(e)
        print("\nNote: For complex models, the model instantiation step might require manual argument passing.")
        print("Also ensure the model's code is compatible with TorchScript syntax.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a generic PyTorch checkpoint to a TorchScript file using torch.jit.script().')
    parser.add_argument('--input', type=str, required=True, help='Path to the input .pth or .pt file (model checkpoint).')
    parser.add_argument('--model-file', type=str, required=True, help='Path to the Python file containing the model definition (e.g., my_model.py).')
    parser.add_argument('--model-class', type=str, required=True, help='The exact name of the model class in the model file (e.g., MyCustomNet).')
    parser.add_argument('--output', type=str, required=True, help='Path to the output TorchScript .ts file.')

    args = parser.parse_args()

    convert_generic_model_to_torchscript(
        args.input,
        args.model_file,
        args.model_class,
        args.output
    )
