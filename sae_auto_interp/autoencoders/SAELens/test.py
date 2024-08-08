from safetensors.torch import save_file
import torch
import os

# Create a tensor
x = torch.randn(5)

# Use a relative path or a path you're sure you have write access to
file_path = "blah.safetensors"

try:
    save_file({"test_key": x}, file_path)
    print(f"File saved successfully at: {os.path.abspath(file_path)}")
except Exception as e:
    print(f"An error occurred while saving the file: {e}")

# Print current working directory
print(f"Current working directory: {os.getcwd()}")