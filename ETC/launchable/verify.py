"""Run with the course Python. No training, network access or data generation."""
from importlib.metadata import version
import sys

import ipywidgets
import torch
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.sym.eq.pde import PDE


assert sys.version_info[:3] == (3, 12, 11), "Expected Python 3.12.11"
assert version("nvidia-physicsnemo") == "2.2.2"
assert torch.__version__ == "2.10.0+cu128"
assert version("torchvision") == "0.25.0+cu128"
assert torch.cuda.is_available(), "CUDA unavailable; inspect the VM GPU and driver"
assert ipywidgets.__version__ == "8.1.9"
print("Course imports and CUDA availability verified:", torch.cuda.get_device_name(0))
