import torch
from typing import Any


class ChainModel(torch.nn.Module):
    def __init__(self, *models: torch.nn.Module):
        super().__init__()
        self.models: torch.nn.ModuleList = torch.nn.ModuleList(models)

    def forward(self, data_object: torch.Tensor, **batch: Any) -> Any:
        for model in self.models:
            data_object = model(data_object, **batch)
        return data_object
