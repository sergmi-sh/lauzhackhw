import torch


class PadTrimTransform1D(torch.nn.Module):
    def __init__(self, target_size: int):
        super().__init__()
        self.target_size = target_size

    def forward(self, x: torch.Tensor):
        if x.shape[1] < self.target_size:
            x = torch.nn.functional.pad(x, (0, self.target_size - x.shape[1]))
        else:
            x = x[:, : self.target_size]
        return x
