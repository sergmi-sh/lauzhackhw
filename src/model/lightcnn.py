import torch


class MFMActivation(torch.nn.Module):
    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def forward(self, x: torch.Tensor):
        sz_along = x.shape[self.axis] // 2
        x1, x2 = x.split(sz_along, dim=self.axis)
        return torch.max(x1, x2)


class LightCNN(torch.nn.Module):
    def __init__(self, input_channels: int, input_height: int, input_width: int):
        super().__init__()
        fc_height = (input_height - 4) // 16
        fc_width = (input_width - 4) // 16
        self.net = torch.nn.Sequential(
            #
            torch.nn.Conv2d(
                input_channels, 64, kernel_size=(5, 5), stride=(1, 1)
            ),  # Conv_1
            MFMActivation(axis=1),  # MFM_2
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # MaxPool_3
            #
            torch.nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1)),  # Conv_4
            MFMActivation(axis=1),  # MFM_5
            torch.nn.BatchNorm2d(32),  # BatchNorm_6
            #
            torch.nn.Conv2d(
                32, 96, kernel_size=(3, 3), stride=(1, 1), padding=1
            ),  # Conv_7
            MFMActivation(axis=1),  # MFM_8
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # MaxPool_9
            torch.nn.BatchNorm2d(48),  # BatchNorm_10
            #
            torch.nn.Conv2d(48, 96, kernel_size=(1, 1), stride=(1, 1)),  # Conv_11
            MFMActivation(axis=1),  # MFM_12
            torch.nn.BatchNorm2d(48),  # BatchNorm_13
            #
            torch.nn.Conv2d(
                48, 128, kernel_size=(3, 3), stride=(1, 1), padding=1
            ),  # Conv_14
            MFMActivation(axis=1),  # MFM_15
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # MaxPool_16
            #
            torch.nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1)),  # Conv_17
            MFMActivation(axis=1),  # MFM_18
            torch.nn.BatchNorm2d(64),  # BatchNorm_19
            #
            torch.nn.Conv2d(
                64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1
            ),  # Conv_20
            MFMActivation(axis=1),  # MFM_21
            torch.nn.BatchNorm2d(32),  # BatchNorm_22
            #
            torch.nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1)),  # Conv_23
            MFMActivation(axis=1),  # MFM_24
            torch.nn.BatchNorm2d(32),  # BatchNorm_25
            #
            torch.nn.Conv2d(
                32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1
            ),  # Conv_26
            MFMActivation(axis=1),  # MFM_27
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # MaxPool_28
            #
            torch.nn.Flatten(),
            torch.nn.Linear(fc_height * fc_width * 32, 160),  # FC_29
            MFMActivation(axis=1),  # MFM_30
            torch.nn.BatchNorm1d(80),  # BatchNorm_31
            torch.nn.Linear(80, 2),  # FC_32
            # torch.nn.Softmax(dim=1),
        )

    def forward(self, data_object: torch.Tensor, **batch):
        return {"logits": self.net(data_object)}

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )
        result_info = "LightCNN:"
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"
        return result_info
