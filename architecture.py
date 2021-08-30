import numpy as np
import torch


class ImageExtrapolationCNN(torch.nn.Module):
    def __init__(self):
        super(ImageExtrapolationCNN, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=2, out_channels=64, kernel_size=7, bias=True, padding=int(7 / 2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, bias=True, padding=int(7 / 2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, bias=True, padding=int(7 / 2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, bias=True, padding=int(7 / 2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, bias=True, padding=int(7 / 2)),
            torch.nn.ReLU()
        )
        self.out = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, bias=True, padding=int(7 / 2))
        pass

    def forward(self, x):
        net_out = self.net(x)
        output = self.out(net_out)

        output_flat = torch.zeros(len(x), 2475)
        for i in range(0, len(x)):
            mask = x[i][1].eq(0).flatten()
            masked = torch.masked_select(output[i][0].flatten(), mask)
            output_flat[i][:len(masked)] = masked[:]
        return output, output_flat
