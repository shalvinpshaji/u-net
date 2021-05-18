import torch
import torch.nn as nn


class DoubleConv2d(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):

    def __init__(self, in_channel, out_channel, conv_out_channel):
        super(UpConv, self).__init__()
        self.trans_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(2, 2), stride=(2, 2))
        )
        self.double_conv = DoubleConv2d(in_channels=out_channel*2, out_channels=conv_out_channel)

    def forward(self, x, skip_input):
        x = self.trans_conv(x)
        skip_size = skip_input.size()[-1]
        x_size = x.size()[-1]
        size_diff = (skip_size - x_size) // 2
        crop_input = skip_input[:, :, size_diff:skip_size-size_diff, size_diff:skip_size-size_diff]
        concat = torch.cat((x, crop_input), 1)
        return self.double_conv(concat)


class UNet(nn.Module):

    def __init__(self, in_filter=1, down_filters=[], up_filters=[]):
        super(UNet, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.outputs = []
        for out_filter in down_filters:
            self.down.append(DoubleConv2d(in_filter, out_filter))
            self.down.append(self.max_pool)
            in_filter = out_filter
        self.bend = DoubleConv2d(in_channels=512, out_channels=1024)
        in_filter = 1024
        for up_filter in up_filters:
            self.up.append(UpConv(in_channel=in_filter, out_channel=up_filter, conv_out_channel=up_filter))
            in_filter = up_filter

    def forward(self, x):
        for down in self.down:
            x = down(x)
            self.outputs.append(x)
        self.outputs = self.outputs[::2]
        x = self.bend(x)
        for up, inp in zip(self.up, reversed(self.outputs)):
            x = up(x, inp)
        return x


if __name__ == "__main__":
    t = torch.randn((1, 1, 572, 572))
    d = UNet(in_filter=1, down_filters=[64, 128, 256, 512], up_filters=[512, 256, 128, 64])
    print(d(t).size())
