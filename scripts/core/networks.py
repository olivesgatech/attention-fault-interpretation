import torch
import torch.nn as nn


# pytorch class defining a 3D Unet with skip connections
class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()

        self.conv1 = self.conv_block(1, 16)
        self.conv2 = self.conv_block(16, 32)
        self.conv3 = self.conv_block(32, 64)
        self.conv4 = self.conv_block(64, 128)

        self.upconv5 = self.upconv_block(128, 64)
        self.conv5 = self.conv_block(128, 64)

        self.upconv6 = self.upconv_block(64, 32)
        self.conv6 = self.conv_block(64, 32)

        self.upconv7 = self.upconv_block(32, 16)
        self.conv7 = self.conv_block(32, 16)

        self.conv8 = nn.Conv3d(16, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = nn.functional.max_pool3d(conv1, kernel_size=2, stride=2)

        conv2 = self.conv2(pool1)
        pool2 = nn.functional.max_pool3d(conv2, kernel_size=2, stride=2)

        conv3 = self.conv3(pool2)
        pool3 = nn.functional.max_pool3d(conv3, kernel_size=2, stride=2)

        conv4 = self.conv4(pool3)

        up5 = torch.cat([self.upconv5(conv4), conv3], dim=1)
        conv5 = self.conv5(up5)

        up6 = torch.cat([self.upconv6(conv5), conv2], dim=1)
        conv6 = self.conv6(up6)

        up7 = torch.cat([self.upconv7(conv6), conv1], dim=1)
        conv7 = self.conv7(up7)

        output = self.conv8(conv7)

        return output