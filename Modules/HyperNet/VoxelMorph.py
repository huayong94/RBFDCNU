from enum import EnumMeta
from tokenize import Decnumber

import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules.Interpolation import SpatialTransformer
from Modules.Loss import LOSSDICT

from .conv import Conv2d, Conv2dRelu


class UnetCore2d(nn.Module):
    def __init__(self, enc_nf, dec_nf) -> None:
        super(UnetCore2d, self).__init__()

        self.shape = {}
        self.encoder = nn.ModuleList()
        self.shape["encoder"] = {}
        for i in range(len(enc_nf)):
            prev_nf = 2 if i == 0 else enc_nf[i - 1]
            self.encoder.append(Conv2dRelu(prev_nf, enc_nf[i], 2))
            self.shape["encoder"]["conv%d" % i] = self.encoder[-1].shape

        self.decoder = nn.ModuleList()
        self.shape["decoder"] = {}
        self.decoder.append(Conv2dRelu(enc_nf[-1], dec_nf[0]))
        self.decoder.append(Conv2dRelu(dec_nf[0] * 2, dec_nf[1]))
        self.decoder.append(Conv2dRelu(dec_nf[1] * 2, dec_nf[2]))
        self.decoder.append(Conv2dRelu(dec_nf[2] + enc_nf[0], dec_nf[3]))
        self.decoder.append(Conv2dRelu(dec_nf[3], dec_nf[4]))
        self.decoder.append(Conv2dRelu(dec_nf[4] + 2, dec_nf[5], 1))
        for i in len(self.decoder):
            self.shape["decoder"]["conv%d" % i] = self.decoder[-1].shape

        self.vm2_conv = Conv2dRelu(dec_nf[5], dec_nf[6])
        self.shape["vm2_conv"] = self.vm2_conv.shape

    def forward(self, x, params):
        x_enc = [x]
        for i, l in enumerate(self.encoder):
            x_enc.append(l(x_enc[-1], params["encoder"]["conv%d" % i]))

        y = x_enc[-1]
        for i in range(3):
            y = self.decoder[i](y, params["decoder"]["conv%d"] % i)
            y = F.upsample(y, scale_factor=2, mode="nearest")
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)

        y = self.decoder[3](y, params["decoder"]["conv3"])
        y = self.decoder[4](y, params["decoder"]["conv4"])

        y = F.upsample(y, scale_factor=2, mode="nearest")
        y = torch.cat([y, x_enc[0]], dim=1)
        y = self.decoder[5](y, params["decoder"]["conv5"])

        y = self.vm2_conv(y, params["vm2_conv"])

        return y


class VoxelMorph(nn.Module):
    def __init__(self, vol_size, enc_nf, dec_nf) -> None:
        super(VoxelMorph, self).__init__()

        self.shape = {}
        self.unet = UnetCore2d(enc_nf, dec_nf)
        self.shape["unet"] = self.unet.shape

        self.flow = Conv2d(dec_nf[-1], 2, 3, 1, 1)
        self.shape["flow"] = self.flow.shape

        self.transformer = SpatialTransformer(vol_size)

    def forward(self, src, tgt, params):
        x = torch.cat([src, tgt], dim=1)
        x = self.unet(x, params["unet"])
        flow = self.flow(x, params["flow"])

        y = self.transformer(src, flow)

        return flow, y

