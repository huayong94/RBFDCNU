import torch
import torch.nn as nn
from Modules.HyperNet import VoxelMorph


class HyperNet(nn.Module):
    def __init__(
        self,
        vol_size,
        enc_nf,
        dec_nf,
        similaity_loss="LCC",
        hyperparams={
            "reg_param": {"min": 0, "max": 10, "step": 100},
            "similarity_loss_param": {},
        },
    ) -> None:
        super(HyperNet, self).__init__()

        self.vm = VoxelMorph(vol_size, enc_nf, dec_nf)
        self.shape = self.vm.shape

        self.commomLayer = nn.Linear(1, 10, bias=True)

        

