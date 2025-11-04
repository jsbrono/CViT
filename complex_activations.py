import torch.nn as nn
import torch

class CReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, input):
        out_real = self.relu(input.real)
        out_imag = self.relu(input.imag)
        return torch.complex(out_real, out_imag) 

class CGELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU()

    def forward(self, input):
        out_real = self.gelu(input.real)
        out_imag = self.gelu(input.imag)
        return torch.complex(out_real, out_imag) 
