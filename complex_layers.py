import torch
import torch.nn as nn

class Complex_Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        if not self.training or self.p == 0.0:
            return input
        else:
            mask = torch.rand_like(input.real) > self.p
            return input * mask / (1 - self.p)