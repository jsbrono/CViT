import torch.nn as nn
import torch

class Complex_LayerNorm_split(nn.Module):

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.layernorm2 = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, input):
        return torch.complex(self.layernorm1(input.real), self.layernorm2(input.imag))

# code based on tensorflow batch normalisation implementation: https://github.com/ChihebTrabelsi/deep_complex_networks/blob/master/complexnn/bn.py
# original authors:  Chiheb Trabelsi, Olexa Bilaniuk
class ComplexLayerNormWhitening(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.beta = nn.Parameter(torch.zeros(normalized_shape, dtype=torch.complex64))
            self.gamma_rr = nn.Parameter(torch.ones(normalized_shape) / torch.sqrt(torch.tensor(2.0)))
            self.gamma_ri = nn.Parameter(torch.zeros(normalized_shape))
            self.gamma_ii = nn.Parameter(torch.ones(normalized_shape) / torch.sqrt(torch.tensor(2.0)))

    def forward(self, x):

        mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - mean

        Vrr = (x_centered.real * x_centered.real).mean(dim=-1, keepdim=True)+self.eps
        Vri = (x_centered.real * x_centered.imag).mean(dim=-1, keepdim=True)+self.eps
        Vii = (x_centered.imag * x_centered.imag).mean(dim=-1, keepdim=True)+self.eps

        trace = Vrr + Vii
        det = Vrr * Vii - Vri * Vri
        
        sqrt_det = torch.sqrt(det)
        trace_thing = torch.sqrt(trace + 2 * sqrt_det)

        inverse_st = 1.0 / (sqrt_det * trace_thing)

        Wrr = inverse_st * (Vii+sqrt_det)
        Wri = inverse_st * -Vri
        Wii = inverse_st * (Vrr+sqrt_det)

        x_white_real = Wrr * x_centered.real + Wri * x_centered.imag
        x_white_imag = Wri * x_centered.real + Wii * x_centered.imag
        x_white = torch.complex(x_white_real, x_white_imag)

        if self.elementwise_affine:
            gamma_matrix_real = self.gamma_rr * x_white.real - self.gamma_ri * x_white.imag
            gamma_matrix_imag = self.gamma_ri * x_white.real + self.gamma_ii * x_white.imag
            
            x_white_real = gamma_matrix_real + self.beta.real
            x_white_imag = gamma_matrix_imag + self.beta.imag

            x_white = torch.complex(x_white_real, x_white_imag)

        return x_white