import numpy as np
import torch.nn as nn
import torch
from attention import RealMultiHeadAttention
from complexlayernorm import Complex_LayerNorm_split,ComplexLayerNormWhitening
from complex_activations import CReLU,CGELU
from complex_layers import Complex_Dropout

class ComplexMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        bias=False,
        score_mode="dot",        # "dot" | "linear" | "complex_learned"
        softmax_mode="real"      # "real" | "abs"
    ):
        super(ComplexMultiHeadAttention, self).__init__()

        # embed_dim = d_model = dk = dv
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.queries = nn.Linear(embed_dim, embed_dim, bias=bias, dtype=torch.complex64)
        self.keys    = nn.Linear(embed_dim, embed_dim, bias=bias, dtype=torch.complex64)
        self.values  = nn.Linear(embed_dim, embed_dim, bias=bias, dtype=torch.complex64)

        if score_mode == "linear":
            self.score_layer = nn.Linear(2 * self.head_dim, self.head_dim, bias=True)
        elif score_mode == "complex_learned":
            self.scoring_layer   = nn.Linear(2 * self.head_dim, self.head_dim, dtype=torch.complex64)
            self.complex_to_real = nn.Linear(2 * self.head_dim, self.head_dim, dtype=torch.float32)

        self.final_linear = nn.Linear(embed_dim, embed_dim, bias=bias, dtype=torch.complex64)

        self.score_mode = score_mode
        self.softmax_mode = softmax_mode

        self.d_model_sqrt_inv = 1 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        self.dropout = Complex_Dropout(0.1)

    # shared reshape for Q,K,V
    def _split_heads(self, x):
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # reshaping input embedding to be shape (BATCH,SEQ_LEN,NUM_HEADS,EMBEDDING/NUM_HEADS)

        Q = self.queries(q)
        K = self.keys(k)
        V = self.values(v)

        batch, seq_len, out_dim = Q.shape

        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        if self.score_mode == "dot":
            Kt = K.transpose(-2, -1).conj()
            att_scores = torch.matmul(Q, Kt) * self.d_model_sqrt_inv

        elif self.score_mode == "linear":
            Kt = K.transpose(-2, -1).conj()
            scores = torch.matmul(Q, Kt) * self.d_model_sqrt_inv
            scores = torch.cat([scores.real, scores.imag], dim=-1)
            att_scores = self.score_layer(scores)

        elif self.score_mode == "complex_learned":
            scored = self.scoring_layer(torch.cat([Q, K], dim=-1)) * self.d_model_sqrt_inv
            cat = torch.cat([scored.real, scored.imag], dim=-1)
            att_scores = self.complex_to_real(cat)

        else:
            raise ValueError(f"Unknown score_mode: {self.score_mode}")

        if self.softmax_mode == "real":
            weights_input = att_scores.real if torch.is_complex(att_scores) else att_scores
        else:  
            weights_input = torch.abs(att_scores)

        attention_weights = torch.softmax(weights_input, dim=-1).type(torch.complex64)
        attention_output  = torch.matmul(attention_weights, V)

        # attention is now shape (batch,head,seq,head_dim)
        # need to get to shape (batch,seq,embed_dim)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch, seq_len, out_dim)

        # final head linear map
        mha = self.final_linear(attention_output)
        mha = self.dropout(mha)

        # preserve old output format
        return mha, att_scores

class ComplexEncoderBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, hidden_dim=512, bias=False):
        super(ComplexEncoderBlock, self).__init__()
        if hidden_dim is None:
            hidden_dim=4*embed_dim
        self.mha = ComplexMultiHeadAttention(embed_dim, num_heads, bias=bias)
        self.layer_norm1 = ComplexLayerNormWhitening(embed_dim)
        self.layer_norm2 = ComplexLayerNormWhitening(embed_dim)
        self.feedforward = nn.Sequential(nn.Linear(embed_dim, hidden_dim,dtype=torch.complex64),CReLU(),nn.Linear(hidden_dim, embed_dim,dtype=torch.complex64),Complex_Dropout(0.1))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        #self attention only
        normed_input = self.layer_norm1(q)
        mha,att_map = self.mha(normed_input, normed_input, normed_input)
        residual_mha = q + mha

        normed_mha = self.layer_norm2(residual_mha)
        ff_out = self.feedforward(normed_mha)
        encoder_out = residual_mha + ff_out

        return encoder_out,att_map

class ComplexViT(nn.Module):
    def __init__(self, image_size, embed_dim=512, num_heads=8, num_blocks=6, num_classes=10, patch_size=16, bias=False,encoder_ff_hidden_dim=1024):
        super(ComplexViT, self).__init__()

        self.patch_embedding = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size,dtype=torch.complex64)
        num_patches = (image_size // patch_size) ** 2
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positional_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        self.encoder_blocks = nn.ModuleList([ComplexEncoderBlock(embed_dim, num_heads, bias=bias,hidden_dim=encoder_ff_hidden_dim) for _ in range(num_blocks)])
        self.final_layer = nn.Linear(embed_dim*2,num_classes)

    def forward(self, input_images):
        patches = self.patch_embedding(input_images)
        patch_embeddings = patches.flatten(2).transpose(1, 2)  

        #input embeds with class token and pos embed
        class_token = self.class_token.expand(len(input_images), -1, -1)
        input_embeddings = torch.cat((class_token, patch_embeddings), dim=1)
        input_embeddings += self.positional_embeddings

        att_maps = []
        encoded = input_embeddings
        for block in self.encoder_blocks:
            encoded,att_map = block(encoded, encoded, encoded)
            att_maps.append(att_map)

        #final class token
        class_output = encoded[:, 0]
        class_outputs = self.final_layer(torch.cat([class_output.real,class_output.imag],dim=-1))
        
        return class_outputs,att_maps
    
class ComplexPatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=512, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

        self.real_stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        self.imag_stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, patches):
        x_real = self.real_stem(patches)
        x_imag = self.imag_stem(patches)
        x_complex = torch.complex(x_real, x_imag) 
        return x_complex

class ComplexViTconv(nn.Module):
    def __init__(self, image_size, embed_dim=512, num_heads=8, num_blocks=6, num_classes=10, patch_size=16, bias=False,encoder_ff_hidden_dim=1024):
        super(ComplexViTconv, self).__init__()

        self.patch_embedding = ComplexPatchEmbedding(in_channels=3, embed_dim=embed_dim, patch_size=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim)) 
        self.positional_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        self.encoder_blocks = nn.ModuleList([ComplexEncoderBlock(embed_dim, num_heads, bias=bias,hidden_dim=encoder_ff_hidden_dim) for _ in range(num_blocks)])
        self.final_layer = nn.Linear(embed_dim*2,num_classes)

    def forward(self, input_images):
        patches = self.patch_embedding(input_images)
        patch_embeddings = patches.flatten(2).transpose(1, 2)  

        #input embeds with class token and pos embed
        class_token = self.class_token.expand(len(input_images), -1, -1)
        input_embeddings = torch.cat((class_token, patch_embeddings), dim=1)
        input_embeddings += self.positional_embeddings


        att_maps = []
        encoded = input_embeddings
        for block in self.encoder_blocks:
            encoded,att_map = block(encoded, encoded, encoded)
            att_maps.append(att_map)

        #final class token
        class_output = encoded[:, 0]
        class_outputs = self.final_layer(torch.cat([class_output.real,class_output.imag],dim=-1))
        
        return class_outputs,att_maps

def complex_he_init(tensor_shape,device):
    n_in = tensor_shape[1]
    scale = 1 / n_in

    modulus = np.random.rayleigh(scale=scale, size=tensor_shape)
    modulus = torch.tensor(modulus,dtype=torch.float32)
    phase = torch.rand(tensor_shape,dtype=torch.float32)*2*np.pi-np.pi

    real = modulus*torch.cos(phase)
    imag = modulus*torch.sin(phase)

    return torch.complex(real, imag).to(device)

def initialise_complex_weights(model,device):
    for _, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
            if module.weight.dtype == torch.complex64:
                module.weight.data = complex_he_init(tensor_shape=module.weight.shape,device=device)
