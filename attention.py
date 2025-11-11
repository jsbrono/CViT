import torch
import torch.nn as nn

class RealSingleAttention(nn.Module):
    def __init__(self,embed_dim,output_dim,bias=False):
        super(RealSingleAttention, self).__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.queries = nn.Linear(embed_dim,output_dim,bias=bias)
        self.keys = nn.Linear(embed_dim,output_dim,bias=bias)
        self.values = nn.Linear(embed_dim,output_dim,bias=bias)
        self.d_model_sqrt_inv = 1/torch.sqrt(torch.tensor(output_dim, dtype=torch.float32))

    def forward(self,input_embeddings):
        #input_embeddings expected shape: (BATCH,SEQ_LEN,LEN(EMBEDDINGS))
        Q = self.queries(input_embeddings)
        K = self.keys(input_embeddings)
        V = self.values(input_embeddings)
        QK_scores = torch.matmul(Q, K.transpose(-2, -1))*self.d_model_sqrt_inv
        attention_weights = torch.softmax(QK_scores,dim=-1)
        attention = torch.matmul(attention_weights, V)
        return attention

class RealMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, bias=False):
        super(RealMultiHeadAttention, self).__init__()
        #embed_dim = d_model = dk = dv, for now
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.queries = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.keys = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.values = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.final_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.d_model_sqrt_inv = 1/torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        self.dropout = nn.Dropout(0.1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        #note: q==k==v in self attention, this is to allow for flexibility for e.g. encoder-decoder attention
        #input_embeddings expected shape: (BATCH,SEQ_LEN,LEN(EMBEDDINGS))

        #reshaping input embedding to be shape (BATCH,SEQ_LEN,NUM_HEADS,LEN(EMBEDDINGS)/NUM_HEADS)
        Q = self.queries(q)
        K = self.keys(k)
        V = self.values(v)

        batch, seq_len, _ = Q.shape

        #transposing to shape (batch,head,seq,head_dim)
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        QK_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.d_model_sqrt_inv
        attention_weights = torch.softmax(QK_scores, dim=-1)
        attention = torch.matmul(attention_weights, V)

        #attention is now shape (batch,head,seq,head_dim)
        #need to get to shape (batch,seq,embed_dim)
        attention = attention.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

        #final head linear map
        mha = self.final_linear(attention)
        mha = self.dropout(mha)

        return mha,attention_weights

class RealEncoderBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, hidden_dim=4096, bias=False):
        super(RealEncoderBlock, self).__init__()
        if hidden_dim is None:
            hidden_dim=4*embed_dim
        
        self.mha = RealMultiHeadAttention(embed_dim, num_heads, bias=bias)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.feedforward = nn.Sequential(nn.Linear(embed_dim, hidden_dim),nn.GELU(),nn.Linear(hidden_dim, embed_dim),nn.Dropout(0.1))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        #self attention only
        normed_input = self.layer_norm1(q)  
        mha,att_weights = self.mha(normed_input, normed_input, normed_input)
        residual_mha = q + mha  

        normed_mha = self.layer_norm2(residual_mha) 
        ff_out = self.feedforward(normed_mha)
        encoder_out = residual_mha + ff_out

        return encoder_out,att_weights

class RealViT(nn.Module):
    def __init__(self, image_size, embed_dim=512, num_heads=8, num_blocks=6, num_classes=10, patch_size=16, bias=False,encoder_ff_hidden_dim=1024):
        super(RealViT, self).__init__()

        self.patch_embedding = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positional_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        self.encoder_blocks = nn.ModuleList([RealEncoderBlock(embed_dim, num_heads, bias=bias,hidden_dim=encoder_ff_hidden_dim) for _ in range(num_blocks)])
        self.final_layer = nn.Linear(embed_dim, num_classes)

    def forward(self, input_images):
        patches = self.patch_embedding(input_images)
        patch_embeddings = patches.flatten(2).transpose(1, 2) 

        #input embeds with class token and pos embed
        class_token = self.class_token.expand(len(input_images), -1, -1)
        input_embeddings = torch.cat((class_token, patch_embeddings), dim=1)
        input_embeddings += self.positional_embeddings

        encoded = input_embeddings
        att_maps = []
        for block in self.encoder_blocks:
            encoded,att_weights = block(encoded, encoded, encoded)
            att_maps.append(att_weights)

        #final class token
        class_output = encoded[:, 0]
        class_outputs = self.final_layer(class_output)
        
        return class_outputs,att_maps

class PatchEmbedding(nn.Module):
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

    def forward(self, patches):
        x = self.real_stem(patches)
        return x

class RealViTconv(nn.Module):
    def __init__(self, image_size, embed_dim=512, num_heads=8, num_blocks=6, num_classes=10, patch_size=16, bias=False,encoder_ff_hidden_dim=1024):
        super(RealViTconv, self).__init__()

        # self.patch_embedding = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embedding = PatchEmbedding(in_channels=3, embed_dim=embed_dim, patch_size=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positional_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        self.encoder_blocks = nn.ModuleList([RealEncoderBlock(embed_dim, num_heads, bias=bias,hidden_dim=encoder_ff_hidden_dim) for _ in range(num_blocks)])
        self.final_layer = nn.Linear(embed_dim, num_classes)

    def forward(self, input_images):
        patches = self.patch_embedding(input_images)
        patch_embeddings = patches.flatten(2).transpose(1, 2) 

        #input embeds with class token and pos embed
        class_token = self.class_token.expand(len(input_images), -1, -1)
        input_embeddings = torch.cat((class_token, patch_embeddings), dim=1)
        input_embeddings += self.positional_embeddings

        encoded = input_embeddings
        att_maps = []
        for block in self.encoder_blocks:
            encoded,att_weights = block(encoded, encoded, encoded)
            att_maps.append(att_weights)

        #final class token
        class_output = encoded[:, 0]
        class_outputs = self.final_layer(class_output)
        
        return class_outputs,att_maps
