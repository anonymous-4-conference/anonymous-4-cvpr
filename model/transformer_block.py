import torch.nn as nn
import torch
from model.Unet_block import UnetResBlock
from model.deform_conv import DeformConvPack

"""
These codes are developed and adapted from the following sources:
https://github.com/xmindflow/deformableLKA/
"""

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class _3d_deform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv3d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.deform_conv = DeformConvPack(in_channels=dim, out_channels=dim, kernel_size=(3,3,3), stride=1, padding=1)
        #print("Using single deformable layers.")
        self.conv1 = nn.Conv3d(dim, dim, 1)
 

    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = attn.contiguous()
        attn = self.deform_conv(attn)
        attn = self.conv1(attn)

        return u * attn

class _Attention3d_deform(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit =_3d_deform(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x, B, C, H, W, D):
        x = x.permute(0,2,1).reshape(B, C, H, W, D) # B N C --> B C N --> B C H W D 
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        return x




class TransformerBlock_3D(nn.Module):


    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        """
        Initially setting gamma to a small value close to zero (like 1e-6) allows for a gradual learning process. This scaled initialization ensures that during early training stages, the contribution of certain transformations (like attention outputs) is minimized. The idea is to start with a network that behaves similarly to an identity function (no significant transformations), then progressively learn stronger transformations as gamma is updated during training.
        """
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        
        
        self.epa_block = _Attention3d_deform(d_model=hidden_size)
        self.Unetconv = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.projconv = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))
        """
        1.Representation mixing: When positional encodings are added to token embeddings, the resulting vector is a mixture of both types of information. Each dimension of this vector contains both positional and semantic information.
        2.Learned interpretation: The network's subsequent layers (particularly the attention mechanisms and feed-forward networks) learn to interpret and utilize this mixed representation during training.
        3.Attention mechanism: The self-attention mechanism in transformers is particularly adept at teasing apart different aspects of the input. It can learn to attend differently to positional and semantic aspects of the input as needed for the task.
        """

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size)) # needs to be added to the embedding
            


    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        # import pdb
        # pdb.set_trace()
        
        if self.pos_embed is not None:
            x = x + self.pos_embed
        features = x + self.gamma * self.epa_block(self.norm(x), B, C, H , W , D)
        # import pdb
        # pdb.set_trace()
        attn_skip = features.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.Unetconv(attn_skip)
        x = attn_skip + self.projconv(attn)

        return x


