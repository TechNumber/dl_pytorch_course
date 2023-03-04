import torch
import math
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        num_patches = (img_size[0] // patch_size[0]) * (
                img_size[1] // patch_size[1]
        )

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.patch_embeddings = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)

    def forward(self, image):
        B, C, H, W = image.shape
        assert H == self.img_size[0] and W == self.img_size[1]

        patches = self.patch_embeddings(image).flatten(2).transpose(1, 2)
        return patches


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=3072, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        # TODO: изменить sequential на обычную сеть

        # Linear Layers
        self.seq = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, out_features if out_features else in_features),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., out_drop=0.):
        super().__init__()
        self.num_heads = num_heads  # Кол-во "голов" MHA
        # dim - длина эмбеддинга
        self.head_dim = dim // num_heads  # Длина запроса, ключа, и значения
        self.scale = self.head_dim ** -0.5  # Коэффициент нормализации коэффициентов значимости

        self.qkv = nn.Linear(in_features=dim, out_features=3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out = nn.Linear(in_features=dim, out_features=dim)
        self.out_drop = nn.Dropout(out_drop)

    def forward(self, x):
        # Attention
        x = self.qkv(x)
        x = self.attn_drop(x)  # TODO: выключить или переставить в другое место
        # TODO: запустить с dropout и без
        x = x.view(x.shape[0], -1, 3, self.num_heads, self.head_dim)

        q, k, v = x.transpose(-2, -4).unbind(dim=-3)
        x = nn.functional.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1) @ v
        #         x = x.transpose(-2, -3).reshape(B, -1, self.head_dim * self.num_heads)
        x = x.transpose(-2, -3).flatten(start_dim=-2)  # TODO: перепроверить

        x = self.out(x)
        x = self.out_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        # Normalization
        self.input_norm = norm_layer(dim)

        # Attention
        self.mha = Attention(dim, num_heads, qkv_bias, attn_drop=attn_drop, out_drop=drop_rate)

        # Dropout
        self.drop = nn.Dropout(drop_path)

        # Normalization
        self.out_norm = norm_layer(dim)

        # MLP
        self.mlp = MLP(dim, math.ceil(dim * mlp_ratio), dim, act_layer=act_layer, drop=drop_rate)

    def forward(self, x):
        # Attetnion
        x_mha = self.input_norm(x)  # TODO: x + ...
        x_mha = self.mha(x_mha)
        x_mha = self.drop(x_mha)
        x_mha = x_mha + x

        # MLP
        x_mlp = self.out_norm(x_mha)
        x_mlp = self.mlp(x_mlp)
        x_mlp = self.drop(x_mlp)
        x = x_mlp + x_mha

        return x


class Transformer(nn.Module):
    def __init__(self, depth, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop, drop_path, act_layer, norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def get_positional_encoding(n_embed, embed_dim):
    pos_enc = torch.zeros((1, n_embed, embed_dim))
    posits = torch.arange(0, n_embed).view(-1, 1)
    #     freqs = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
    freqs = 1 / 10000 ** (2 * torch.arange(0, embed_dim, 2) / embed_dim)

    pos_enc[:, :, 0::2] = torch.sin(posits * freqs)
    pos_enc[:, :, 1::2] = torch.cos(posits * freqs)

    return pos_enc


class ViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, drop_rate=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        # Присвоение переменных
        assert (embed_dim % num_heads == 0)

        # Path Embeddings, CLS Token, Position Encoding
        self.patch_embeddings = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim))
        self.pos_encoding = nn.Parameter(torch.rand(1, self.patch_embeddings.num_patches + 1, embed_dim))
        # self.pos_encoding = nn.Parameter(get_positional_encoding(self.patch_embeddings.num_patches + 1, embed_dim),
        #                                  requires_grad=False)  # TODO: сделать обучаемыми, заменить на обычные веса
        print(self.pos_encoding.shape)

        # self.pos_drop = nn.Dropout(p=drop)  # TODO: добавить dropout
        # Transformer Encoder
        self.transformer = Transformer(depth, embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate,
                                       attn_drop, drop_path, act_layer, norm_layer)
        # Classifier
        self.classifier = MLP(embed_dim, math.ceil(embed_dim * mlp_ratio), num_classes, act_layer, drop_rate)
        # self.norm = nn.LayerNorm(embed_dim)  # TODO: обычная полносвязная сеть вместо MLP

    def forward(self, x):
        B = x.shape[0]

        # Path Embeddings, CLS Token, Position Encoding
        x = self.patch_embeddings(x)

        cls_token = self.cls_token.expand(B, -1, -1)
        #         print(cls_token.shape, x.shape)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_encoding
        #         print(x.shape)
        # x = self.pos_drop(x)  # TODO: добавить dropout

        # Transformer Encoder
        x = self.transformer(x)

        # Classifier
        x = self.classifier(x[:, 0])

        return x
