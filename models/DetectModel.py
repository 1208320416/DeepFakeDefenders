import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.1):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda f: rearrange(f, 'b n (h d) -> b h n d', h = h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.1):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv1 = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_qkv2 = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out1 = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.to_out2 = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x1, x2 = x[0], x[1]
        b, n, _, h = *x1.shape, self.heads
        qkv1 = self.to_qkv1(x1).chunk(3, dim = -1)
        qkv2 = self.to_qkv2(x2).chunk(3, dim = -1)
        q1, k1, v1 = map(lambda f: rearrange(f, 'b n (h d) -> b h n d', h = h), qkv1)
        q2, k2, v2 = map(lambda f: rearrange(f, 'b n (h d) -> b h n d', h = h), qkv2)
        dots1 = einsum('b h i d, b h j d -> b h i j', q2, k1) * self.scale
        attn1 = dots1.softmax(dim=-1)
        out1 = einsum('b h i j, b h j d -> b h i d', attn1, v1)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')
        out1 =  self.to_out1(out1)
        dots2 = einsum('b h i d, b h j d -> b h i j', q1, k2) * self.scale
        attn2 = dots2.softmax(dim=-1)
        out2 = einsum('b h i j, b h j d -> b h i d', attn2, v2)
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        out2 =  self.to_out2(out2)       
        return out1, out2


class CrossTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, CrossAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
            ]))

    def forward(self, x1, x2):
        for cross_attn, attn1, attn2, ff1, ff2 in self.layers:
            a1, a2 = cross_attn(torch.stack((x1, x2), 0))
            x1 += a1
            x2 += a2
            x1 = attn1(x1) + x1
            x2 = attn2(x2) + x2
            x1 = ff1(x1) + x1
            x2 = ff2(x2) + x2
        return x1, x2


class DetectModel(nn.Module):
    def __init__(self, **kwargs):
        super(DetectModel, self).__init__()
        self.image_to_patch_embedding = nn.Sequential(
            Rearrange('b f c (h p1) (w p2) -> b f (h w) (p1 p2 c)', p1 = 16, p2 = 16),
            nn.Linear(768, 768),
        )
        self.image_space_pos = nn.Parameter(torch.randn(1, 90, 197, 768))
        self.image_space_token = nn.Parameter(torch.randn(1, 1, 768))
        self.image_space_transformer = Transformer(768, 6, 4, 64, 1536, 0.2)
        self.image_temporal_pos = nn.Parameter(torch.randn(1, 11, 768))
        self.image_temporal_token = nn.Parameter(torch.randn(1, 1, 768))
        self.image_temporal_transformer = Transformer(768, 6, 4, 64, 1536, 0.2)

        self.audio_to_patch_embedding = nn.Sequential(
            Rearrange('b f c (h p1) (w p2) -> b f (h w) (p1 p2 c)', p1 = 57, p2 = 5),
            nn.Linear(570, 768),
        )
        self.audio_space_pos = nn.Parameter(torch.randn(1, 90, 19, 768))
        self.audio_space_token = nn.Parameter(torch.randn(1, 1, 768))
        self.audio_space_transformer = Transformer(768, 6, 4, 64, 1536, 0.2)
        self.audio_temporal_pos = nn.Parameter(torch.randn(1, 11, 768))
        self.audio_temporal_token = nn.Parameter(torch.randn(1, 1, 768))
        self.audio_temporal_transformer = Transformer(768, 6, 4, 64, 1536, 0.2)
 
        self.image_cross_token = nn.Parameter(torch.randn(1, 1, 768))
        self.audio_cross_token = nn.Parameter(torch.randn(1, 1, 768))
        self.cross_transformer = CrossTransformer(768, 8, 4, 64, 1536, 0.2)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(1536),
            nn.Linear(1536, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Linear(512, 2)
        )

    def forward(self, image, audio):
        #image (B, 90, 3, 224, 224)  audio (B, 90, 2, 513, 10)

        x = self.image_to_patch_embedding(image) 
        b, f, _, _ = x.shape
        # concat space CLS tokens
        image_cls_space_tokens = repeat(self.image_space_token, '() n d -> b f n d', b = b, f = f) # b,f,1,dim
        x = torch.cat((image_cls_space_tokens, x), dim = 2) # x:(b,f,196,768) -> (b,f,197,768)
        # add positional embeddingadd
        x += self.image_space_pos # (b,f,197,768)
        # cal space attention
        x = rearrange(x, 'b f n d -> (b f) n d') # (b*f,197,768)
        x = self.image_space_transformer(x) # (b*f,197,768)
        # select CLS token out of each frame
        x = rearrange(x[:, 0], '(b f) ... -> b f ...', b=b) # (b,f,768)
        # clip temporal
        x = rearrange(x, 'b (s t) d -> (b s) t d', s = 9, t = 10) # (b*s,t,768)
        #concat time CLS tokens
        image_cls_temporal_tokens = repeat(self.image_temporal_token, '() n d -> b n d', b = 9 * b) # (b*s,1,768)
        x = torch.cat((image_cls_temporal_tokens, x), dim=1)# x: (b*s,t+1,768)
        # add positional embeddingadd
        x += self.image_temporal_pos # (b*s,t+1,768)
        # cal time attention
        x = self.image_temporal_transformer(x) # x: (b*s,t+1,768)
        x = rearrange(x[:, 0], '(b s) ... -> b s ...', b = b) # (b,s,768)
        #concat cross CLS tokens
        image_cls_cross_tokens = repeat(self.image_cross_token, '() n d -> b n d', b = b) # (b,1,768)
        x = torch.cat((image_cls_cross_tokens, x), dim=1)# x: (b,s+1,768)

        y = self.audio_to_patch_embedding(audio) 
        # concat space CLS tokens
        audio_cls_space_tokens = repeat(self.audio_space_token, '() n d -> b f n d', b = b, f = f) # b,f,1,dim
        y = torch.cat((audio_cls_space_tokens, y), dim = 2) # y:(b,f,18,768) -> (b,f,19,768)
        # add positional embeddingadd
        y += self.audio_space_pos # (b,f,19,768)
        # cal space attention
        y = rearrange(y, 'b f n d -> (b f) n d') # (b*f,19,768)
        y = self.audio_space_transformer(y) # (b*f,19,768)
        # select CLS token out of each frame
        y = rearrange(y[:, 0], '(b f) ... -> b f ...', b=b) # (b,f,768)
        # clip temporal
        y = rearrange(y, 'b (s t) d -> (b s) t d', s = 9, t = 10) # (b*s,t,768)
        #concat time CLS tokens
        audio_cls_temporal_tokens = repeat(self.audio_temporal_token, '() n d -> b n d', b = 9 * b) # (b*s,1,768)
        y = torch.cat((audio_cls_temporal_tokens, y), dim=1)# y: (b*s,t+1,768)
        # add positional embeddingadd
        y += self.audio_temporal_pos # (b*s,t+1,768)
        # cal time attention
        y = self.audio_temporal_transformer(y) # y: (b*s,t+1,768)
        y = rearrange(y[:, 0], '(b s) ... -> b s ...', b = b) # (b,s,768)
        #concat cross CLS tokens
        audio_cls_cross_tokens = repeat(self.audio_cross_token, '() n d -> b n d', b = b) # (b,1,768)
        y = torch.cat((audio_cls_cross_tokens, y), dim=1)# x: (b,s+1,768)

        x_cross, y_cross = self.cross_transformer(x, y)
        
        z = torch.cat([x_cross[:, 0], y_cross[:, 0]], dim = -1)
        z = self.classifier(z)
        return z