from torch import nn
import torch
import math


class CondBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(emb_dim, out_ch)
        self.cond_mlp = nn.Linear(emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, t_emb, c_emb):
        h = self.relu(self.conv1(x))
        time_bias = self.relu(self.time_mlp(t_emb))[(...,) + (None,) * 2]
        cond_bias = self.relu(self.cond_mlp(c_emb))[(...,) + (None,) * 2]
        h = h + time_bias + cond_bias
        h = self.relu(self.conv2(h))
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalUnet(nn.Module):
    def __init__(self, num_classes, emb_dim=64):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )
        self.class_embed = nn.Embedding(num_classes, emb_dim)

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
        self.downs = nn.ModuleList(
            [CondBlock(down_channels[i], down_channels[i + 1], emb_dim) for i in range(len(down_channels) - 1)]
        )
        self.ups = nn.ModuleList(
            [CondBlock(up_channels[i], up_channels[i + 1], emb_dim, up=True) for i in range(len(up_channels) - 1)]
        )
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep, class_ids):
        t_emb = self.time_mlp(timestep)
        c_emb = self.class_embed(class_ids)
        x = self.conv0(x)

        residual_inputs = []
        for down in self.downs:
            x = down(x, t_emb, c_emb)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t_emb, c_emb)

        return self.output(x)
