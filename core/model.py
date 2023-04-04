import torch
import numpy as np


class KiloNerf(torch.nn.Module):
    def __init__(self, N, embedding_dim_pos=10, embedding_dim_direction=4, scene_scale=3):
        super(KiloNerf, self).__init__()

        # KiloNerf with Xavier initialization
        self.layer1_w = torch.nn.Parameter(torch.zeros((N, N, N, 63, 32)).uniform_(-np.sqrt(6. / 85), np.sqrt(6. / 85)))
        self.layer1_b = torch.nn.Parameter(torch.zeros((N, N, N, 1, 32)))
        self.layer2_w = torch.nn.Parameter(torch.zeros((N, N, N, 32, 33)).uniform_(-np.sqrt(6. / 64), np.sqrt(6. / 64)))
        self.layer2_b = torch.nn.Parameter(torch.zeros((N, N, N, 1, 33)))
        self.layer3_w = torch.nn.Parameter(torch.zeros((N, N, N, 32, 32)).uniform_(-np.sqrt(6. / 64), np.sqrt(6. / 64)))
        self.layer3_b = torch.nn.Parameter(torch.zeros((N, N, N, 1, 32)))
        self.layer4_w = torch.nn.Parameter(torch.zeros((N, N, N, 27 + 32, 32)).uniform_(-np.sqrt(6. / 64), np.sqrt(6. / 64)))
        self.layer4_b = torch.nn.Parameter(torch.zeros((N, N, N, 1, 32)))
        self.layer5_w = torch.nn.Parameter(torch.zeros((N, N, N, 32, 3)).uniform_(-np.sqrt(6. / 35), np.sqrt(6. / 35)))
        self.layer5_b = torch.nn.Parameter(torch.zeros((N, N, N, 1, 3)))

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.N = N
        self.scale = scene_scale

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, x, d):
        color = torch.zeros_like(x) # màu 
        sigma = torch.zeros((x.shape[0]), device=x.device) # mật độ

        mask = (x[:, 0].abs() < (self.scale / 2)) & (x[:, 1].abs() < (self.scale / 2)) & (x[:, 2].abs() < (self.scale / 2))
        idx = (x[mask] / (self.scale / self.N) + self.N / 2).long().clip(0, self.N - 1)

        emb_x = self.positional_encoding(x[mask], self.embedding_dim_pos)
        emb_d = self.positional_encoding(d[mask], self.embedding_dim_direction)

        # Implementation of the MLP architecture from Figure 2
        h = torch.relu(emb_x.unsqueeze(1) @ self.layer1_w[idx[:, 0], idx[:, 1], idx[:, 2]] + self.layer1_b[idx[:, 0], idx[:, 1], idx[:, 2]])
        h = torch.relu(h @ self.layer2_w[idx[:, 0], idx[:, 1], idx[:, 2]] + self.layer2_b[idx[:, 0], idx[:, 1], idx[:, 2]])
        h, density = h[:, :, :-1], h[:, :, -1]
        h = h @ self.layer3_w[idx[:, 0], idx[:, 1], idx[:, 2]] + self.layer3_b[idx[:, 0], idx[:, 1], idx[:, 2]]
        h = torch.relu(torch.cat((h, emb_d.unsqueeze(1)), dim=-1) @ self.layer4_w[idx[:, 0], idx[:, 1], idx[:, 2]] + self.layer4_b[idx[:, 0], idx[:, 1], idx[:, 2]])
        c = torch.sigmoid(h @ self.layer5_w[idx[:, 0], idx[:, 1], idx[:, 2]] + self.layer5_b[idx[:, 0], idx[:, 1], idx[:, 2]])
        color[mask] = c.squeeze(1)
        sigma[mask] = density.squeeze(1)
        return color, sigma