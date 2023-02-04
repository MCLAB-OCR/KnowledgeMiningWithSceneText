from torch import nn

from utils.config import cfg


def generate_head(self):
    dim1 = 768
    dim2 = 2048
    dim3 = 2048
    dim4 = 4096 * 2
    dim5 = 1024

    if cfg.head_type == 18:
        self.head_vision = nn.Sequential(
            nn.Linear(dim1, dim1),
            nn.BatchNorm1d(dim1),
            nn.GELU(),
        )
        self.head_text = nn.Sequential(
            nn.Linear(dim1, dim1),
            nn.BatchNorm1d(dim1),
            nn.GELU(),
        )
        last_dim = dim1 * 2
        self.head = nn.Sequential(nn.Linear(last_dim, self.cfg.NUM_CLASS), )
