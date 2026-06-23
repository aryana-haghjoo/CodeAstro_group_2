import torch
import torch.nn as nn


class ResConvBlock(nn.Module):
    """Conv1d block with a residual connection."""
    def __init__(self, channels, kernel_size=7, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class ZHead1D(nn.Module):
    """
    Continuous redshift head (high-res oracle).
    Input: (B, 1, L_high) -- normalized high-resolution flux.
    Output: mu_z, log_var_z  (both (B,))
    """
    def __init__(self, in_channels=1, hidden_dim=128, num_blocks=6,
                 dropout=0.1):
        super().__init__()

        self.in_channels = in_channels

        # Project input channels to hidden_dim
        self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=7,
                                    padding=3, bias=True)

        # Residual conv blocks
        self.flux_net = nn.Sequential(
            *[ResConvBlock(hidden_dim, kernel_size=7, dropout=dropout)
              for _ in range(num_blocks)]
        )

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden_dim, 1)
        self.log_var = nn.Linear(hidden_dim, 1)

        nn.init.constant_(self.log_var.weight, 0.0)
        nn.init.constant_(self.log_var.bias, -2.0)

    def forward(self, x):
        h = self.input_proj(x)  # (B, hidden_dim, L_high)
        h = self.flux_net(h)    # (B, hidden_dim, L_high)
        h = h.mean(dim=-1)      # (B, hidden_dim)

        h = self.head(h)        # (B, hidden_dim)
        mu = self.mu(h).squeeze(-1)           # (B,)
        log_var = self.log_var(h).squeeze(-1)  # (B,)

        return mu, log_var


def heteroscedastic_nll(mu, log_var, y, var_floor=1e-6):
    var = torch.exp(log_var).clamp_min(var_floor)
    return 0.5 * (torch.log(var) + (y - mu) ** 2 / var).mean()
