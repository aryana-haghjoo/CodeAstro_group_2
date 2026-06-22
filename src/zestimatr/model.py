import torch
import torch.nn as nn


class ZHead1D(nn.Module):
    """
    Continuous redshift head (high-res oracle).
    Input: (B, 1, L_high) -- normalized high-resolution flux.
    Output: mu_z, log_var_z  (both (B,))
    """
    def __init__(self, in_channels=1, hidden_dim=64, num_blocks=4, dropout=0.1):
        super().__init__()

        self.in_channels = in_channels
        self.flux_net = self._make_conv_blocks(in_channels, hidden_dim, num_blocks, dropout)

        self.mu = nn.Linear(hidden_dim, 1)
        self.log_var = nn.Linear(hidden_dim, 1)

        nn.init.constant_(self.log_var.weight, 0.0)
        nn.init.constant_(self.log_var.bias, -2.0)

    def _make_conv_blocks(self, in_ch, out_ch, num_blocks, dropout):
        layers = []
        c = in_ch
        for i in range(num_blocks):
            layers += [
                nn.Conv1d(c, out_ch, kernel_size=7, padding=3, bias=True),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            c = out_ch
        return nn.Sequential(*layers)

    def forward(self, x):
        h = self.flux_net(x)   # (B, hidden_dim, L_high)
        h = h.mean(dim=-1)     # (B, hidden_dim)

        mu = self.mu(h).squeeze(-1)           # (B,)
        log_var = self.log_var(h).squeeze(-1)  # (B,)

        return mu, log_var


def heteroscedastic_nll(mu, log_var, y, var_floor=1e-6):
    var = torch.exp(log_var).clamp_min(var_floor)
    return 0.5 * (torch.log(var) + (y - mu) ** 2 / var).mean()
