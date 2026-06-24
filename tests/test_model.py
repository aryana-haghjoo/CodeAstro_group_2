import torch
import pytest

from zestimatr.model import ZHead1D, ResConvBlock, heteroscedastic_nll


class TestResConvBlock:

    def test_output_shape_matches_input(self):
        block = ResConvBlock(channels=32, kernel_size=7)
        x = torch.randn(4, 32, 100)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        block = ResConvBlock(channels=16)
        block.eval()
        x = torch.zeros(2, 16, 50)
        out = block(x)
        assert out.shape == x.shape


class TestZHead1D:

    def test_output_shapes(self):
        model = ZHead1D(in_channels=1, hidden_dim=32, num_blocks=2)
        x = torch.randn(8, 1, 2500)
        mu, log_var = model(x)
        assert mu.shape == (8,)
        assert log_var.shape == (8,)

    def test_single_sample(self):
        model = ZHead1D(in_channels=1, hidden_dim=16, num_blocks=1)
        x = torch.randn(1, 1, 500)
        mu, log_var = model(x)
        assert mu.shape == (1,)
        assert log_var.shape == (1,)

    def test_different_sequence_lengths(self):
        model = ZHead1D(in_channels=1, hidden_dim=16, num_blocks=2)
        for length in [100, 500, 2500]:
            x = torch.randn(2, 1, length)
            mu, log_var = model(x)
            assert mu.shape == (2,)
            assert log_var.shape == (2,)

    def test_deterministic_in_eval(self):
        model = ZHead1D(in_channels=1, hidden_dim=16, num_blocks=2)
        model.eval()
        x = torch.randn(4, 1, 200)
        mu1, lv1 = model(x)
        mu2, lv2 = model(x)
        assert torch.allclose(mu1, mu2)
        assert torch.allclose(lv1, lv2)

    def test_gradients_flow(self):
        model = ZHead1D(in_channels=1, hidden_dim=16, num_blocks=2)
        x = torch.randn(2, 1, 200)
        mu, log_var = model(x)
        loss = mu.sum() + log_var.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestHeteroscedasticNLL:

    def test_perfect_prediction_low_loss(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        mu = y.clone()
        log_var = torch.zeros(3)
        loss = heteroscedastic_nll(mu, log_var, y)
        assert loss.item() < 1.0

    def test_high_variance_reduces_penalty(self):
        y = torch.tensor([1.0, 2.0])
        mu = torch.tensor([5.0, 10.0])
        loss_low_var = heteroscedastic_nll(mu, torch.zeros(2), y)
        loss_high_var = heteroscedastic_nll(mu, torch.full((2,), 6.0), y)
        assert loss_high_var < loss_low_var

    def test_returns_scalar(self):
        loss = heteroscedastic_nll(
            torch.randn(10), torch.randn(10), torch.randn(10)
        )
        assert loss.dim() == 0

    def test_var_floor(self):
        loss = heteroscedastic_nll(
            torch.zeros(5), torch.full((5,), -100.0), torch.zeros(5),
            var_floor=1e-6,
        )
        assert torch.isfinite(loss)
