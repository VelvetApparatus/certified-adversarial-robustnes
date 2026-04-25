import torch
import torch.nn.functional as F

from src.adversaries.common import Adversary


class StAdv(Adversary):
    """

    Link: https://arxiv.org/abs/1801.02612
    """

    def __init__(self, steps, alpha, tau, loss_fn, targeted=False):
        super().__init__(
            name="StAdv",
            loss_fn=loss_fn,
            params={
                "steps": steps,
                "alpha": alpha,
                "tau": tau,
                "targeted": targeted,
            }
        )
        self.steps = steps
        self.alpha = alpha
        self.tau = tau
        self.targeted = targeted

    def _gen(self, model, X, y, y_target=None):
        X = X.detach().clone()

        B, C, H, W = X.shape
        flow = torch.zeros(B, H, W, 2, device=X.device)

        for _ in range(self.steps):
            flow.requires_grad_(True)

            X_adv = warp_image(X, flow)
            logits = model(X_adv)

            if self.targeted:
                if y_target is None:
                    raise ValueError("y_target is required for targeted StAdv")
                adv_loss = -self.loss_fn(logits, y_target)
            else:
                adv_loss = self.loss_fn(logits, y)

            reg_loss = flow_loss(flow)
            total_loss = adv_loss + self.tau * reg_loss

            grad = torch.autograd.grad(total_loss, flow)[0]

            if self.targeted:
                flow = flow.detach() - self.alpha * grad
            else:
                flow = flow.detach() + self.alpha * grad

        X_adv = warp_image(X, flow)
        X_adv = torch.clamp(X_adv, 0.0, 1.0)
        return X_adv.detach()


def make_base_grid(x):
    B, C, H, W = x.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=x.device),
        torch.linspace(-1, 1, W, device=x.device),
        indexing="ij"
    )
    grid = torch.stack([xx, yy], dim=-1)  # [H, W, 2]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]
    return grid


def warp_image(x, flow):
    B, C, H, W = x.shape
    base_grid = make_base_grid(x)

    flow_x = flow[..., 0] * (2.0 / max(W - 1, 1))
    flow_y = flow[..., 1] * (2.0 / max(H - 1, 1))
    flow_norm = torch.stack([flow_x, flow_y], dim=-1)

    sampling_grid = base_grid + flow_norm
    x_warped = F.grid_sample(
        x,
        sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return x_warped


def flow_loss(flow):
    dx = flow[:, 1:, :, :] - flow[:, :-1, :, :]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    return dx.pow(2).mean() + dy.pow(2).mean()
