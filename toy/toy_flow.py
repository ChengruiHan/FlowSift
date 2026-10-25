import math

import torch
from torch import nn

if __package__:
    from toy.visualization import save_sample_comparison
else:
    # Also support: python toy/toy_flow.py
    from visualization import save_sample_comparison

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = 2
CENTERS = 4.0 * torch.tensor(
    [[math.cos(i * math.pi / 4), math.sin(i * math.pi / 4)] for i in range(8)]
)


def sample_data(n: int) -> torch.Tensor:
    idx = torch.randint(0, 8, (n,))
    return CENTERS[idx] + 0.3 * torch.randn(n, D)


class VNet(nn.Module):
    """x:(B, D) + t:(B, 1) -> y:(B, D)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D + 1, 128),  # D + 1 (x + t)
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, D),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, D), t: (B, 1)
        xt = torch.cat([x, t], dim=1)  # (B, D + 1)
        return self.net(xt)  # (B, D)


def train_cfm(
    model: nn.Module,
    steps: int = 5000,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> list[float]:
    model.train()
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []

    for step in range(steps):
        a = sample_data(batch_size).to(device)
        z = torch.randn(batch_size, D, device=device)
        t = torch.rand(batch_size, 1, device=device)

        x_t = (1 - t) * a + t * z
        target = z - a
        y = model(x_t, t)
        loss = ((y - target) ** 2).sum(-1).mean()

        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"Step {step}: loss={loss.item():.4f}")

    return losses


@torch.no_grad()
def ode_sample(
    model: nn.Module,
    n: int = 512,
    steps: int = 100,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrate the learned ODE from t=1 to t=0 with explicit Euler."""
    if n <= 0:
        raise ValueError("n must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")

    model.eval()
    device = next(model.parameters()).device

    if initial_noise is None:
        x = torch.randn(n, D, device=device)
    else:
        if initial_noise.shape != (n, D):
            raise ValueError(
                f"initial_noise must have shape {(n, D)}, "
                f"got {tuple(initial_noise.shape)}"
            )
        x = initial_noise.detach().clone().to(device)

    dt = -1 / steps

    for step in range(steps):
        time = 1 + step * dt
        t = torch.full(
            (n, 1),
            time,
            dtype=x.dtype,
            device=device,
        )
        velocity = model(x, t)
        x = x + velocity * dt

    return x


def main() -> None:
    torch.manual_seed(0)

    print(f"Using device: {DEVICE}")

    # Train the model
    model = VNet().to(DEVICE)
    losses = train_cfm(model)

    # Keep the trained velocity field fixed and vary only Euler integration accuracy.
    target_samples = sample_data(5000)
    generated_by_steps = {
        steps: ode_sample(model, n=5000, steps=steps)
        for steps in (2, 10, 50)
    }
    panels = {"Target: 8 Gaussians": target_samples}
    panels.update(
        {
            f"Generated: Euler N={steps}": samples
            for steps, samples in generated_by_steps.items()
        }
    )
    output_path = save_sample_comparison(
        panels=panels,
        losses=losses,
        output_path="results/day1_ode.png",
    )

    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss:   {losses[-1]:.6f}")
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
