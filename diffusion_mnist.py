"""Class-conditional DDPM on MNIST with PyTorch (classifier-free guidance).

Train:
    python diffusion_mnist.py --epochs 20
Generate chosen digits from a trained model:
    python diffusion_mnist.py --generate --digits 3 7 --n 16 --guidance 2.0
"""
import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
NULL_CLASS = NUM_CLASSES  # extra "no label" token used for unconditional pass


# ---------------------------------------------------------------- noise schedule
class Diffusion:
    def __init__(self, timesteps=300, beta_start=1e-4, beta_end=0.02):
        self.T = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise):
        """Forward process: x_t = sqrt(ab)*x0 + sqrt(1-ab)*noise"""
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, labels, guidance=2.0, size=28):
        """Reverse process with classifier-free guidance.

        labels: LongTensor (n,) of digits 0-9.
        guidance w: eps = (1+w)*eps_cond - w*eps_uncond  (w=0 -> plain conditional)
        """
        model.eval()
        n = labels.size(0)
        labels = labels.to(device)
        null = torch.full_like(labels, NULL_CLASS)
        x = torch.randn(n, 1, size, size, device=device)
        for t in reversed(range(self.T)):
            tt = torch.full((n,), t, device=device, dtype=torch.long)
            eps_c = model(x, tt, labels)
            if guidance != 0:
                eps_u = model(x, tt, null)
                eps = (1 + guidance) * eps_c - guidance * eps_u
            else:
                eps = eps_c
            a, ab, b = self.alphas[t], self.alpha_bar[t], self.betas[t]
            mean = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
            x = mean + b.sqrt() * torch.randn_like(x) if t > 0 else mean
        return x.clamp(-1, 1)


# ---------------------------------------------------------------- model (small U-Net)
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim * 4))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.t_proj(cond_emb)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        t_dim = base * 4
        self.time_emb = TimeEmbedding(base)
        self.label_emb = nn.Embedding(NUM_CLASSES + 1, t_dim)  # +1 = null class
        self.inp = nn.Conv2d(1, base, 3, padding=1)
        self.down1 = Block(base, base, t_dim)            # 28x28
        self.down2 = Block(base, base * 2, t_dim)        # 14x14
        self.mid = Block(base * 2, base * 2, t_dim)      # 7x7
        self.up2 = Block(base * 4, base, t_dim)          # 14x14 (concat skip)
        self.up1 = Block(base * 2, base, t_dim)          # 28x28 (concat skip)
        self.out = nn.Conv2d(base, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, t, y):
        te = self.time_emb(t) + self.label_emb(y)        # time + class conditioning
        x = self.inp(x)
        d1 = self.down1(x, te)
        d2 = self.down2(self.pool(d1), te)
        m = self.mid(self.pool(d2), te)
        u2 = self.up2(torch.cat([self.upsample(m), d2], 1), te)
        u1 = self.up1(torch.cat([self.upsample(u2), d1], 1), te)
        return self.out(u1)


# ---------------------------------------------------------------- training / generation
def train(args):
    os.makedirs(args.out, exist_ok=True)
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])  # -> [-1, 1]
    ds = datasets.MNIST("data", train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    diff = Diffusion(args.timesteps)
    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"device={device}, params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    grid_labels = torch.arange(NUM_CLASSES).repeat_interleave(1).repeat(10)  # rows of 0..9
    losses = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x0, y in loader:
            x0, y = x0.to(device), y.to(device)
            # classifier-free guidance: randomly drop the label
            drop = torch.rand(y.size(0), device=device) < args.p_uncond
            y = torch.where(drop, torch.full_like(y, NULL_CLASS), y)
            t = torch.randint(0, diff.T, (x0.size(0),), device=device)
            noise = torch.randn_like(x0)
            xt = diff.q_sample(x0, t, noise)
            loss = F.mse_loss(model(xt, t, y), noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        losses.append(total / len(loader))
        print(f"epoch {epoch}/{args.epochs}  loss={losses[-1]:.4f}", flush=True)

        if epoch % 5 == 0 or epoch == args.epochs:
            imgs = (diff.sample(model, grid_labels, args.guidance) + 1) / 2
            save_image(imgs, f"{args.out}/sample_epoch{epoch}.png", nrow=NUM_CLASSES)  # each row = 0..9
            torch.save(model.state_dict(), f"{args.out}/model.pt")

    with open(f"{args.out}/losses.csv", "w") as f:
        f.write("epoch,loss\n")
        for i, l in enumerate(losses, 1):
            f.write(f"{i},{l:.6f}\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(losses) + 1), losses, marker="o")
        plt.xlabel("epoch"); plt.ylabel("MSE loss"); plt.title("Training loss"); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f"{args.out}/loss_curve.png", dpi=120)
    except ImportError:
        pass
    print(f"Done. Samples and weights saved in '{args.out}/'")


def generate(args):
    model = UNet().to(device)
    model.load_state_dict(torch.load(f"{args.out}/model.pt", map_location=device))
    diff = Diffusion(args.timesteps)
    labels = torch.tensor(args.digits).repeat_interleave(args.n)  # n images per requested digit
    imgs = (diff.sample(model, labels, args.guidance) + 1) / 2
    name = "".join(map(str, args.digits))
    path = f"{args.out}/generated_{name}.png"
    save_image(imgs, path, nrow=args.n)  # one row per digit
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=300)
    p.add_argument("--p_uncond", type=float, default=0.1, help="prob. of dropping the label in training")
    p.add_argument("--guidance", type=float, default=2.0, help="classifier-free guidance weight")
    p.add_argument("--out", default="outputs_cond")
    p.add_argument("--generate", action="store_true", help="generate from a trained model instead of training")
    p.add_argument("--digits", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--n", type=int, default=10, help="images per digit when generating")
    args = p.parse_args()
    generate(args) if args.generate else train(args)


if __name__ == "__main__":
    main()
