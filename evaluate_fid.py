"""FID of generated MNIST digits vs. real MNIST test set, for several guidance weights.

Lower FID = closer to the real distribution (quality AND diversity). Since guidance trades
diversity for label fidelity, FID vs. guidance shows that trade-off.

    python evaluate_fid.py --n_per_class 500 --guidances 0 1 2 4
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets, transforms

from diffusion_mnist import Diffusion, UNet, device, NUM_CLASSES


def to_uint8_rgb(x):
    """[-1,1] grayscale (N,1,28,28) -> uint8 RGB (N,3,28,28); torchmetrics resizes to 299."""
    x = ((x.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
    return x.repeat(1, 3, 1, 1)


def real_loader(train, n=None):
    ds = datasets.MNIST("data", train=train, download=True,
                        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]))
    if n is not None:
        g = torch.Generator().manual_seed(0)
        ds = torch.utils.data.Subset(ds, torch.randperm(len(ds), generator=g)[:n].tolist())
    return DataLoader(ds, batch_size=500)


def new_fid():
    return FrechetInceptionDistance(feature=2048, normalize=False).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs_cond")
    p.add_argument("--n_per_class", type=int, default=500)
    p.add_argument("--guidances", type=float, nargs="+", default=[0, 1, 2, 4])
    p.add_argument("--timesteps", type=int, default=300)
    args = p.parse_args()
    torch.manual_seed(0)

    model = UNet().to(device)
    model.load_state_dict(torch.load(f"{args.out}/model.pt", map_location=device))
    diff = Diffusion(args.timesteps)
    n_gen = args.n_per_class * NUM_CLASSES

    # Reference statistics: the full 10k real test set
    ref_batches = [to_uint8_rgb(x).to(device) for x, _ in real_loader(train=False)]

    results = {"n_generated": n_gen, "reference": "MNIST test set (10000 images)", "fid": {}}

    # Floor: real train images (same count as generated) vs. test set
    fid = new_fid()
    for b in ref_batches:
        fid.update(b, real=True)
    for x, _ in real_loader(train=True, n=n_gen):
        fid.update(to_uint8_rgb(x).to(device), real=False)
    results["fid_real_train_vs_test"] = fid.compute().item()
    print(f"FID real train ({n_gen}) vs test: {results['fid_real_train_vs_test']:.2f}  (floor)", flush=True)

    labels = torch.arange(NUM_CLASSES).repeat_interleave(args.n_per_class)
    for w in args.guidances:
        fid = new_fid()
        for b in ref_batches:
            fid.update(b, real=True)
        for i in range(0, n_gen, 500):
            imgs = diff.sample(model, labels[i:i + 500], w)
            fid.update(to_uint8_rgb(imgs), real=False)
        results["fid"][str(w)] = fid.compute().item()
        print(f"guidance={w}: FID={results['fid'][str(w)]:.2f}", flush=True)

    with open(f"{args.out}/fid.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {args.out}/fid.json")


if __name__ == "__main__":
    main()
