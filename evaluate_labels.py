"""Measure label accuracy of the class-conditional diffusion model.

1. Train a small CNN classifier on real MNIST (report its test accuracy).
2. Generate N images per digit with the diffusion model for several guidance weights.
3. Classify the generated images; accuracy = fraction where predicted == requested digit.

    python evaluate_labels.py --n_per_class 200 --guidances 0 1 2 4
"""
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from diffusion_mnist import Diffusion, UNet, device, NUM_CLASSES


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = self.drop(F.relu(self.fc1(x.flatten(1))))
        return self.fc2(x)


def train_classifier(epochs=3):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])  # same [-1,1] range
    train = DataLoader(datasets.MNIST("data", train=True, download=True, transform=tfm), batch_size=128, shuffle=True)
    test = DataLoader(datasets.MNIST("data", train=False, download=True, transform=tfm), batch_size=1000)
    clf = CNN().to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for _ in range(epochs):
        clf.train()
        for x, y in train:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(clf(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    clf.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test:
            correct += (clf(x.to(device)).argmax(1).cpu() == y).sum().item()
    acc = correct / len(test.dataset)
    print(f"Classifier test accuracy on real MNIST: {acc:.4f}", flush=True)
    return clf, acc


@torch.no_grad()
def evaluate(clf, model, diff, guidance, n_per_class, batch=500):
    labels = torch.arange(NUM_CLASSES).repeat_interleave(n_per_class)
    preds = []
    for i in range(0, len(labels), batch):
        imgs = diff.sample(model, labels[i:i + batch], guidance)
        preds.append(clf(imgs).argmax(1).cpu())
    preds = torch.cat(preds)
    per_class = [(preds[labels == c] == c).float().mean().item() for c in range(NUM_CLASSES)]
    return (preds == labels).float().mean().item(), per_class


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs_cond")
    p.add_argument("--n_per_class", type=int, default=200)
    p.add_argument("--guidances", type=float, nargs="+", default=[0, 1, 2, 4])
    p.add_argument("--timesteps", type=int, default=300)
    args = p.parse_args()

    torch.manual_seed(0)
    clf, clf_acc = train_classifier()
    model = UNet().to(device)
    model.load_state_dict(torch.load(f"{args.out}/model.pt", map_location=device))
    diff = Diffusion(args.timesteps)

    results = {"classifier_test_acc": clf_acc, "n_per_class": args.n_per_class, "guidance": {}}
    for w in args.guidances:
        acc, per_class = evaluate(clf, model, diff, w, args.n_per_class)
        results["guidance"][str(w)] = {"accuracy": acc, "per_class": per_class}
        print(f"guidance={w}: label accuracy={acc:.4f}  per-digit={[round(a, 2) for a in per_class]}", flush=True)

    with open(f"{args.out}/label_accuracy.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {args.out}/label_accuracy.json")


if __name__ == "__main__":
    main()
